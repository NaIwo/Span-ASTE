from ASTE.utils import config
from ASTE.dataset.reader import Batch
from ASTE.dataset.domain.const import ChunkCode
from ASTE.dataset.domain.sentence import Sentence
from ASTE.aste.losses import DiceLoss
from ASTE.aste.tools.metrics import Metric, get_selected_metrics
from ASTE.aste.models import ModelOutput, ModelLoss, ModelMetric, BaseModel

import torch
from typing import List, Tuple
from functools import lru_cache
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

START_TAG: int = 3
STOP_TAG: int = 4


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, torch.argmax(vec, dim=1)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class ChunkerModel(BaseModel):
    def __init__(self, input_dim: int, model_name: str = 'Chunker Model'):
        super(ChunkerModel, self).__init__(model_name)
        self.metrics: Metric = Metric(name='Chunker', metrics=get_selected_metrics(multiclass=True, num_classes=3),
                                      ignore_index=ChunkCode.NOT_RELEVANT).to(config['general']['device'])

        self.input_dim: int = input_dim

        self.tagset_size: int = 3 + 2  # B, I, O, START, STOP

        self.lstm = torch.nn.LSTM(input_dim, input_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.linear_layer = torch.nn.Linear(input_dim, input_dim // 2)
        self.final_layer = torch.nn.Linear(input_dim // 2, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = torch.nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size)).to(config['general']['device'])

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[START_TAG, :] = -10000
        self.transitions.data[:, STOP_TAG] = -10000

    def forward(self, data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        features: torch.Tensor = self.get_features(data, mask)
        return features

    def get_features(self, data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths: torch.Tensor = torch.sum(mask, dim=1).cpu()
        hidden: Tuple[torch.Tensor, torch.Tensor] = self.init_hidden(self.input_dim, len(lengths))

        embeddings = pack_padded_sequence(data, lengths, batch_first=True)
        context, _ = self.lstm(embeddings, hidden)
        context, _ = pad_packed_sequence(context, batch_first=True)

        context = self.linear_layer(context)
        return self.final_layer(context)

    @staticmethod
    def init_hidden(size: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.randn(2, batch_size, size // 2).to(config['general']['device']),
                torch.randn(2, batch_size, size // 2).to(config['general']['device']))

    def get_spans(self, batch: Batch, features: torch.Tensor) -> List[torch.Tensor]:
        results: List[torch.Tensor] = list()

        for sample, s_features in zip(batch, features):
            results.append(self.get_span_from_single_sample(sample, s_features))

        return results

    def get_span_from_single_sample(self, sample: Batch, features: torch.Tensor) -> torch.Tensor:
        length: torch.Tensor = torch.sum(sample.mask, dim=1).cpu()
        best_path: torch.Tensor = self._viterbi(features[:length])
        offset: int = sample.sentence_obj[0].encoder.offset
        best_path[:offset] = ChunkCode.NOT_SPLIT
        best_path[length - offset:] = ChunkCode.NOT_SPLIT
        return self.get_spans_from_sequence(best_path)

    @staticmethod
    def get_spans_from_sequence(seq: torch.Tensor) -> torch.Tensor:
        begins: torch.Tensor = torch.where(seq == ChunkCode.BEGIN_SPLIT)[0]
        begins = torch.cat((begins, torch.tensor([len(seq)], device=config['general']['device'])))
        results: List = list()

        idx: int
        b_idx: torch.Tensor
        end_idx: int
        for idx, b_idx in enumerate(begins[:-1]):
            s: torch.Tensor = seq[b_idx:begins[idx + 1]]
            if ChunkCode.NOT_SPLIT in s:
                end_idx = torch.where(s == ChunkCode.NOT_SPLIT)[0][0]
                end_idx += b_idx - 1
            else:
                end_idx = begins[idx + 1] - 1
            results.append([b_idx, end_idx])
        if not results:
            results.append([0, len(seq) - 1])
        return torch.tensor(results).to(config['general']['device'])

    @lru_cache(maxsize=None)
    def _viterbi(self, features: torch.Tensor) -> torch.Tensor:
        back_pointers: List = []
        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.).to(config['general']['device'])
        init_vvars[0][START_TAG] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in features:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = torch.argmax(next_tag_var, dim=1)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            back_pointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[STOP_TAG]
        best_tag_id = torch.argmax(terminal_var, dim=1)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(back_pointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == START_TAG  # Sanity check
        best_path.reverse()
        best_path = torch.cat(best_path)
        return best_path

    def get_loss(self, model_out: ModelOutput) -> ModelLoss:
        loss: List = list()
        for features, sample in zip(model_out.chunker_output, model_out.batch):
            length: torch.Tensor = torch.sum(sample.mask, dim=1).cpu()
            forward_score: torch.Tensor = self._forward_alg(features[:length])
            gold_score: torch.Tensor = self._score_sentence(features[:length], sample.chunk_label[:length][0])
            loss.append(forward_score - gold_score)
        loss: torch.Tensor = torch.tensor(loss).to(config['general']['device'])
        return ModelLoss(chunker_loss=torch.mean(loss))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.).to(config['general']['device'])
        # START_TAG has all of the score.
        init_alphas[0][START_TAG] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[STOP_TAG]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).to(config['general']['device'])
        tags = torch.cat([torch.tensor([START_TAG], dtype=torch.long).to(config['general']['device']), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[STOP_TAG, tags[-1]]
        return score

    def update_metrics(self, model_out: ModelOutput) -> None:
        for feature, label in zip(model_out.chunker_output, model_out.batch.chunk_label):
            self.metrics(
                self._viterbi(feature),
                label.view([-1])
            )

    def get_metrics(self) -> ModelMetric:
        return ModelMetric(chunker_metric=self.metrics.compute())

    def reset_metrics(self) -> None:
        self.metrics.reset()

