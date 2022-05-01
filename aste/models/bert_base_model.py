from .base_model import BaseModel
from .outputs import ModelOutput
from ASTE.utils import config
from ASTE.aste.model_elements.embeddings.bert_embeddings import Bert
from ASTE.aste.model_elements.span_aggregators import BaseAggregator, EndPointAggregator
from ASTE.dataset.reader import Batch
from ASTE.dataset.domain.const import ChunkCode

import torch
from typing import Optional, List


class BertBaseModel(BaseModel):
    def __init__(self, model_name='Bert Base Model'):
        super(BertBaseModel, self).__init__(model_name)
        self.embeddings_layer: Bert = Bert()
        self.chunker: BaseModel = ChunkerModel()
        self.aggregator: BaseAggregator = EndPointAggregator()

    def forward(self, batch: Batch) -> ModelOutput:
        embeddings: torch.Tensor = self.embeddings_layer(batch.sentence, batch.mask)
        chunker_output: torch.Tensor = self.chunker(embeddings)
        predicted_spans: List[torch.Tensor] = self.chunker.get_spans_from_last_prediction(batch)
        aggregated_embeddings: torch.Tensor = self.aggregator.aggregate(embeddings, predicted_spans)

        return ModelOutput(chunker_output=chunker_output,
                           predicted_spans=predicted_spans)


class ChunkerModel(BaseModel):
    def __init__(self, model_name: str = 'Chunker Model'):
        super(ChunkerModel, self).__init__(model_name)
        self.last_output: Optional[torch.Tensor] = None
        self.dropout = torch.nn.Dropout(0.1)
        self.linear_layer_1 = torch.nn.Linear(config['encoder']['bert']['embedding-dimension'], 100)
        self.linear_layer_2 = torch.nn.Linear(100, 10)
        self.final_layer = torch.nn.Linear(10, 2)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        layer: torch.nn.Linear
        for layer in [self.linear_layer_1, self.linear_layer_2]:
            data = layer(data)
            data = torch.relu(data)
            data = self.dropout(data)
        data = self.final_layer(data)
        self.last_output = self.softmax(data)
        return self.last_output

    def get_spans_from_last_prediction(self, batch) -> List[torch.Tensor]:
        if self.last_output is None:
            raise Exception('Due to the fact that spans are taken based on the last prediction, '
                            'model has to make prediction first.')
        results: List[torch.Tensor] = list()
        offset: int = batch.sentence_obj[0].encoder.offset
        predictions: torch.Tensor = torch.argmax(self.last_output, dim=-1)

        sample: Batch
        prediction: torch.Tensor
        for sample, prediction in zip(batch, predictions):
            prediction = prediction[:sample.sentence_obj[0].encoded_sentence_length]
            sub_mask: torch.Tensor = sample.sub_words_mask[0][:sample.sentence_obj[0].encoded_sentence_length]
            # Prediction help - if a token consists of several sub-tokens, we certainly do not split in those sub-tokens
            prediction = torch.where(sub_mask.bool(), prediction, int(ChunkCode.NOT_SPLIT))
            max_token: int = len(prediction)
            prediction = torch.where(prediction)[0]
            # Start and end of spans are the same as start and end of sentence
            prediction = torch.nn.functional.pad(prediction, [1, 0], mode='constant', value=offset)
            prediction = torch.nn.functional.pad(prediction, [0, 1], mode='constant', value=max_token - offset)
            predicted_spans: torch.Tensor = prediction.unfold(0, 2, 1)
            # Because we perform split ->before<- selected word.
            predicted_spans[:, 1] -= 1
            # This deletion is due to offset in padding. Some spans can started from this offset and
            # we could end up with wrong extracted span.
            predicted_spans = predicted_spans[torch.where(predicted_spans[:, 0] <= predicted_spans[:, 1])[0]]
            results.append(predicted_spans)
        return results
