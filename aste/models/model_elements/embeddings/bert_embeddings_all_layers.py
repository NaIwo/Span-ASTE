from transformers import BertModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions as EmbOut
import torch

from ASTE.utils import config
from .base_embeddings import BaseEmbedding


class WeightedBert(BaseEmbedding):
    def __init__(self, model_name: str = 'Bert embedding (all layers) model'):
        dim: int = config['encoder']['bert']['embedding-dimension']
        super(WeightedBert, self).__init__(embedding_dim=dim, model_name=model_name)
        self.model: BertModel = BertModel.from_pretrained(config['model']['bert']['source'])

        self.linear_layer_1 = torch.nn.Linear(dim, 500)
        self.linear_layer_2 = torch.nn.Linear(500, 300)
        self.linear_layer_3 = torch.nn.Linear(300, 100)
        self.final_layer = torch.nn.Linear(100, 1)
        self.dropout = torch.nn.Dropout(0.1)
        self.batch_norm = torch.nn.BatchNorm1d(dim)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        emb_out: EmbOut = self.model.forward(inputs, mask, output_hidden_states=True)
        embeddings: torch.Tensor = torch.stack(emb_out.hidden_states, dim=1)

        attention: torch.Tensor = self._get_attention_weights(embeddings)
        embeddings = attention[..., None, None] * embeddings

        return torch.sum(embeddings, dim=1)

    def _get_attention_weights(self, embeddings: torch.Tensor) -> torch.Tensor:
        attention: torch.Tensor = embeddings[:, :, 0, :]
        attention = self.batch_norm(torch.permute(attention, (0, 2, 1)))
        attention = torch.permute(attention, (0, 2, 1))

        layer: torch.nn.Linear
        for layer in [self.linear_layer_1, self.linear_layer_2, self.linear_layer_3]:
            attention = layer(attention)
            attention = torch.relu(attention)
            attention = self.dropout(attention)
        attention = self.final_layer(attention).squeeze()

        return self.softmax(attention)
