from typing import Any, Union

from aste.utils import config
from transformers import DebertaModel, AutoModel

from ....models import BaseModel


class BaseEmbedding(BaseModel):
    def __init__(self, embedding_dim: int, model_name: str = 'Base embedding model'):
        super(BaseEmbedding, self).__init__(model_name=model_name)
        self.model: Any = None
        self.embedding_dim: int = embedding_dim

    @staticmethod
    def get_transformer_encoder_from_config() -> Union[DebertaModel, AutoModel]:
        if 'deberta' in config['encoder']['transformer']['source']:
            return DebertaModel.from_pretrained(config['encoder']['transformer']['source'])
        elif 'bert' in config['encoder']['transformer']['source']:
            return AutoModel.from_pretrained(config['encoder']['transformer']['source'])
        else:
            raise Exception(f"We do not support this transformer model {config['encoder']['transformer']['source']}!")
