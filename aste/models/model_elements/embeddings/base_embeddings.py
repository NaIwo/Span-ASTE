from typing import Any

from ....models import BaseModel


class BaseEmbedding(BaseModel):
    def __init__(self, embedding_dim: int, model_name: str = 'Base embedding model'):
        super(BaseEmbedding, self).__init__(model_name=model_name)
        self.model: Any = None
        self.embedding_dim: int = embedding_dim
