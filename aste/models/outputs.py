import torch
from typing import List


class ModelOutput:
    def __init__(self, chunker_output: torch.Tensor, predicted_spans: List[torch.Tensor]):
        self.chunker_output: torch.Tensor = chunker_output
        self.predicted_spans: List[torch.Tensor] = predicted_spans
