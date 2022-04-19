from torch import nn, Tensor

from ASTE.aste.utils import select_index


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, select_index: int, *args, **kwargs):
        self.select_index: int = select_index
        super(CrossEntropyLoss, self).__init__(*args, **kwargs)

    @select_index
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super(CrossEntropyLoss, self).forward(input, target)
