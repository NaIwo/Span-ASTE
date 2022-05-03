from ASTE.dataset.domain.const import ChunkCode

import torch
from torch import Tensor


def ignore_index(func):
    def ignore_and_call(self, preds: Tensor, target: Tensor, *args, **kwargs):
        if self.ignore_index is not None:
            indices = (target != self.ignore_index).nonzero(as_tuple=False).flatten()
            preds_ignored = preds[indices]
            target_ignored = target[indices]
            if 'mask' in kwargs.keys():
                kwargs['mask'] = kwargs['mask'][indices]
            res = func(self, preds_ignored, target_ignored, *args, **kwargs)
        else:
            res = func(self, preds, target, *args, **kwargs)
        return res

    return ignore_and_call


def select_index(func):
    def select_and_call(self, preds: Tensor, target: Tensor, *args, **kwargs):
        if self.select_index is not None:
            indices = (target == self.select_index).nonzero(as_tuple=False).flatten()
            preds_selected = preds[indices]
            target_selected = target[indices]
            if 'mask' in kwargs.keys():
                kwargs['mask'] = kwargs['mask'][indices]
            if len(torch.unique(target_selected)) == 1 and ChunkCode.NOT_RELEVANT in target_selected:
                target_selected = torch.full_like(target_selected, ChunkCode.NOT_SPLIT)
            res = func(self, preds_selected, target_selected, *args, **kwargs)
        else:
            res = func(self, preds, target, *args, **kwargs)
        return res

    return select_and_call
