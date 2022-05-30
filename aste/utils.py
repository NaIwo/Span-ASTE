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

