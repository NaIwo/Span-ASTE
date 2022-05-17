import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from ASTE.utils import config

'''
Implementation taken from: 
https://github.com/migonch/crfseg
'''


class CRF(nn.Module):
    """
    Class for learning and inference in conditional random field model using mean field approximation
    and convolutional approximation in pairwise potentials term.
    Parameters
    ----------
    filter_size : int or sequence of ints
        Size of the gaussian filters in message passing.
        If it is a sequence its length must be equal to ``n_spatial_dims`` (2 in this case - height and width).
    n_iter : int
        Number of iterations in mean field approximation.
    smoothness_weight : float
        Initial weight of smoothness kernel.
    smoothness_theta : float or sequence of floats
        Initial bandwidths for each spatial feature in the gaussian smoothness kernel.
        If it is a sequence its length must be equal to ``n_spatial_dims``.
    returns : str
        Can be 'logits', 'proba', 'log-proba'.
    model_name : str
        Name of the model
    """

    def __init__(self, filter_size: int = 9, n_iter: int = 5, smoothness_weight: int = 1, smoothness_theta: float = 0.5,
                 returns: str = 'logits', model_name: str = 'CRF model'):
        super(CRF, self).__init__()

        self.model_name: str = model_name
        # CRF Implementation
        self.n_iter = n_iter
        self.returns = returns
        self.filter_size = np.broadcast_to(filter_size, 2)
        self._set_param('smoothness_weight', smoothness_weight)
        self._set_param('inv_smoothness_theta', 1 / np.broadcast_to(smoothness_theta, 2))

    def _set_param(self, name, init_value):
        setattr(self, name, nn.Parameter(
            torch.tensor(init_value, dtype=torch.float, requires_grad=True, device=config['general']['device'])))

    def forward(self, x, spatial_spacings=None, verbose=False):
        """
        Parameters
        ----------
        x : torch.tensor
            Tensor of shape ``(batch_size, height, width, n_classes)`` with negative unary potentials, e.g. the CNN's output.
        spatial_spacings : array of floats or None
            Array of shape ``(batch_size, 2)`` with spatial spacings of tensors in batch ``x``.
            None is equivalent to all ones. Used to adapt spatial gaussian filters to different inputs' resolutions.
        verbose : bool
            Whether to display the iterations using tqdm-bar.
        Returns
        -------
        output : torch.tensor
            Tensor of shape ``(batch_size, height, width, n_classes)``
            with logits or (log-)probabilities of assignment to each class.
        """
        x = x.permute(0, 3, 1, 2)

        batch_size, *remaining = x.shape

        if spatial_spacings is None:
            spatial_spacings = np.ones((batch_size, 2))

        negative_unary = x.clone()

        self._set_num_iterations()
        for _ in tqdm(range(self.n_iter), disable=not verbose):
            # normalizing
            x = F.softmax(x, dim=1)

            # message passing
            x = self.smoothness_weight * self._smoothing_filter(x, spatial_spacings)

            # compatibility transform
            x = self._compatibility_transform(x)

            # adding unary potentials
            x = negative_unary - x

        if self.returns == 'logits':
            output = x
        elif self.returns == 'proba':
            output = F.softmax(x, dim=1)
        elif self.returns == 'log-proba':
            output = F.log_softmax(x, dim=1)
        else:
            raise ValueError("Attribute ``returns`` must be 'logits', 'proba' or 'log-proba'.")

        return output.permute(0, 2, 3, 1)

    def _smoothing_filter(self, x, spatial_spacings):
        """
        Parameters
        ----------
        x : torch.tensor
            Tensor of shape ``(batch_size, n_classes, height, width)`` with negative unary potentials, e.g. logits.
        spatial_spacings : torch.tensor or None
            Tensor of shape ``(batch_size, 2)`` with spatial spacings of tensors in batch ``x``.
        Returns
        -------
        output : torch.tensor
            Tensor of shape ``(batch_size, n_classes, height, width)``.
        """
        return torch.stack([self._single_smoothing_filter(x[i], spatial_spacings[i]) for i in range(x.shape[0])])

    @staticmethod
    def _pad(x, filter_size):
        padding = []
        for fs in filter_size:
            padding += 2 * [fs // 2]

        return F.pad(x, list(reversed(padding)))  # F.pad pads from the end

    def _single_smoothing_filter(self, x, spatial_spacing):
        """
        Parameters
        ----------
        x : torch.tensor
            Tensor of shape ``(n, height, width)``.
        spatial_spacing : sequence of 2 floats
        Returns
        -------
        output : torch.tensor
            Tensor of shape ``(n, height, width)``.
        """
        x = self._pad(x, self.filter_size)
        for i, dim in enumerate(range(1, x.ndim)):
            # reshape to (-1, 1, x.shape[dim])
            x = x.transpose(dim, -1)
            shape_before_flatten = x.shape[:-1]
            x = x.flatten(0, -2).unsqueeze(1)

            # 1d gaussian filtering
            kernel = self._create_gaussian_kernel1d(self.inv_smoothness_theta[i], spatial_spacing[i],
                                                    self.filter_size[i]).view(1, 1, -1).to(x)
            x = F.conv1d(x, kernel)

            # reshape back to (n, *spatial)
            x = x.squeeze(1).view(*shape_before_flatten, x.shape[-1]).transpose(-1, dim)

        return x

    @staticmethod
    def _create_gaussian_kernel1d(inverse_theta, spacing, filter_size):
        """
        Parameters
        ----------
        inverse_theta : torch.tensor
            Tensor of shape ``(,)``
        spacing : float
        filter_size : int
        Returns
        -------
        kernel : torch.tensor
            Tensor of shape ``(filter_size,)``.
        """
        distances = spacing * torch.arange(-(filter_size // 2), filter_size // 2 + 1).to(inverse_theta)
        kernel = torch.exp(-(distances * inverse_theta) ** 2 / 2)
        zero_center = torch.ones(filter_size).to(kernel)
        zero_center[filter_size // 2] = 0
        return kernel * zero_center

    def _compatibility_transform(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor of shape ``(batch_size, n_classes, height, width)``.
        Returns
        -------
        output : torch.tensor of shape ``(batch_size, n_classes, height, width)``.
        """
        labels = torch.arange(x.shape[1])
        compatibility_matrix = self._compatibility_function(labels, labels.unsqueeze(1)).to(x)
        return torch.einsum('ij..., jk -> ik...', x, compatibility_matrix)

    @staticmethod
    def _compatibility_function(label1, label2):
        """
        Input tensors must be broadcastable.
        Parameters
        ----------
        label1 : torch.Tensor
        label2 : torch.Tensor
        Returns
        -------
        compatibility : torch.Tensor
        """
        return -(label1 == label2).float()

    def _set_num_iterations(self):
        # Due to faster training (authors used this solution (prevent from gradient vanishing/exploding))
        if not self.training:  # if model in eval mode
            self.n_iter = 10
        else:
            self.n_iter = 5
