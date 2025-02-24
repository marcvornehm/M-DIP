from typing import Sequence

import torch
import torch.nn as nn


class TVLoss(nn.Module):
    def __init__(self, method: str):
        """Total variation loss.

        Implemented methods, exemplified for the 2D case, where u is the input
        tensor and D_x and D_y are the partial derivative operators:
        * method = 'ani' (anisotropic TV): ||D_x u||_1 + ||D_y u||_1
        * method = 'iso' (isotropic TV): || sqrt(|D_x u|^2 + |D_y u|^2) ||_1
        * method = 'sos' (sum of squares): ||D_x u||_2^2 + ||D_y u||_2^2

        Args:
            method (str, optional): 'ani', 'iso' or 'sos'.
        """
        super().__init__()
        self.method = method

    def forward(self, x: torch.Tensor, dims: Sequence[int] | None = None) -> torch.Tensor:
        if dims is None:
            dims = list(range(2, x.ndim))  # assume [batch, channel, ...]

        # compute gradients along each dimension
        diffs = []
        for dim in dims:
            x_temp = x.moveaxis(dim, 0)  # move dimension to front
            x_temp = torch.concat([x_temp, x_temp[-1:]], dim=0)  # extend at boundary
            diff = x_temp[1:] - x_temp[:-1]  # difference

            if self.method == 'ani':
                # absolute value (anisotropic TV)
                diff = torch.abs(diff)
            elif self.method in ['iso', 'sos']:
                # square (isotropic TV and sum of squares)
                diff = torch.real(diff * diff.conj())
            else:
                raise ValueError(f'method can only be ani, iso, or sos, but was {self.method}')

            diff = diff.moveaxis(0, dim)  # move dimension back
            diffs.append(diff)

        # average over the gradient directions
        diffs = torch.mean(torch.stack(diffs), dim=0)

        # take the square root for isotropic TV
        if self.method == 'iso':
            diffs = torch.sqrt(diffs + torch.finfo(diffs.dtype).eps)

        # average and return
        return diffs.mean()
