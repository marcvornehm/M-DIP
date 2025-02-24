from typing import Sequence

import torch


def fftnc(data: torch.Tensor, axes: Sequence[int], norm: str = 'ortho') -> torch.Tensor:
    return _fftnc(data, False, axes, norm=norm)


def ifftnc(data: torch.Tensor, axes: Sequence[int], norm: str = 'ortho') -> torch.Tensor:
    return _fftnc(data, True, axes, norm=norm)


def _fftnc(data: torch.Tensor, inverse: bool, axes: Sequence[int], norm: str = 'ortho') -> torch.Tensor:
    real_view = False

    if not torch.is_complex(data):
        assert data.shape[-1] == 2, 'Last dimension must be of size 2 if data is not complex'
        data = torch.view_as_complex(data)
        real_view = True

    data = torch.fft.ifftshift(data, dim=axes)
    if inverse:
        data = torch.fft.ifftn(data, dim=axes, norm=norm)
    else:
        data = torch.fft.fftn(data, dim=axes, norm=norm)
    data = torch.fft.fftshift(data, dim=axes)

    if real_view:
        data = torch.view_as_real(data)  # type: ignore

    return data
