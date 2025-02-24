from typing import Sequence

import numpy as np


def fftnc(data: np.ndarray, axes: Sequence[int], norm: str = 'ortho') -> np.ndarray:
    return _fftnc(data, False, axes, norm=norm)


def ifftnc(data: np.ndarray, axes: Sequence[int], norm: str = 'ortho') -> np.ndarray:
    return _fftnc(data, True, axes, norm=norm)


def _fftnc(data: np.ndarray, inverse: bool, axes: Sequence[int], norm: str = 'ortho') -> np.ndarray:
    real_view = False

    if not np.iscomplexobj(data):
        assert data.shape[-1] == 2, 'Last dimension must be of size 2 if data is not complex'
        data = data[..., 0] + 1j*data[..., 1]
        real_view = True

    dtype = data.dtype

    data = np.fft.ifftshift(data, axes=axes)
    if inverse:
        data = np.fft.ifftn(data, norm=norm, axes=axes)  # type: ignore
    else:
        data = np.fft.fftn(data, norm=norm, axes=axes)  # type: ignore
    data = np.fft.fftshift(data, axes=axes)

    # np.fft.fftn returns dtype complex128 regardless of input dtype
    if dtype in [np.complex64, np.float32]:
        data = data.astype(np.complex64)

    if real_view:
        data = np.stack([data.real, data.imag], axis=-1)

    return data
