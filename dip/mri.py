from typing import Sequence

import numpy as np
import scipy.signal

from .fft_np import fftnc, ifftnc


def crop_readout_oversampling(data: np.ndarray, ro_axis: int = 0, apodize: bool = False) -> np.ndarray:
    # move readout axis to the front
    data = np.moveaxis(data, ro_axis, 0)

    # extract acquired readouts
    mask = abs(data[data.shape[0]//2]) > 0
    readouts = data[:, mask]

    # get asymmetric echo
    pre_z = np.where(np.abs(readouts[:, 0]) > 0)[0][0]

    # ifft along readout
    readouts = ifftnc(readouts, axes=[0])  # type: ignore

    # crop
    readouts = center_crop(readouts, readouts.shape[0] // 2, 0)

    # apodize
    if apodize:
        window = scipy.signal.windows.tukey(readouts.shape[0], alpha=0.2)
        readouts *= window[:, None]

    # fft along readout
    readouts = fftnc(readouts, [0])  # type: ignore

    # re-apply asymmetric echo
    readouts[:pre_z // 2] = 0

    # insert cropped data into new array
    data = np.zeros(shape=(readouts.shape[0],) + data.shape[1:], dtype=data.dtype)
    data[:, mask] = readouts

    # move readout axis back
    data = np.moveaxis(data, 0, ro_axis)

    return data


def _center_crop_axis(data: np.ndarray, size: int, axis: int) -> np.ndarray:
    # check if crop is necessary
    if size >= data.shape[axis]:
        return data

    # move axis to the front
    if axis != 0:
        data = np.moveaxis(data, axis, 0)

    # crop
    low = data.shape[0] // 2 - size // 2
    high = low + size
    data = data[low:high]

    # move axis back
    if axis != 0:
        data = np.moveaxis(data, 0, axis)

    return data


def center_crop(data: np.ndarray, size: int | Sequence[int], axes: int | Sequence[int]) -> np.ndarray:
    if isinstance(size, int):
        size = [size]
    if isinstance(axes, int):
        axes = [axes]
    assert len(size) == len(axes), 'Size and axes must have the same length'

    # crop along each axis
    for s, a in zip(size, axes):
        data = _center_crop_axis(data, s, a)

    return data


def coil_compression(kspace: np.ndarray, n_coils: int, ch_axis: int = 0) -> np.ndarray:
    # transpose and reshape
    kspace = np.moveaxis(kspace, ch_axis, -1)
    sh = kspace.shape[:-1]
    kspace = kspace.reshape(-1, kspace.shape[-1])

    # compress
    cov = np.conj(kspace.T) @ kspace
    u, _, _ = np.linalg.svd(cov)
    kspace = kspace @ u
    kspace = kspace[:, :n_coils]

    # reshape and transpose back
    kspace = kspace.reshape(*sh, n_coils)
    kspace = np.moveaxis(kspace, -1, ch_axis)

    return kspace


def average_data(data: np.ndarray, axis: int):
    mask = np.abs(data) > 0
    data_avg = np.sum(data, axis=axis) / (np.sum(mask, axis = axis) + np.finfo(np.float64).eps)
    return data_avg.astype(data.dtype)


def rss(data: np.ndarray, ch_axis: int):
    return np.sqrt(np.sum(np.abs(data) ** 2, axis=ch_axis))
