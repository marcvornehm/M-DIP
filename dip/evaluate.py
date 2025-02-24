from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from skimage.color import gray2rgb
from skimage.metrics import normalized_root_mse, peak_signal_noise_ratio, structural_similarity

from .plotting import save_gif, save_image


def get_ssim(gt: np.ndarray, pred: np.ndarray, mean_axis: int | None = None) -> float:
    gt = gt[None] if mean_axis is None else np.moveaxis(gt, mean_axis, 0)
    pred = pred[None] if mean_axis is None else np.moveaxis(pred, mean_axis, 0)
    maxval = gt.max()
    ssims = [structural_similarity(gt[i], pred[i], data_range=maxval) for i in range(gt.shape[0])]
    return sum(ssims) / len(ssims)


def get_psnr(gt: np.ndarray, pred: np.ndarray, mean_axis: int | None = None) -> float:
    gt = gt[None] if mean_axis is None else np.moveaxis(gt, mean_axis, 0)
    pred = pred[None] if mean_axis is None else np.moveaxis(pred, mean_axis, 0)
    maxval = gt.max()
    psnrs = [peak_signal_noise_ratio(gt[i], pred[i], data_range=maxval) for i in range(gt.shape[0])]
    return sum(psnrs) / len(psnrs)


def get_nrmse(gt: np.ndarray, pred: np.ndarray, mean_axis: int | None = None) -> float:
    gt = gt[None] if mean_axis is None else np.moveaxis(gt, mean_axis, 0)
    pred = pred[None] if mean_axis is None else np.moveaxis(pred, mean_axis, 0)
    nrmses = [normalized_root_mse(gt[i], pred[i]) for i in range(gt.shape[0])]
    return sum(nrmses) / len(nrmses)


def get_metrics(
        gt: np.ndarray, *preds: tuple[str, np.ndarray] | np.ndarray, bbox: Sequence[int] | None = None,
        center: Sequence[int] | None = None,
) -> pd.DataFrame:
    if bbox is not None:
        assert len(bbox) == 4
        gt = gt[:, bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
    if center is not None:
        assert len(center) == 2
        gt = np.stack([gt[:, center[1], :], gt[:, :, center[0]]], axis=0)
    metrics = []
    index = []
    for pred in preds:
        name, pred = pred if isinstance(pred, tuple) else (len(index), pred)
        if bbox is not None:
            pred = pred[:, bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        if center is not None:
            pred = np.stack([pred[:, center[1], :], pred[:, :, center[0]]], axis=0)
        index.append(name)
        metrics.append({
            'SSIM': get_ssim(gt, pred, mean_axis=0),
            'PSNR': get_psnr(gt, pred, mean_axis=0),
            'NRMSE': get_nrmse(gt, pred, mean_axis=0),
        })
    return pd.DataFrame(metrics, index)


def update_metrics_csv(csv_path: str | Path, *metrics: tuple[str, pd.DataFrame]):
    dfs = []
    for context, df in metrics:
        df['method'] = df.index
        df = pd.melt(df, id_vars=['method'], var_name='metric')
        df['context'] = context
        dfs.append(df)
    all_metrics = pd.concat(dfs)
    if Path(csv_path).exists():
        prev_metrics = pd.read_csv(csv_path)
        all_metrics = pd.concat([prev_metrics, all_metrics])
        all_metrics = all_metrics[~all_metrics.duplicated(subset=['method', 'metric', 'context'], keep='last')]
    all_metrics.to_csv(csv_path, index=False)


def save_cine_roi(cine: np.ndarray, bbox: Sequence[int], path: str | Path, tres: int):
    assert len(bbox) == 4
    cine = cine[:, bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
    save_gif(cine, Path(path), normalize=True, equalize_histogram=True, duration=tres)


def save_temporal_profiles(cine: np.ndarray, center: Sequence[int], path: str | Path):
    assert len(center) == 2
    xt = cine[:, center[1], :]
    yt = cine[:, :, center[0]]
    path = Path(path)
    save_image(xt, path.with_name(f'{path.stem}_xt.png'))
    save_image(yt, path.with_name(f'{path.stem}_yt.png'))


def save_error_map(gt: np.ndarray, pred: np.ndarray, path: str | Path, tres: int, scale: float = 1):
    error = np.abs(gt - pred) / gt.max()
    error = np.clip(error * scale, 0, 1)
    save_gif(error, Path(path), normalize=False, equalize_histogram=False, duration=tres)


def save_overview_image(frame: np.ndarray, bbox: Sequence[int], center: Sequence[int], path: str | Path):
    assert len(bbox) == 4
    assert len(center) == 2
    frame = gray2rgb(frame)
    frame = (frame - frame.min()) / (frame.max() - frame.min()) * 255
    frame[center[1], :] = (255, 0, 0)
    frame[:, center[0]] = (255, 0, 0)
    frame[bbox[1]:bbox[1]+bbox[3], bbox[0]] = (255, 255, 0)
    frame[bbox[1]:bbox[1]+bbox[3], bbox[0]+bbox[2]] = (255, 255, 0)
    frame[bbox[1], bbox[0]:bbox[0]+bbox[2]] = (255, 255, 0)
    frame[bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = (255, 255, 0)
    save_image(frame, path)
