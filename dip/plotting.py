import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.exposure import equalize_adapthist


def plot_stacked(
    signal: np.ndarray, timepoints: np.ndarray | None = None, channel_axis: int = 1,
    figsize: tuple[float | None, float | None] = (None, None), save_path: Path | str | None = None,
    show: bool = True, **plt_kwargs,
):
    """Plot a multi-channel signal as stacked channels.

    Args:
        signal (np.ndarray): Multi-channel signal as a 2-D array.
        timepoints (np.ndarray | None, optional): Timepoints in ms corresponding
            to the signal samples. Defaults to None.
        channel_axis (int, optional): The axis of `signal` representing the
            channel dimension. Defaults to 1.
        figsize (tuple[float  |  None, float  |  None], optional): Figure size.
            If the first value is `None`, the figure width will be either 10 or
            20, depending on whether the signal is complex- or real-valued. If
            the second value is `None`, the figure height will be max(1, C//2),
            where C is the number of channels. Defaults to (None, None).
        show (bool, optional): Whether the figure should be displayed or not.
            Defaults to True.
        save_path (Path | str | None, optional): If given, the figure will
            additionally be saved to this path. Defaults to None.
        **plt_kwargs: Keyword arguments passed on to `plt.plot`.
    """

    # move channel dimension to the front and setup figure
    signal = np.moveaxis(signal, channel_axis, 0)
    figsize = (
        figsize[0] or 10 * (1 + np.iscomplexobj(signal)),
        figsize[1] or max(1, signal.shape[0] // 2),
    )
    plt.figure(figsize=figsize)

    # set default pyplot keyword arguments if not given in `plt_kwargs`
    plt_kwargs = {'linewidth': 0.6} | plt_kwargs

    # offset signal channels
    def stack_signals(s):
        d = 1.4 * np.max(np.abs(s - np.mean(s, axis=1, keepdims=True)))
        return s + np.arange(s.shape[0])[:, None] * d

    # function that calls plt.plot depending on whether timepoints are given or not
    def plot(s, t, **kwargs):
        if t is not None:
            plt.plot(t, s.T, **kwargs)
        else:
            plt.plot(s.T, **kwargs)
        plt.yticks([])

    # two subplots for real and imaginary parts
    if np.iscomplexobj(signal):
        for i, part in enumerate([np.real, np.imag]):
            plt.subplot(1, 2, i+1)
            signalpart = part(signal)
            signalpart_stacked = stack_signals(signalpart)
            plot(signalpart_stacked, timepoints, **plt_kwargs)

    # single plot if signal is real-valued
    else:
        signal_stacked = stack_signals(signal)
        plot(signal_stacked, timepoints, **plt_kwargs)

    # show plot
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_multichannel(
    data: np.ndarray, timepoints: np.ndarray | None = None, channel_axis: int = 1, columns: int = 8,
    figheight_per_row: int = 2, figsize: tuple[float | None, float | None] = (None, None),
    complex: str = 'abs', save_path: Path | str | None = None, show: bool = True,
    clip: bool = False, **plt_kwargs,
):
    """Plot a multi-channel signal in several subplots.

    Args:
        data (np.ndarray): Multi-channel signal or image as a 2-D or 3-D array,
            respectively.
        timepoints (np.ndarray | None, optional): Timepoints in ms corresponding
            to the data samples. Defaults to None.
        channel_axis (int, optional): The axis of `data` representing the
            channel dimension. Defaults to 1.
        columns (int, optional): The number of subplot columns. Defaults to 8.
        figheight_per_row (int, optional): The figure height per subplot row.
            Defaults to 2.
        figsize (tuple[float  |  None, float  |  None], optional): Figure size.
            If the first value is `None`, the figure width will be 20. If
            the second value is `None`, the figure height will be `rows` times
            `figheight_per_row`, where `rows` is determined by the number of
            channels and `columns`. Defaults to (None, None).
        complex (str, optional): How complex data should be plotted. 'abs' plots
            the absolute values. 'separate' plots real and imaginary parts as
            separate signals (only for 1-D signals). Defaults to 'abs'.
        save_path (Path | str | None, optional): If given, the figure will
            additionally be saved to this path. Defaults to None.
        show (bool, optional): Whether the figure should be displayed or not.
            Defaults to True.
        clip (bool, optional): Apply percentile clipping.
        **plt_kwargs: Keyword arguments passed on to `plt.plot`.

    Raises:
        ValueError: Raised if `complex` is neither 'abs' nor 'split'.
    """

    # move channel dimension to the front and setup figure
    data = np.moveaxis(data, channel_axis, 0)
    nchans = data.shape[0]
    rows = (nchans - 1) // columns + 1
    figsize = (
        figsize[0] or 20,
        figsize[1] or rows * figheight_per_row,
    )
    _, axs = plt.subplots(rows, columns, figsize=figsize, tight_layout=True)
    if rows == 1 and columns == 1:
        axs = np.asarray(axs)
    axs = axs.flat

    # set default pyplot keyword arguments if not given in `plt_kwargs`
    if data.ndim == 2:
        plt_kwargs = {'linewidth': 0.6} | plt_kwargs

    # function that calls plt.plot depending on whether timepoints are given or not
    def plot(ax, s, t, title=None, **kwargs):
        if s.ndim == 1:
            # 1-D signal
            if t is not None:
                t = (t - t.min()) / 1000
                ax.plot(t, s.T, **kwargs)
            else:
                ax.plot(s.T, **kwargs)

        elif s.ndim == 2:
            # 2-D image
            if t is not None:
                logging.warning('Ignoring time dimension for image data')
            if clip:
                s = np.clip(s, np.percentile(s, 1), np.percentile(s, 99))
            ax.imshow(s, **kwargs)

        else:
            raise ValueError(f'Data must have 1 or 2 dimensions, but had {s.ndim}')

        if title is not None:
            ax.set_title(title)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])

    # iterate over channels
    for c, ax in enumerate(axs):
        if c >= nchans:
            break
        title = f'channel {c}'
        if np.iscomplexobj(data):
            if complex.lower() == 'split':
                if data.ndim == 3:
                    raise ValueError('Image data cannot be plotted with `complex="split"`')
                plot(ax, np.real(data[c]), timepoints, title, **plt_kwargs)
                plot(ax, np.imag(data[c]), timepoints, title, **plt_kwargs)
            elif complex.lower() == 'abs':
                plot(ax, np.abs(data[c]), timepoints, title, **plt_kwargs)
            else:
                raise ValueError(f'Unexpected value for `complex`: {complex}')
        else:
            plot(ax, data[c], timepoints, title, **plt_kwargs)

    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def save_gif(
    frames: np.ndarray, output_path: Path | str, normalize: bool = True, equalize_histogram: bool = False,
    duration: int = 50,
):
    if normalize:
        frames = (frames - frames.min()) / (frames.max() - frames.min())
    if equalize_histogram:
        frames = equalize_adapthist(frames, clip_limit=0.02)
    frames = (frames * 255).astype(np.uint8)

    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(
        output_path, format='GIF', append_images=pil_frames[1:], save_all=True, duration=duration, loop=0,
    )


def save_image(image: np.ndarray, output_path: Path | str, normalize: bool = True, equalize_histogram: bool = False):
    if normalize:
        image = (image - image.min()) / (image.max() - image.min())
    if equalize_histogram:
        image = equalize_adapthist(image, clip_limit=0.02)
    image = (image * 255).astype(np.uint8)

    Image.fromarray(image).save(output_path, format='PNG')
