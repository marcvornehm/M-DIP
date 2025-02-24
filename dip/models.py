import inspect
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F


class MyBaseModule(nn.Module, ABC):
    def __init__(self):
        super().__init__()

        try:
            # quite an assumption that we only need to go back one frame tbh...
            frame = inspect.currentframe().f_back  # type: ignore
        except AttributeError as e:
            raise e
        assert frame is not None
        args, varargs, keywords, locals_ = inspect.getargvalues(frame)
        args.remove('self')
        del locals_['self']
        del locals_['__class__']
        self._initargvalues = inspect.formatargvalues(args, varargs, keywords, locals_)
        self._config = locals_

    @property
    def config(self):
        return deepcopy(self._config)

    @abstractmethod
    def get_output_size(self, *input_sizes: Any) -> tuple[int, ...]:
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}{self._initargvalues}'


class UNet(MyBaseModule):
    def __init__(
        self,
        enc_channels: Sequence[int],
        dec_channels: Sequence[int],
        out_channels: int,
        kernel_size: int,
        n_convs_per_block: int = 2,
        skip_convs: bool = False,
        p_dropout: float = 0,
        interpolation_mode: str = 'nearest',
    ):
        super().__init__()
        self.interpolation_mode = interpolation_mode
        assert (len(enc_channels) > 1) and (len(enc_channels) == len(dec_channels))
        enc_channels = list(enc_channels)
        dec_channels = list(dec_channels)

        # make sure that the number of channels always increases in the encoder and decreases in the decoder
        for i in range(1, len(enc_channels)):
            enc_channels[i] = max(enc_channels[i], enc_channels[i-1])
        dec_channels[-1] = max(dec_channels[-1], out_channels)
        for i in range(0, len(dec_channels) - 1)[::-1]:
            dec_channels[i] = max(dec_channels[i], dec_channels[i+1])

        # encoder
        self.enc_blocks = nn.ModuleList()
        for i in range(len(enc_channels) - 1):
            block = ConvBlock(
                enc_channels[i], enc_channels[i+1], kernel_size, n_convs=n_convs_per_block, p_dropout=p_dropout,
            )
            self.enc_blocks.append(block)

        # bottleneck
        self.bottleneck = ConvBlock(
            enc_channels[-1], dec_channels[0], kernel_size, n_convs=n_convs_per_block, p_dropout=p_dropout,
        )

        # decoder
        self.dec_blocks = nn.ModuleList()
        for i in range(len(dec_channels) - 1):
            in_channels = dec_channels[i] + enc_channels[-i-1]
            block = ConvBlock(
                in_channels, dec_channels[i+1], kernel_size, n_convs=n_convs_per_block, p_dropout=p_dropout,
            )
            self.dec_blocks.append(block)

        # output conv
        self.out_conv = nn.Conv2d(dec_channels[-1], out_channels, 1)

        # skip convs
        self.skip_blocks = nn.ModuleList()
        for i in range(1, len(enc_channels)):
            if skip_convs:
                block = ConvBlock(
                    enc_channels[i], enc_channels[i], kernel_size, n_convs=n_convs_per_block, p_dropout=p_dropout,
                )
            else:
                block = nn.Identity()
            self.skip_blocks.append(block)

    def forward(self, x: torch.Tensor):
        # encoder
        skip_stack = []
        for enc, skip in zip(self.enc_blocks, self.skip_blocks):
            x = enc(x)
            skip_stack.append(skip(x))
            x = F.avg_pool2d(x, 2)

        # bottleneck
        x = self.bottleneck(x)

        # decoder
        for dec in self.dec_blocks:
            x = F.interpolate(x, scale_factor=2, mode=self.interpolation_mode)
            x = torch.cat([x, skip_stack.pop()], dim=1)
            x = dec(x)

        # output conv
        x = self.out_conv(x)
        return x

    def get_bottleneck_size(self, input_size: tuple[int, int]):
        bottleneck_size = (
            input_size[0] / 2**len(self.enc_blocks),
            input_size[1] / 2**len(self.enc_blocks),
        )
        assert bottleneck_size[0] % 1 == 0 and bottleneck_size[1] % 1 == 0
        return (int(bottleneck_size[0]), int(bottleneck_size[1]))

    def get_output_size(self, input_size: tuple[int, int]):
        return input_size

    def required_input_size(self, image_size: tuple[int, int], min_bottleneck_size: int):
        # allowed input sizes are {min_size + k*multiple_of}, where k \in N_0
        mul = 2**len(self.enc_blocks)
        min_ = min_bottleneck_size * 2**len(self.enc_blocks)
        return tuple(max(min_, ((s-1) // mul + 1) * mul) for s in image_size)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        n_convs: int = 2,
        p_dropout: float = 0,
    ):
        super().__init__()
        assert n_convs > 0

        padding = kernel_size // 2

        self.layers = nn.ModuleList()

        for i in range(n_convs):
            ch_in = in_channels if i == 0 else out_channels
            self.layers.append(nn.Conv2d(ch_in, out_channels, kernel_size, padding=padding, padding_mode='reflect'))
            self.layers.append(nn.LeakyReLU())
            self.layers.append(nn.BatchNorm2d(out_channels))
            self.layers.append(nn.Dropout2d(p_dropout))

    def forward(self, x: torch.Tensor):
        for m in self.layers:
            x = m(x)
        return x


class ConvDecoder(MyBaseModule):
    def __init__(
        self,
        channels: Sequence[int],
        kernel_size: int,
        n_convs_per_block: int = 2,
        p_dropout: float = 0,
        interpolation_mode: str = 'nearest',
    ):
        super().__init__()
        self.interpolation_mode = interpolation_mode
        self.dec_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            block = ConvBlock(channels[i], channels[i+1], kernel_size, n_convs=n_convs_per_block, p_dropout=p_dropout)
            self.dec_blocks.append(block)

    def forward(self, x: torch.Tensor):
        for i, dec in enumerate(self.dec_blocks):
            x = dec(x)
            if i < len(self.dec_blocks) - 1:
                x = F.interpolate(x, scale_factor=2, mode=self.interpolation_mode)
        return x

    def get_output_size(self, input_size: tuple[int, int]):
        output_size = (
            input_size[0] * 2**(len(self.dec_blocks) - 1),
            input_size[1] * 2**(len(self.dec_blocks) - 1),
        )
        assert output_size[0] % 1 == 0 and output_size[1] % 1 == 0
        return (int(output_size[0]), int(output_size[1]))


class MLP(MyBaseModule):
    def __init__(
        self,
        feature_lengths: Sequence[int],
        last_activation: bool = True,
        p_dropout: float = 0,
    ):
        super().__init__()

        self.feature_lengths = feature_lengths
        self.layers = nn.ModuleList()
        for i in range(len(feature_lengths) - 1):
            layer = nn.Linear(feature_lengths[i], feature_lengths[i+1])
            self.layers.append(layer)
            self.layers.append(nn.Dropout(p_dropout))
            if i < len(feature_lengths) - 2 or last_activation:
                self.layers.append(nn.LeakyReLU())

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

    def get_output_size(self):
        return (self.feature_lengths[-1],)


class SpatialTransformer(MyBaseModule):
    """
    N-D Spatial Transformer

    Code based on https://github.com/voxelmorph/voxelmorph/blob/dev/voxelmorph/torch/layers.py
    """

    def __init__(self, size: Sequence[int], mode: str='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.float32)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, x: torch.Tensor, flow: torch.Tensor):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(x, new_locs, align_corners=True, mode=self.mode)

    def get_output_size(self, input_size: tuple[int, int]):
        return input_size


class FlowGenerator(MyBaseModule):
    def __init__(
        self,
        mlp_features: Sequence[int],
        conv_input_size: tuple[int, int],
        conv_channels: Sequence[int],
        n_convs_per_block: int = 2,
        p_dropout: float = 0,
        interpolation_mode: str = 'nearest',
    ):
        super().__init__()
        self.conv_input_size = conv_input_size

        # MLP
        self.mlp = MLP(list(mlp_features) + [int(np.prod(conv_input_size))], last_activation=True, p_dropout=p_dropout)

        # convolutional decoder
        self.dec = ConvDecoder(
            [1] + list(conv_channels), 3, n_convs_per_block=n_convs_per_block, p_dropout=p_dropout,
            interpolation_mode=interpolation_mode,
        )

        # convolutional layer that generates flow fields
        self.flow = nn.Conv2d(conv_channels[-1], 2, 3, padding=1, padding_mode='reflect')
        self.flow.weight = nn.Parameter(dist.normal.Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))  # type: ignore

        # flag to activate/deactivate flow generation
        self.generate_flow = True

    def forward(self, x: torch.Tensor):
        if not self.generate_flow:
            return torch.zeros(x.shape[0], 2, *self.get_output_size(), device=x.device, dtype=x.dtype)

        # pass through MLP
        x = self.mlp(x)

        # reshape 1-D to 2-D
        x = x.reshape(x.shape[0], *self.conv_input_size)

        # insert channel dimension
        x = x[:, None]

        # pass through convolutional decoder
        x = self.dec(x)

        # generate flow fields
        x = self.flow(x)

        return x

    def get_output_size(self):
        output_size = self.dec.get_output_size(self.conv_input_size)
        assert output_size[0] % 1 == 0 and output_size[1] % 1 == 0
        return (int(output_size[0]), int(output_size[1]))
