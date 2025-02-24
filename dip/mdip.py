import inspect
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.types
from tqdm import tqdm

from . import evaluate, models, plotting
from .fft_torch import fftnc
from .loss import TVLoss


class MDIP:
    def __init__(
        self,
        zs: torch.Tensor,
        zt: torch.Tensor,
        basis_gen: models.UNet,
        coeff_gen: models.MLP | None,
        flow_gen: models.FlowGenerator,
        transformer: models.SpatialTransformer,
        matrix_size: tuple[int, int],  # without padding
        n_frames: int,
        imaging_fs: float,
        lambda_flow_spatial: float,
        lambda_flow_temporal: float,
        lambda_zt: float,
        lambda_basis: float,
        noise_reg: float,
        lr_max: float,
        lr_min: float,
        lr_static_factor: float,
        weight_decay: float,
        output_path: str | Path = '.',
    ):
        # save initialization arguments
        self._save_init_args()

        # code vectors
        assert zs.shape[0] == 1
        self.zs = zs  # [1, zs_chans, x, y]
        self.zt = zt  # [frame, zt_chans]

        # time vector
        self.t = torch.arange(0, n_frames, device=zt.device) / imaging_fs

        # modules
        self.basis_gen = basis_gen
        self.coeff_gen = coeff_gen
        self.flow_gen = flow_gen
        self.transformer = transformer

        # hyperparameters
        self.matrix_size = matrix_size
        self.n_frames = n_frames
        self.imaging_fs = imaging_fs  # in Hz
        self.lambda_flow_spatial = lambda_flow_spatial
        self.lambda_flow_temporal = lambda_flow_temporal
        self.lambda_zt = lambda_zt
        self.lambda_basis = lambda_basis
        self.noise_reg = noise_reg
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.lr_static_factor = lr_static_factor
        self.weight_decay = weight_decay

        # losses
        self.kspace_loss = nn.MSELoss(reduction='sum')  # use reduction mode `sum` so we can divide by the number sampled k-space locations and ignore zeros
        self.flow_loss = TVLoss(method='sos')  # sum of squares
        self.zt_loss = TVLoss(method='sos')  # sum of squares
        self.basis_loss = TVLoss(method='iso')  # isotropic TV

        # check sizes
        input_size = (self.zs.shape[2], self.zs.shape[3])
        basis_size = self.basis_gen.get_output_size(input_size)
        assert self.flow_gen.get_output_size() == basis_size
        assert transformer.grid.shape[2:] == basis_size

        # output path
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # metrics
        self.metrics = defaultdict(list)

    def to_device(self, device: torch.types.Device):
        self.zs = self.zs.to(device=device)
        self.zt = self.zt.to(device=device)
        self.basis_gen.to(device=device)
        if self.coeff_gen is not None:
            self.coeff_gen.to(device=device)
        self.flow_gen.to(device=device)
        self.transformer.to(device=device)
        return self

    def _save_init_args(self):
        try:
            frame = inspect.currentframe().f_back  # type: ignore
        except AttributeError as e:
            raise e
        assert frame is not None
        args, _, _, locals_ = inspect.getargvalues(frame)
        args.remove('self')
        locals_ = dict(
            filter(
                lambda x: not isinstance(x[1], torch.Tensor)
                      and not isinstance(x[1], nn.Module),
                locals_.items(),
            ),
        )
        del locals_['self']
        self.init_args = locals_

    def generate_basis(self, noise_reg: float = 0) -> torch.Tensor:
        z_noise = torch.empty_like(self.zs).normal_()
        z_in = self.zs + noise_reg * z_noise
        basis = self.basis_gen(z_in)[0]  # type: torch.Tensor  # [2*basis, x, y]
        basis = basis.reshape(2, basis.shape[0]//2, *basis.shape[1:])  # [2, basis, x, y]
        basis = torch.view_as_complex(basis.moveaxis(0, -1).contiguous())  # [basis, x, y]
        return basis

    def generate_coefficients(self, encoding: torch.Tensor) -> torch.Tensor:
        if self.coeff_gen is not None:
            coeffs = self.coeff_gen(encoding)  # [frame, 2*basis]
            coeffs = coeffs.reshape(coeffs.shape[0], coeffs.shape[1]//2, 2)  # [frame, basis, 2]
            coeffs = torch.view_as_complex(coeffs.contiguous())  # [frame, basis]
            return coeffs
        return torch.ones((encoding.shape[0], 1), device=encoding.device)  # [frame, 1]

    def generate_flow_fields(self, encoding: torch.Tensor) -> torch.Tensor:
        return self.flow_gen(encoding)  # [frame, 2, x, y]

    def generate_cine_frame(self, basis, coeffs, flow) -> torch.Tensor:
        # combine spatial bases
        frame = basis * coeffs.conj()  # [basis, x, y]
        frame = torch.sum(frame, dim=0)  # [x, y]

        # apply spatial transformation
        frame = torch.view_as_real(frame).moveaxis(-1, 0)  # [2, x, y]
        frame = self.transformer(frame[None], flow[None])[0]  # [2, x, y]
        frame = torch.view_as_complex(frame.moveaxis(0, -1).contiguous())  # [x, y]
        return frame

    def forward_(self, noise_reg: float = 0, batch_start: int = 0, batch_end: int = -1) \
            -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # generate spatial basis functions
        basis = self.generate_basis(noise_reg)  # [basis, x, y]

        # extract batch from temporal code vector
        if batch_end < 0:
            batch_end = self.n_frames
        zt_batch = self.zt[batch_start:batch_end]  # [frame, channel]

        # generate basis coefficients
        coeffs = self.generate_coefficients(zt_batch)[..., None, None]  # [frame, basis, 1, 1]

        # generate flow fields
        flow = self.generate_flow_fields(zt_batch)  # [frame, 2, x, y]

        # generate and stack frames
        frames = []
        for f in range(zt_batch.shape[0]):
            frame = self.generate_cine_frame(basis, coeffs[f], flow[f])  # [x, y]
            frames.append(frame)
        cine = torch.stack(frames)  # [frame, x, y]

        # crop everything to matrix size
        crop_x = (cine.shape[1] - self.matrix_size[0]) // 2
        crop_y = (cine.shape[2] - self.matrix_size[1]) // 2
        cine = cine[:, crop_x:crop_x+self.matrix_size[0], crop_y:crop_y+self.matrix_size[1]]
        basis = basis[:, crop_x:crop_x+self.matrix_size[0], crop_y:crop_y+self.matrix_size[1]]
        flow = flow[:, :, crop_x:crop_x+self.matrix_size[0], crop_y:crop_y+self.matrix_size[1]]

        return cine, basis, coeffs, flow

    def forward(self, noise_reg: float = 0) -> torch.Tensor:
        cine, _, _, _ = self.forward_(noise_reg=noise_reg)
        return cine

    def optimize(
        self, k: torch.Tensor, sens: torch.Tensor, mask: torch.Tensor, n_iter: int, save_every: int,
        activate_flow_after: int = 0, batch_size: int = -1, monitor_every: int = -1,
        monitor_gt: np.ndarray | None = None,
    ):
        if batch_size <= 0 or batch_size > k.shape[0]:
            batch_size = k.shape[0]

        # prepare optimizer
        params_basis_gen = list(self.basis_gen.parameters())
        params_flow_gen = list(self.flow_gen.parameters())
        params_coeff_gen = list(self.coeff_gen.parameters()) if self.coeff_gen is not None else []
        self.zs.requires_grad = True
        self.zt.requires_grad = True
        optimizer1 = torch.optim.Adam(
            [
                {'params': params_flow_gen + params_coeff_gen, 'weight_decay': self.weight_decay},
                {'params': self.zt, 'weight_decay': 0},
            ],
            lr=self.lr_max,
        )
        optimizer2 = torch.optim.Adam(
            [
                {'params': params_basis_gen, 'weight_decay': self.weight_decay},
                {'params': self.zs, 'weight_decay': 0},
            ],
            lr=self.lr_max * self.lr_static_factor,
        )
        scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=n_iter, eta_min=self.lr_min)
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer2, T_max=n_iter, eta_min=self.lr_min * self.lr_static_factor,
        )

        # deactivate flow generator
        self.flow_gen.generate_flow = False

        pbar = tqdm(range(n_iter))
        for i in pbar:
            optimizer1.zero_grad()
            optimizer2.zero_grad()

            # activate flow generator
            if i >= activate_flow_after:
                self.flow_gen.generate_flow = True

            # select random mini-batch
            if batch_size < k.shape[0]:
                batch_center = np.random.randint(k.shape[0])
                batch_start = max(0, batch_center - batch_size // 2)
                batch_end = min(k.shape[0], batch_center + batch_size // 2)
            else:
                batch_start = 0
                batch_end = k.shape[0]
            mask_batch = mask[batch_start:batch_end]
            k_batch = torch.view_as_real(k[batch_start:batch_end])

            # run forward pass
            noise_reg_i = self.noise_reg * (1 - 0.9 * i / n_iter)
            cine, basis, coeffs, flow = self.forward_(
                noise_reg=noise_reg_i, batch_start=batch_start, batch_end=batch_end,
            )

            # k-space loss
            cine_tmp = cine[:, None] * sens[None]  # [frame, coil, x, y]
            k_pred = mask_batch * fftnc(cine_tmp, [2, 3])  # [frame, coil, x, y]
            k_pred = torch.view_as_real(k_pred)  # [frame, coil, x, y, 2]
            kspace_loss = self.kspace_loss(k_pred, k_batch)
            kspace_loss = kspace_loss / torch.count_nonzero(k_batch)

            # flow loss
            flow_loss_spatial = self.flow_loss(flow, dims=(2, 3))
            flow_loss_temporal = self.flow_loss(flow, dims=(0,))

            # zt loss
            zt_loss = self.zt_loss(self.zt[batch_start:batch_end], dims=(0,))

            # basis (dictionary item) loss
            basis_loss = self.basis_loss(basis, dims=(1, 2))

            # total loss
            total_loss = kspace_loss \
                + self.lambda_flow_spatial * flow_loss_spatial \
                + self.lambda_flow_temporal * flow_loss_temporal \
                + self.lambda_zt * zt_loss \
                + self.lambda_basis * basis_loss
            total_loss.backward()

            # save loss terms
            self.metrics['kspace_loss'].append(kspace_loss.item())
            self.metrics['flow_loss_spatial'].append(flow_loss_spatial.item())
            self.metrics['flow_loss_temporal'].append(flow_loss_temporal.item())
            self.metrics['zt_loss'].append(zt_loss.item())
            self.metrics['basis_loss'].append(basis_loss.item())
            self.metrics['total_loss'].append(total_loss.item())

            # residual
            residual = torch.linalg.norm(k_pred - k_batch)**2 / torch.linalg.norm(k_batch)**2
            self.metrics['residual'].append(residual.item())
            pbar.set_postfix({'Residual': f'{residual.item():.2e}'})

            # update
            optimizer1.step()
            optimizer2.step()
            scheduler1.step()
            scheduler2.step()

            # save intermediate results
            if save_every > 0 and (i == 0 or (i+1) % save_every == 0):
                with self.no_grad_and_eval():
                    cine, basis, coeffs, flow = self.forward_()
                suffix = f'_epoch_{i+1:05d}'
                self.save_cine(cine, suffix=suffix, equalize_histogram=True, as_npy=False)
                self.save_basis(basis, suffix=suffix)
                if coeffs.shape[1] > 1:
                    self.save_coeffs(coeffs, suffix=suffix)
                self.save_flow(flow, suffix=suffix)
                self.save_static_code_vector(suffix=suffix)
                self.save_temporal_code_vector(suffix=suffix)

            # monitor quantitative metrics
            if monitor_every > 0 and (i+1) % monitor_every == 0:
                if monitor_gt is None:
                    raise ValueError('Ground truth cine data must be provided for monitoring')
                with self.no_grad_and_eval():
                    cine = self.forward()
                metrics = evaluate.get_metrics(np.abs(monitor_gt), np.abs(cine.cpu().numpy()))
                self.metrics['SSIM'].append(metrics['SSIM'].item())
                self.metrics['PSNR'].append(metrics['PSNR'].item())
                self.metrics['NRMSE'].append(metrics['NRMSE'].item())

    @contextmanager
    def no_grad_and_eval(self):
        with torch.no_grad():
            self.basis_gen.eval()
            if self.coeff_gen is not None:
                self.coeff_gen.eval()
            self.flow_gen.eval()
            self.transformer.eval()
            yield
            self.basis_gen.train()
            if self.coeff_gen is not None:
                self.coeff_gen.train()
            self.flow_gen.train()
            self.transformer.train()

    def save_cine(
        self, cine: torch.Tensor | np.ndarray, name: str = 'cine', suffix: str = '', equalize_histogram: bool = False,
        as_gif: bool = True, as_npy: bool = True,
    ):
        cine_np = cine.detach().cpu().numpy() if isinstance(cine, torch.Tensor) else cine
        if as_npy:
            np.save(self.output_path / f'{name}{suffix}.npy', cine_np)
        if as_gif:
            plotting.save_gif(
                np.abs(cine_np), self.output_path / f'{name}{suffix}.gif', normalize=True,
                equalize_histogram=equalize_histogram, duration=min(round(1000 / self.imaging_fs), 200),
            )

    def save_basis(self, basis: torch.Tensor | np.ndarray, name: str = 'basis', suffix: str = '', show: bool = False):
        basis_np = basis.detach().cpu().numpy() if isinstance(basis, torch.Tensor) else basis
        plotting.plot_multichannel(
            basis_np, channel_axis=0, columns=min(5, basis_np.shape[0]), complex='abs', figheight_per_row=4,
            save_path=self.output_path / f'{name}{suffix}.png', show=show, cmap='gray',
        )

    def save_coeffs(self, coeffs: torch.Tensor | np.ndarray, name: str = 'coeffs', suffix: str = '', show: bool = False):
        coeffs_np = coeffs.squeeze().detach().cpu().numpy() if isinstance(coeffs, torch.Tensor) else coeffs.squeeze()
        t_np = np.arange(0, coeffs_np.shape[0]) / self.imaging_fs
        t_np -= t_np[0]
        plotting.plot_stacked(
            coeffs_np, t_np, channel_axis=1, save_path=self.output_path / f'{name}{suffix}',
            show=show,
        )

    def save_flow(self, flow: torch.Tensor | np.ndarray, name: str = 'flow', suffix: str = ''):
        flow_np = flow.detach().cpu().numpy() if isinstance(flow, torch.Tensor) else flow
        flow_np = (flow_np - np.min(flow_np, (0, 2, 3), keepdims=True))
        flow_np = flow_np / np.max(flow_np, (0, 2, 3), keepdims=True)
        duration = min(round(1000 / self.imaging_fs), 200)
        plotting.save_gif(
            flow_np[:, 0], self.output_path / f'{name}_x{suffix}.gif', normalize=False, equalize_histogram=False,
            duration=duration,
        )
        plotting.save_gif(
            flow_np[:, 1], self.output_path / f'{name}_y{suffix}.gif', normalize=False, equalize_histogram=False,
            duration=duration,
        )

    def save_static_code_vector(self, name: str = 'zs', suffix: str = '', show: bool = False):
        zs_np = self.zs[0].detach().cpu().numpy()
        plotting.plot_multichannel(
            zs_np, channel_axis=0, columns=min(5, zs_np.shape[0]), figheight_per_row=4,
            save_path=self.output_path / f'{name}{suffix}.png', show=show, cmap='gray',
        )

    def save_temporal_code_vector(
        self, *other_signals: tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor | None], name: str = 'zt',
        suffix: str = '', show: bool = False,
    ):
        # plot temporal code vector
        zt_np = self.zt.detach().cpu().numpy()
        zt_np = (zt_np - zt_np.mean(axis=0)) / zt_np.std(axis=0)
        t_np = np.arange(0, self.n_frames) / self.imaging_fs
        plt.figure(figsize=(20, 5))
        plt.plot(t_np, zt_np, linewidth=1, linestyle='solid')

        # plot other signals
        for signal in other_signals:
            signal_y = signal[0].detach().cpu().numpy() if isinstance(signal[0], torch.Tensor) else signal[0]
            signal_t = signal[1].detach().cpu().numpy() if isinstance(signal[1], torch.Tensor) else signal[1]
            if signal_t is not None:
                plt.plot(signal_t, signal_y, linewidth=0.6, linestyle='dashed')
            else:
                plt.plot(signal_y, linewidth=0.6, linestyle='dashed')

        # finalize figure, save and show
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(self.output_path / f'{name}{suffix}.png')
        if show:
            plt.show()
        else:
            plt.close()

    def save_metrics(self, name: str = 'metrics', show: bool = False):
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        if 'SSIM' in self.metrics:
            plt.plot(self.metrics['SSIM'])
            plt.ylim(bottom=0.6, top=1)
        plt.title('SSIM')
        plt.subplot(132)
        if 'PSNR' in self.metrics:
            plt.plot(self.metrics['PSNR'])
            plt.ylim(bottom=25, top=42)
        plt.title('PSNR')
        plt.subplot(133)
        if 'NRMSE' in self.metrics:
            plt.plot(self.metrics['NRMSE'])
            plt.ylim(bottom=0, top=0.4)
        plt.title('NRMSE')
        plt.tight_layout()
        plt.savefig(self.output_path / f'{name}.png')
        if show:
            plt.show()
        else:
            plt.close()

    def save(self):
        init_args = {k: (str(v) if isinstance(v, Path) else v) for k, v in self.init_args.items()}
        ckpt = {
            'zs': self.zs,
            'zt': self.zt,
            'basis_gen_repr': repr(self.basis_gen),
            'coeff_gen_repr': repr(self.coeff_gen),
            'flow_gen_repr': repr(self.flow_gen),
            'transformer_repr': repr(self.transformer),
            'basis_gen_state_dict': self.basis_gen.state_dict(),
            'coeff_gen_state_dict': self.coeff_gen.state_dict() if self.coeff_gen is not None else None,
            'flow_gen_state_dict': self.flow_gen.state_dict(),
            'transformer_state_dict': self.transformer.state_dict(),
            'metrics': self.metrics,
            'init_args': init_args,
        }
        torch.save(ckpt, self.output_path / 'state.pt')

    def load(self, device: torch.types.Device = None):
        map_location = None if device is None else torch.device(device)
        ckpt = torch.load(self.output_path / 'state.pt', map_location=map_location)
        self.zs = ckpt['zs']
        self.zt = ckpt['zt']
        self.basis_gen.load_state_dict(ckpt['basis_gen_state_dict'])
        self.flow_gen.load_state_dict(ckpt['flow_gen_state_dict'])
        self.transformer.load_state_dict(ckpt['transformer_state_dict'])
        if self.coeff_gen is not None:
            self.coeff_gen.load_state_dict(ckpt['coeff_gen_state_dict'])
        self.metrics = ckpt['metrics']

    @staticmethod
    def from_checkpoint(
        ckpt_path: Path | str, dtype: torch.dtype, device: torch.types.Device, output_path: str | Path | None = None,
    ) -> 'MDIP':
        ckpt = torch.load(ckpt_path)
        basis_gen = eval('models.' + ckpt['basis_gen_repr']).to(dtype=dtype, device=device)
        coeff_gen = None
        if ckpt['coeff_gen_repr'] is not None:
            coeff_gen = eval('models.' + ckpt['coeff_gen_repr']).to(dtype=dtype, device=device)
        flow_gen = eval('models.' + ckpt['flow_gen_repr']).to(dtype=dtype, device=device)
        transformer = eval('models.' + ckpt['transformer_repr']).to(dtype=dtype, device=device)
        init_args = ckpt['init_args']
        if output_path is not None:
            init_args['output_path'] = output_path
        dip = MDIP(
            zs=ckpt['zs'],  # code vectors are overwritten by dip.load() anyways
            zt=ckpt['zt'],
            basis_gen=basis_gen,
            coeff_gen=coeff_gen,
            flow_gen=flow_gen,
            transformer=transformer,
            **init_args,
        )
        dip.load(device)
        return dip
