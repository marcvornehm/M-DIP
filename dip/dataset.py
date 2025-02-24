import logging
import os
from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from pathlib import Path

import ismrmrd
import numpy as np
import scipy.io
import scipy.signal
import sigpy.mri
from tqdm import tqdm

from .fft_np import fftnc
from .gro import GRO, GROParam
from .mri import average_data, crop_readout_oversampling

PhysioSignal = namedtuple('PhysioSignal', ['data', 'time'])


class Dataset(ABC):
    def __init__(self, filename: Path | str):
        self.filename: Path | str = filename

        self._k: np.ndarray  # [slice, frame, coil, kx, ky]
        self._k_t: np.ndarray  # [slice, frame, ky]

        self.header: ismrmrd.xsd.ismrmrdHeader
        self.tres: int
        self.recon_size: tuple[int, int]

        self._physio: dict[int, PhysioSignal] = {}  # wave ID -> ([time, channel], [time,])

        self.noise: np.ndarray  # [coil, readout, acquisition]

        self._sl = None
        self._t_min = None
        self._t_max = None

    @property
    def k(self):
        if self._sl is None:
            return self._k.copy()
        else:
            return self._k[self._sl].copy()

    @property
    def k_t(self):
        if self._sl is None:
            return self._k_t.copy()
        else:
            return self._k_t[self._sl].copy()

    def get_physio(self, id: int, normalize: bool = True):
        p = self._get_physio(self._physio[id], 'data')
        if normalize:
            mean = np.mean(p, axis=0, keepdims=True)
            std = np.std(p, axis=0, keepdims=True)
            std[std == 0] = 1  # some physio channels may be zero, which would lead to a division by 0
            p = (p - mean) / std
        return p

    def get_physio_t(self, id: int):
        return self._get_physio(self._physio[id], 'time')

    def _get_physio(self, physio: PhysioSignal, attr: str):
        if self._sl is not None:
            mask = (physio.time >= self._t_min) & (physio.time <= self._t_max)
        else:
            mask = np.full(physio.time.shape, True, dtype=bool)
        return physio.__getattribute__(attr)[mask].copy()

    @property
    def n_slices(self):
        return self._k.shape[0]

    @property
    def n_phases(self):
        return self._k.shape[1]

    @property
    def n_coils(self):
        return self._k.shape[2]

    @property
    def matrix_size(self):
        return (self._k.shape[3], self._k.shape[4])

    @property
    def sl(self):
        return self._sl

    @sl.setter
    def sl(self, sl):
        if sl is not None:
            assert 0 <= sl < self.n_slices
            k_tmp = self._k_t[sl]
            self._t_min = k_tmp[k_tmp >= 0].min()
            self._t_max = k_tmp[k_tmp >= 0].max()
        else:
            self._t_min = None
            self._t_max = None
        self._sl = sl

    @abstractmethod
    def crop_readout_oversampling(self):
        pass

    def whiten(self):
        if self.noise.size == 0 or np.abs(self.noise).sum() == 0:
            return

        noise_temp = np.reshape(self.noise.astype(np.complex128), (self.noise.shape[0], -1))
        cov = (noise_temp @ np.conj(noise_temp.T)) / (noise_temp.shape[1] - 1)
        chol = np.linalg.cholesky(cov)
        chol_inv = np.linalg.inv(chol)

        k_temp = np.moveaxis(self._k, 2, 0)  # move coil dimension to the front
        s = k_temp.shape
        k_temp = np.reshape(k_temp, (k_temp.shape[0], -1))
        k_temp = np.matmul(chol_inv, k_temp)
        k_temp = k_temp / np.linalg.norm(k_temp) * np.linalg.norm(self._k)
        k_temp = np.reshape(k_temp, s)
        k_temp = np.moveaxis(k_temp, 0, 2)  # move coil dimension back
        self._k = k_temp.astype(self._k.dtype)


class MRDDataset(Dataset):
    def __init__(self, filename: Path | str, apodize: bool = False):
        super().__init__(filename)

        self.apodize = apodize

        self.read_kspace()
        self.apply_phase_padding()
        self.read_physio()

    def read_kspace(self):
        if not os.path.isfile(self.filename):
            raise FileNotFoundError(f'File not found: {self.filename}')

        logging.info(f'Loading file {self.filename}')

        dset = ismrmrd.Dataset(self.filename, 'dataset', create_if_needed=False)
        self.header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
        enc = self.header.encoding[0]
        nAcq = dset.number_of_acquisitions()
        assert isinstance(nAcq, int)

        # temporal resolution
        self.tres = self.header.sequenceParameters.TR[0]  # type: ignore

        # matrix size
        eNx = enc.encodedSpace.matrixSize.x  # type: ignore
        eNy = enc.encodingLimits.kspace_encoding_step_1.maximum + 1  # type: ignore
        eNz = enc.encodedSpace.matrixSize.z  # type: ignore
        rNx = enc.reconSpace.matrixSize.x  # type: ignore
        rNy = enc.reconSpace.matrixSize.y  # type: ignore
        rNz = enc.reconSpace.matrixSize.z  # type: ignore
        assert eNz == 1
        assert rNz == 1
        if rNx == eNx:
            self.recon_size = (rNx, rNy // 2)
        else:
            self.recon_size = (rNx, rNy)

        # read number of Slices, Reps, Contrasts, etc.
        nSlices = enc.encodingLimits.slice.maximum + 1  # type: ignore
        nPhases = enc.encodingLimits.phase.maximum + 1  # type: ignore
        nCoils = self.header.acquisitionSystemInformation.receiverChannels  # type: ignore
        nAverage = enc.encodingLimits.average.maximum + 1  # type: ignore
        nSets = enc.encodingLimits.set.maximum + 1  # type: ignore
        nReps = enc.encodingLimits.repetition.maximum + 1  # type: ignore
        if nCoils is None:
            raise RuntimeError('Number of receiver channels is not specified in the header')
        if nAverage > 1:
            logging.warning(f'Dataset contains {nAverage} averages, all but the first will be ignored!')
        if nSets > 1:
            logging.warning(f'Dataset contains {nSets} sets, all but the first will be ignored!')

        assert (nReps == 1) ^ (nPhases == 1), 'Either nReps or nPhases must be 1'
        t_dim = 'repetition' if nReps > 1 else 'phase'

        # noise data
        firstacq = 0
        noise_acquisitions = []
        noise_dwelltime_us = None
        for acqnum in range(nAcq):  # type: ignore
            acq = dset.read_acquisition(acqnum)
            if acq.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
                noise_acquisitions.append(acq.data)
                noise_dwelltime_us = noise_dwelltime_us or acq.getHead().sample_time_us
            else:
                firstacq = acqnum
                break
        self.noise = np.stack(noise_acquisitions, -1)  # [coil, readout, acquisition]

        # asymmetric echo
        kx_pre = 0
        acq = dset.read_acquisition(firstacq)
        head = acq.getHead()
        if head.center_sample*2 < eNx:
            kx_pre = eNx - head.number_of_samples

        # initialize a storage array
        nTime = nPhases if t_dim == 'phase' else nReps
        self._k = np.zeros((nSlices, nTime, nCoils, eNx, eNy), dtype=np.complex64)
        self._k_t = np.full((nSlices, nTime, eNy), -1, dtype=np.float64)

        # check if pilot tone (PT) is on
        pilottone = 0
        try:
            if (self.header.userParameters.userParameterLong[3].name == 'PilotTone'):  # type: ignore
                pilottone = self.header.userParameters.userParameterLong[3].value  # type: ignore
        except:
            pass

        # loop through the rest of the acquisitions
        acquisition_dwelltime_us = None
        for acqnum in tqdm(range(firstacq, nAcq), desc='Reading acquisitions'):
            acq = dset.read_acquisition(acqnum)
            head = acq.getHead()
            if head.idx.average > 0:
                continue
            if head.idx.set > 0:
                continue

            if pilottone == 1:
                # discard the first three and the last k-space points
                acq.data[:, :3] = 0
                acq.data[:, -1] = 0

            acquisition_dwelltime_us = acquisition_dwelltime_us or head.sample_time_us

            # stuff into the buffer
            sli = head.idx.slice
            time = head.idx.phase if t_dim == 'phase' else head.idx.repetition
            ky = head.idx.kspace_encode_step_1
            self._k[sli, time, :, kx_pre:, ky] = acq.data
            self._k_t[sli, time, ky] = head.acquisition_time_stamp * 2.5

        # close dataset
        dset.close()

        # scale noise data
        if self.noise.size > 0 and noise_dwelltime_us and acquisition_dwelltime_us:
            self.noise *= np.sqrt(noise_dwelltime_us / acquisition_dwelltime_us)

    def apply_phase_padding(self):
        enc = self.header.encoding[0]
        enc_mat_y = enc.encodedSpace.matrixSize.y  # type: ignore
        enc_lim_center = enc.encodingLimits.kspace_encoding_step_1.center  # type: ignore
        enc_lim_max = enc.encodingLimits.kspace_encoding_step_1.maximum  # type: ignore
        pad_left = enc_mat_y // 2 - enc_lim_center
        pad_right = enc_mat_y - pad_left - enc_lim_max - 1

        if pad_left < 0 or pad_right < 0:
            raise ValueError('Phase padding is negative')
        if pad_left == 0 and pad_right == 0:
            return

        self._k = np.pad(self._k, ((0, 0),) * 4 + ((pad_left, pad_right),), mode='constant', constant_values=0)
        self._k_t = np.pad(self._k_t, ((0, 0),) * 2 + ((pad_left, pad_right),), mode='constant', constant_values=-1)

    def read_physio(self):
        if not os.path.isfile(self.filename):
            raise FileNotFoundError(f'File not found: {self.filename}')

        logging.info(f'Loading file {self.filename}')

        dset = ismrmrd.Dataset(self.filename, 'dataset', create_if_needed=False)
        nWav = dset.number_of_waveforms()

        physio_data = defaultdict(list)
        physio_timestamps = defaultdict(list)
        for i in tqdm(range(nWav), 'Reading waveforms   '):  # type: ignore
            wav = dset.read_waveform(i)
            head = wav.getHead()
            wav_id = head.waveform_id
            if wav_id >= 5:
                continue
            physio_data[wav_id].append(wav.data.T)
            timestamps = head.time_stamp * 2.5 + np.arange(wav.data.shape[1]) * head.sample_time_us / 1000
            physio_timestamps[wav_id].append(timestamps)
        for wav_id in physio_data:
            data = np.concatenate(physio_data[wav_id], axis=0)
            timestamps = np.concatenate(physio_timestamps[wav_id])
            self._physio[wav_id] = PhysioSignal(data, timestamps)

        # close dataset
        dset.close()

    def crop_readout_oversampling(self):
        self._k = crop_readout_oversampling(self._k, ro_axis=3, apodize=self.apodize)
        self.noise = crop_readout_oversampling(self.noise, ro_axis=1, apodize=self.apodize)


class PhantomDataset(Dataset):
    def __init__(self, filename: Path | str, apodize: bool = False, acceleration_rate: float = 8, snr: float = np.inf):
        super().__init__(filename)

        self.apodize = apodize
        self.acceleration_rate = acceleration_rate
        self.snr_db = snr
        self.ground_truth: np.ndarray

        self.read_data()

    def read_data(self):
        if not os.path.isfile(self.filename):
            raise FileNotFoundError(f'File not found: {self.filename}')
        logging.info(f'Loading file {self.filename}')
        data = scipy.io.loadmat(self.filename)['D']

        # create empty MRD header
        self.header = ismrmrd.xsd.ismrmrdHeader()

        # load coil images and sensitivity maps
        img = data['img'].item().astype(np.complex64)  # [x, y, frame, coil]
        img = img.transpose(2, 3, 0, 1)[None]  # [slice=1, frame, coil, x, y]

        # apodize
        if self.apodize:
            window = scipy.signal.windows.tukey(img.shape[3], alpha=0.2)
            img *= window[None, None, None, :, None]

        # coil sensitivity maps
        kspace_avg = average_data(fftnc(img, axes=(-2, -1)), 1)  # type: ignore  # [slice, coil, kx, ky]
        kspace_avg = np.moveaxis(kspace_avg, 1, 0)  # [coil, slice, kx, ky]
        sens_maps = sigpy.mri.app.EspiritCalib(kspace_avg, calib_width=24, thresh=0.02, crop=0, show_pbar=False).run()  # [coil, slice, x, y]
        assert isinstance(sens_maps, np.ndarray), 'ESPIRiT failed'
        sens_maps = np.moveaxis(sens_maps, 0, 1)[:, None]  # [slice, frame=1, coil, x, y]

        # combine coil images
        self.ground_truth = np.sum(img * sens_maps.conj(), axis=2)  # [slice, frame, x, y]

        # add noise
        # SNR = 10 * log10( ||k||_2^2 / M / sigma^2 )
        # sigma^2 = ||k||_2^2 / M / 10^(SNR/10)
        noise_var = np.linalg.norm(img)**2 / img.size / 10**(self.snr_db / 10)
        noise_std = np.sqrt(noise_var / 2)  # factor of 2 because noise is complex
        self.noise = np.random.default_rng().normal(loc=0, scale=noise_std, size=(2, *img.shape))
        self.noise = self.noise[0] + 1j * self.noise[1]  # make it complex
        img += self.noise
        self.noise = fftnc(self.noise, axes=(-2, -1))  # type: ignore  # [slice, frame, coil, kx/readout, ky/phase]
        self.noise = self.noise.transpose(2, 3, 0, 1, 4)  # [coil, kx/readout, slice, frame, ky/phase]
        self.noise = self.noise.reshape(*self.noise.shape[:2], -1)  # [coil, kx/readout, acquisition]

        # save noisy image
        self.noisy_img = np.sum(img * sens_maps.conj(), axis=2)  # [slice, frame, x, y]

        # save k-space
        self._k = fftnc(img, axes=(-2, -1))  # type: ignore  # [slice, frame, coil, kx, ky]
        self._k_t = np.full((*self._k.shape[:2], self._k.shape[4]), -1, dtype=np.float64)  # [slice, frame, ky]

        # temporal resolution
        par = data['par'].item()
        cardiac_cycles = par['card'].item().squeeze() * 1000  # ms
        duration = cardiac_cycles.sum()
        self.tres = duration / img.shape[1]

        # recon size
        self.recon_size = img.shape[-2:]

        # prepare physio data
        self.create_physio(cardiac_cycles)

        # generate mask, populate self._k_t, and downsample self._k
        gro_param = GROParam(PE=img.shape[-1], FR=img.shape[1], n=int(round(img.shape[-1] / self.acceleration_rate)))
        mask = GRO(gro_param).T  # [echo/slice=1, frame, ky]
        acquisition_time_stamps = np.linspace(0, duration, np.count_nonzero(mask), endpoint=False)  # ms
        self._k_t[mask == 1] = acquisition_time_stamps
        self._k *= mask[:, :, None, None, :]

    def create_physio(self, cardiac_cycles: np.ndarray, beats_per_resp_cycle: int = 5):
        # ECG
        ecg = []
        for cycle in cardiac_cycles:
            steps = int(round(cycle / (self.tres / 1000)))
            ecg.append(np.cos(np.linspace(0, 2 * np.pi, steps, endpoint=False)))
        ecg = np.concatenate(ecg)[:, None]  # [time, channel]
        t_ecg = np.linspace(0, sum(cardiac_cycles), ecg.shape[0], endpoint=False)  # ms
        self._physio[0] = PhysioSignal(ecg, t_ecg)

        # RESP
        resp = []
        for i in range(0, len(cardiac_cycles), beats_per_resp_cycle):
            steps = int(round(sum(cardiac_cycles[i:i + beats_per_resp_cycle]) / (self.tres / 1000)))
            resp.append(np.cos(np.linspace(0, 2 * np.pi, steps, endpoint=False)))
        resp = np.concatenate(resp)[:, None]  # [time, channel]
        t_resp = np.linspace(0, sum(cardiac_cycles), resp.shape[0], endpoint=False)  # ms
        self._physio[2] = PhysioSignal(resp, t_resp)

    def crop_readout_oversampling(self):
        # do nothing, phantom data does not have readout oversampling
        pass
