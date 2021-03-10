import sys
import os
import logging
from typing import Iterable, List, Tuple

import numpy as np
import torch
from torch import nn
import torchaudio
from pathos.threading import ThreadPool
from torchaudio.transforms import MFCC, Resample
import torchlibrosa

logger = logging.getLogger()

# Use sox_io backend if available
if (
    torchaudio.get_audio_backend() != "sox_io"
    and "sox_io" in torchaudio.list_audio_backends()
):
    torchaudio.set_audio_backend("sox_io")
    logger.debug("Set audio backend to sox_io")

# Required because as of 0.7.2, torchaudio links its own OpenMP runtime in addition to pytorch
# This tells OpenMP not to crash when this happens.
if sys.platform == "darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class AudioTooShortError(ValueError):
    pass


# class JittableLogmelFilterBank(torchlibrosa.LogmelFilterBank):
#     """
#     Cannot have literal infinity in the model graph for tensorflow.js 32bit
#     """

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def power_to_db(self, input):
#         r"""Power to db, this function is the pytorch implementation of
#         librosa.power_to_lb
#         """
#         GIANT_32B_UFLOAT = torch.tensor(2.0 ** 126, dtype=torch.float32)
#         ref_value = self.ref
#         log_spec = (
#             10.0
#             * torch.log(torch.clamp(input, min=self.amin, max=GIANT_32B_UFLOAT))
#             / torch.log(torch.tensor(10.0))
#         )
#         log_spec -= 10.0 * np.log10(np.maximum(self.amin, ref_value))

#         if self.top_db is not None:
#             if self.top_db < 0:
#                 raise ValueError("top_db must be non-negative")
#             log_spec = torch.clamp(
#                 log_spec, min=log_spec.max().item() - self.top_db, max=GIANT_32B_UFLOAT
#             )

#         return log_spec


# class JittableWaveformtoTensor(nn.Module):
#     def __init__(
#         self,
#         sample_rate: int = 15950,
#         seconds: float = 10.0,
#         win_length: int = 1024,
#         hop_length: int = 256,
#         n_mels: int = 128,
#     ):
#         """
#         Jittable class for onnx export. Zero centers and normalizes, resamples, and
#         returns mel spectrogram.
#         Does NOT trim length or standardize outputs.
#         """
#         super().__init__()
#         self.sample_rate = sample_rate
#         self.seconds = seconds
#         self.min_samples = int(sample_rate * seconds)
#         self.win_length = win_length
#         self.hop_length = hop_length
#         self.n_mels = n_mels
#         self.sample_input = torch.randn(
#             (1, int(self.sample_rate * self.seconds)), dtype=torch.float32
#         )

#         self.mel_spectogram = nn.Sequential(
#             torchlibrosa.Spectrogram(
#                 n_fft=self.win_length,
#                 hop_length=self.hop_length,
#                 win_length=self.win_length,
#             ),
#             JittableLogmelFilterBank(
#                 sr=self.sample_rate,
#                 n_fft=self.win_length,
#                 n_mels=self.n_mels,
#             ),
#         )

#     def forward(self, waveform):
#         # Input shape (n_channels, time)
#         # Zero center and normalize to [-1, 1] to match torchaudio.load()
#         # TODO: Normalize IN THE AUDIO PREPROCESSING IN THE MODEL!! THE JS SHOULD SEND AN AUDIO
#         # data tensor NOT normalized in any way
#         waveform = waveform.sub(waveform.mean())
#         waveform = waveform.div(waveform.abs().max())
#         waveform = waveform.mean(dim=0, keepdims=True)
#         sgram = self.mel_spectogram(waveform)
#         return sgram.transpose(2, 3).squeeze(0)


class TensorPreprocesser:
    def __init__(self, model) -> None:
        self.model = model

    def __call__(
        self, in_paths: List[str], out_paths: List[str], mfcc=False, n_workers=2
    ) -> Tuple[List[str], List[str]]:
        def write_out(inp, out):
            try:
                tensor = self.model.from_path(inp, mfcc)
                dir = os.path.dirname(out)
                if not os.path.exists(dir):
                    os.mkdir(dir)
                torch.save(tensor, out)
                print("Success:", out)
                return (out, True)
            except:
                print("Failure:", out)
                return (out, False)

        with ThreadPool(nodes=n_workers) as P:
            results = P.imap(write_out, in_paths, out_paths)
        successes = [path for path, res in results if res]
        failures = [path for path, res in results if not res]
        return successes, failures


class DoubleSqueeze(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.squeeze().squeeze()


class WaveformToDBSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate: int = 22050,
        seconds: float = 3.0,
        win_length: int = 1024,
        hop_length: int = 512,
        n_mels: int = 128,
        validate_target: Tuple[int, int] = (128, 130),
    ):
        """
        Returns log10 (not dB) scaled mel spectrograms from random crop of length `seconds` from
        audio files.
        Output size depends on parameters; pass shape to `validate_target` to raise an error if the
        output tensor does not match. Spectrograms are NOT standardized for friendlier gradients;
        this should be done by the relevant dataset class or inference code.
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.seconds = seconds
        self.min_samples = int(sample_rate * seconds)
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.validate_target = validate_target
        self.sample_input = torch.randn(
            (1, int(self.seconds * self.sample_rate)), dtype=torch.float32
        )
        self.mel_spectogram = nn.Sequential(
            torchlibrosa.Spectrogram(
                n_fft=self.win_length,
                center=True,
                hop_length=self.hop_length,
                win_length=self.win_length,
            ),
            # Because tf.js reduces dims on BMM (the last op from spectrogram),
            # the next line would fail in the tf.js model unless we explicitly
            # squeeze here, so the shapes line up between pytorch, tf, and tf.js.
            # Without this, tf.js will error on the next op, because it expects
            # 4 input dims but only has two as a result of its BMM behavior.
            # We could maybe get around this by performing an explicit BMM in
            # spectrogram; right now, it relies on pytorch's broadcasting semantics,
            # (i.e., matmul becomes batch matmul implicitly), and I think this is
            # what gets lost in translation.
            DoubleSqueeze(),
            torchlibrosa.LogmelFilterBank(
                sr=self.sample_rate,
                n_fft=self.win_length,
                n_mels=self.n_mels,
                fmin=40,
                fmax=(self.sample_rate // 2) + 40,
                is_log=False,
            ),
        )

    def forward(self, waveform: torch.Tensor):
        """
        Input: (1, self.sample_rate * self.seconds)
        Output: self.validate_target
        This is intended mainly for use in the frontend through export to tf.js
        """
        sgram = self.mel_spectogram(waveform).t()
        if self.validate_target:
            assert sgram.shape == self.validate_target
        log_spec = torch.log(torch.clamp(sgram, min=1e-10)).div(torch.log(torch.tensor(10.)))
        return log_spec
        # return torch.log(torch.clamp(sgram, min=1e-10)).div(
        #     torch.log(torch.tensor(10.0))
        # )
        # return torch.log10(1 + sgram * 10)

    def get_mfcc(self, waveform: torch.Tensor):
        melkwargs = {
            "win_length": self.win_length,
            "hop_length": self.hop_length,
            "n_fft": self.win_length,
            "n_mels": self.n_mels,
        }
        coefs = MFCC(self.sample_rate, n_mfcc=30, melkwargs=melkwargs)(waveform)
        means = coefs.mean(dim=2)
        stds = coefs.std(dim=2)
        return torch.cat((means, stds), dim=1)

    def from_path(self, path: str, return_mfcc=False):
        logger.info(f"Processing {path} to tensor of spectrogram")
        waveform, inp_freq = torchaudio.load(path)
        waveform = waveform.mean(dim=0, keepdims=True)
        waveform = Resample(inp_freq, self.sample_rate)(waveform)
        n_samples = waveform.shape[1]
        if n_samples < self.min_samples:
            raise AudioTooShortError(
                f"Input must be at least {self.seconds} seconds long"
            )
        start_idx = torch.randint(0, n_samples - self.min_samples, (1,))
        waveform = waveform[:, start_idx : (start_idx + self.min_samples)]  # type: ignore
        sgram = self.forward(waveform)
        if return_mfcc:
            return sgram, self.get_mfcc(waveform)
        else:
            return self.forward(waveform)
