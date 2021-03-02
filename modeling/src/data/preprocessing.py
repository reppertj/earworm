import sys
import os
import logging
from typing import Iterable, Tuple

import numpy as np
import torch
from torch import nn
import torchaudio
from torchaudio.transforms import MFCC, MelSpectrogram, Resample, AmplitudeToDB
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


class JittableLogmelFilterBank(torchlibrosa.LogmelFilterBank):
    """
    log10 is not in the ONNX opset; subclass and override a method that uses it. :(
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def power_to_db(self, input):
        r"""Power to db, this function is the pytorch implementation of
        librosa.power_to_lb
        """
        ref_value = self.ref
        log_spec = (
            10.0
            * torch.log(torch.clamp(input, min=self.amin, max=np.inf))
            / torch.log(torch.tensor(10.0))
        )
        log_spec -= 10.0 * np.log10(np.maximum(self.amin, ref_value))

        if self.top_db is not None:
            if self.top_db < 0:
                raise ValueError("top_db must be non-negative")
            log_spec = torch.clamp(
                log_spec, min=log_spec.max().item() - self.top_db, max=np.inf
            )

        return log_spec


class JittableWaveformtoTensor(nn.Module):
    def __init__(
        self,
        sample_rate: int = 15950,
        seconds: float = 10.0,
        win_length: int = 1024,
        hop_length: int = 256,
        n_mels: int = 128,
    ):
        """
        Jittable class for onnx export. Zero centers and normalizes, resamples, and
        returns mel spectrogram.
        Does NOT trim length or standardize outputs.
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.seconds = seconds
        self.min_samples = int(sample_rate * seconds)
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.mel_spectogram = nn.Sequential(
            torchlibrosa.Spectrogram(
                n_fft=self.win_length,
                hop_length=self.hop_length,
                win_length=self.win_length,
            ),
            JittableLogmelFilterBank(
                sr=self.sample_rate,
                n_fft=self.win_length,
                n_mels=self.n_mels,
            ),
        )

    def forward(self, waveform):
        # Input shape (n_channels, time)
        # Zero center and normalize to [-1, 1] to match torchaudio.load()
        waveform = waveform.sub(waveform.mean())
        waveform = waveform.div(waveform.abs().max())
        waveform = waveform.mean(dim=0, keepdims=True)
        sgram = self.mel_spectogram(waveform)
        return sgram.transpose(2, 3).squeeze(0)


class SoundtoTensor(object):
    def __init__(
        self,
        sample_rate: int = 15950,
        seconds: float = 10.0,
        win_length: int = 1024,
        hop_length: int = 256,
        n_mels: int = 128,
        validate_target: Tuple[int] = (1, 128, 624),
    ):
        """
        Yields dB-scaled mel spectrograms from random crop of length `seconds` from audio files.
        Output size depends on parameters; pass shape to `validate_target` to raise an error if the
        output tensor does not match. Spectrograms are NOT standardized for friendlier gradients;
        this should be done by the relevant dataset class or inference code.
        """
        self.sample_rate = sample_rate
        self.seconds = seconds
        self.min_samples = int(sample_rate * seconds)
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.validate_target = validate_target

        self.mel_spectogram = nn.Sequential(
            torchlibrosa.Spectrogram(
                n_fft=self.win_length,
                hop_length=self.hop_length,
                win_length=self.win_length,
            ),
            torchlibrosa.LogmelFilterBank(
                sr=self.sample_rate,
                n_fft=self.win_length,
                n_mels=self.n_mels,
            ),
        )

        # self.mel_spectogram = MelSpectrogram(
        #     sample_rate=self.sample_rate,
        #     win_length=self.win_length,
        #     hop_length=self.hop_length,
        #     n_fft=self.win_length,
        #     n_mels=self.n_mels,
        # )
        # self.amplitude_to_db = AmplitudeToDB(top_db=80)

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

    def __call__(self, paths: Iterable[str], return_mfcc=False):
        for path in paths:
            logger.info(f"Processing {path} to tensor of spectrogram")
            waveform, inp_freq = torchaudio.load(path)
            waveform = waveform.mean(dim=0, keepdims=True)
            waveform = Resample(inp_freq, self.sample_rate)(waveform)
            n_samples = waveform.shape[1]
            if n_samples < self.min_samples:
                raise AudioTooShortError("Input must be at least 10 seconds long")
            start_idx = torch.randint(0, n_samples - self.min_samples, (1,))
            waveform = waveform[:, start_idx : (start_idx + self.min_samples)]
            if return_mfcc:
                mfcc = self.get_mfcc(waveform)
            sgram = self.mel_spectogram(waveform).squeeze(0).transpose(1, 2)
            # sgram = self.amplitude_to_db(sgram)
            if self.validate_target and sgram.shape != self.validate_target:
                raise ValueError(
                    f"Out tensor of {sgram.shape} does not match target of {self.validate_target}"
                )
            if return_mfcc:
                yield sgram, mfcc
            else:
                yield sgram
