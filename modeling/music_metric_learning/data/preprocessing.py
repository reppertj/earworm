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

# Required because as of 0.7.2 on OSX, torchaudio links its own OpenMP runtime in addition to pytorch
# This tells OpenMP not to crash when this happens.
if sys.platform == "darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class AudioTooShortError(ValueError):
    pass


class TensorPreprocesser:
    def __init__(self, model, return_two: bool = False) -> None:
        self.model = model
        self.return_two = return_two

    def __call__(
        self, in_paths: List[str], out_paths: List[str], mfcc=False, n_workers=2
    ) -> Tuple[List[str], List[str]]:
        def write_out(inp, out):
            try:
                if os.path.exists(out):
                    print("Skipping:", out)
                    return (out, True)
                tensors = self.model.from_path(
                    inp, return_two=self.return_two, return_mfcc=mfcc
                )
                dir = os.path.dirname(out)
                if not os.path.exists(dir):
                    os.mkdir(dir)
                torch.save(tensors, out)
                print("Success:", out)
                return (out, True)
            except Exception as e:
                print("Failure:", e, out)
                return (out, False)

        with ThreadPool(nodes=n_workers) as P:
            results = P.uimap(write_out, in_paths, out_paths)
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
        Returns log10 (dB/10) scaled mel spectrograms from random crop of length `seconds` from
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
        # if self.validate_target:
        #     assert sgram.shape == self.validate_target
        log_spec = torch.log(torch.clamp(sgram, min=1e-10)).div(
            torch.log(torch.tensor(10.0))
        )
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

    def from_path(self, path: str, return_two=False, return_mfcc=False):
        logger.info(f"Processing {path} to tensor of spectrogram")
        waveform, inp_freq = torchaudio.load(path)
        waveform = waveform.mean(dim=0, keepdims=True)
        waveform = Resample(inp_freq, self.sample_rate)(waveform)
        n_samples = waveform.shape[1]
        min_samples = self.min_samples * 2 if return_two else self.min_samples
        if n_samples < min_samples:
            raise AudioTooShortError(
                f"Input must be at least {self.seconds * 2 if return_two else self.seconds} seconds long"
            )
        first_idx = torch.randint(0, n_samples - min_samples, (1,))
        first_waveform = waveform[:, first_idx : (first_idx + self.min_samples)]  # type: ignore
        first_sgram = self.forward(first_waveform)
        if return_mfcc:
            first_mfcc = self.get_mfcc(first_waveform)
        if return_two:
            second_idx = torch.randint(first_idx.item() + self.min_samples, n_samples - self.min_samples, (1,))  # type: ignore
            second_waveform = waveform[:, second_idx : (second_idx + self.min_samples)]
            second_sgram = self.forward(second_waveform)
            if return_mfcc:
                second_mfcc = self.get_mfcc(second_waveform)
                return first_sgram, first_mfcc, second_sgram, second_mfcc
            return first_sgram, second_sgram
        if return_mfcc:
            return first_sgram, first_mfcc
        return first_sgram
