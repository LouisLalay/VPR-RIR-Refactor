from .cbf import CTF
from torch import Tensor
from torch.fft import rfft, irfft
from typing import cast
import torch


class Baseline1:
    """
    Oracle baseline.
    """

    def __init__(
        self,
        source: Tensor,
        reverberant_signal: Tensor,
        reference_rir: Tensor,
    ):
        self.reference_rir = reference_rir

    def fit_experiment(self):
        pass

    def get_estimates(self):
        return self.reference_rir, {}


def validate_baseline1_kwargs(kwargs: dict):
    return {}


class Baseline2:
    """
    Spectral deconvolution of the entire signal.
    """

    def __init__(
        self,
        source: Tensor,
        reverberant_signal: Tensor,
        reference_rir: Tensor,
    ):
        self.Ly = reverberant_signal.shape[-1]
        self.source = source
        self.reverberant_signal = reverberant_signal
        self.Lh = self.Ly - source.shape[-1] + 1

    def fit_experiment(self):
        source_fourier = cast(Tensor, rfft(self.source, self.Ly))
        reverberant_fourier = cast(Tensor, rfft(self.reverberant_signal, self.Ly))
        estimated_fourier = reverberant_fourier / source_fourier
        estimated_rir = cast(Tensor, irfft(estimated_fourier, self.Ly))
        self.estimated_rir = estimated_rir[: self.Lh].real

    def get_estimates(self):
        return self.estimated_rir, {}


def validate_baseline2_kwargs(kwargs: dict):
    return {}


class Baseline3:
    """
    Crossband filtering baseline.
    """

    def __init__(
        self,
        source: Tensor,
        reverberant_signal: Tensor,
        reference_rir: Tensor,
        nfft: int,
        hop_length: int,
        num_absolute_cross_bands: int,
    ):
        self.source = source
        self.reverberant_signal = reverberant_signal

        self.nfft = nfft
        self.hop_length = hop_length
        self.num_absolute_cross_bands = num_absolute_cross_bands

        Lh = reverberant_signal.shape[-1] - source.shape[-1] + 1
        source_frames = (source.shape[-1] - nfft) // hop_length + 1
        max_rir_frames = (source_frames - 1) // (2 * num_absolute_cross_bands + 1)
        max_Lh = (max_rir_frames - 1) * hop_length
        self.Lh = min(Lh, max_Lh)

    def fit_experiment(self):
        window = torch.hann_window(
            self.nfft,
            device=self.source.device,
            dtype=self.source.dtype,
        )

        source_stft = torch.stft(
            self.source,
            n_fft=self.nfft,
            hop_length=self.hop_length,
            win_length=self.nfft,
            window=window,
            return_complex=True,
        )

        reverberant_stft = torch.stft(
            self.reverberant_signal,
            n_fft=self.nfft,
            hop_length=self.hop_length,
            win_length=self.nfft,
            window=window,
            return_complex=True,
        )

        # CTF expects batch and channel dimensions:
        # [batch, channel, frequency, time]
        source_stft = source_stft.unsqueeze(0).unsqueeze(0)
        reverberant_stft = reverberant_stft.unsqueeze(0).unsqueeze(0)

        # Determine the number of output frames required by CTF.
        dummy = torch.stft(
            torch.zeros(self.Lh),
            n_fft=self.nfft,
            hop_length=self.hop_length,
            win_length=self.nfft,
            window=window,
            return_complex=True,
        )

        ctf = CTF(
            output_len=dummy.shape[-1] + 1,
            num_absolute_cross_bands=self.num_absolute_cross_bands,
        ).to(self.source.device)

        estimated_stft = ctf.forward(
            Y_tf=reverberant_stft,
            S_tf=source_stft,
        ).squeeze()

        self.estimated_rir: Tensor = torch.istft(
            estimated_stft,
            n_fft=self.nfft,
            hop_length=self.hop_length,
            win_length=self.nfft,
            window=window,
            length=self.Lh,
        )

    def get_estimates(self):
        return self.estimated_rir, {}


def validate_baseline3_kwargs(kwargs: dict):
    required_keys = {
        "nfft": int,
        "hop_length": int,
        "num_absolute_cross_bands": int,
    }

    validated_kwargs = {}
    for key, expected_type in required_keys.items():
        if key not in kwargs:
            raise ValueError(f"Missing required argument: {key}")
        value = kwargs[key]
        try:
            converted = int(value)
        except ValueError:
            raise ValueError(f"Arg {key} is not an integer: {value}")
        validated_kwargs[key] = converted

    return validated_kwargs
