import torch
from torch import Tensor
from torchaudio.functional import fftconvolve
from typing import NamedTuple


class Mix(NamedTuple):
    noisy: Tensor
    reverberant: Tensor
    noise: Tensor
    gain: Tensor


def reverberant_mix(
    source: Tensor,
    ref_rir: Tensor,
    noise: Tensor,
    snr_dB: float,
) -> Mix:
    """
    Mix source with reverberation and noise at the desired SNR.

    Args:
        source (Tensor):
            Dry source signal
        ref_rir (Tensor):
            Reference RIR
        noise (Tensor):
            Noise signal
        snr_dB (float):
            Desired signal-to-noise ratio in dB
    Returns:
        Mix:
            A named tuple containing the noisy, reverberant, noise, and gain signals.
    """
    L_noise = noise.shape[0]
    L_conv = source.shape[0] + ref_rir.shape[0] - 1

    assert L_noise == L_conv, "Signals lengths do not match."

    reverberant_signal = fftconvolve(source, ref_rir, mode="full")

    sigma_noise_2 = noise.pow(2).mean()
    sigma_reverb_2 = reverberant_signal.pow(2).mean()
    snr = 10 ** (-snr_dB / 10)

    noise_gain = (sigma_reverb_2 / sigma_noise_2 * snr).sqrt()
    noise = noise * noise_gain

    return Mix(
        noisy=reverberant_signal + noise,
        reverberant=reverberant_signal,
        noise=noise,
        gain=noise_gain,
    )
