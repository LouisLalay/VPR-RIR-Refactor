import torch
from typing import cast
from torch import Tensor
from warnings import warn
from scipy.interpolate import PchipInterpolator

from . import constants


def create_linear_RT_profile(
    rt60_profile: Tensor,
    frequency_bands: Tensor,
    n_fft: int,
) -> Tensor:
    """
    Create a linear RT60 profile.

    Args:
        rt60_profile (Tensor):
            The RT60 profile tensor of shape (n_f_bands) in seconds.
        frequency_bands (Tensor):
            The frequency bands tensor of shape (n_f_bands) containing the center frequencies of the bands in Hz.
        n_fft (int):
            The number of FFT points for the interpolation.
    Returns:
        Tensor:
            A tensor of shape (n_fft) containing the interpolated RT60 values for each FFT point.
    Raises:
        AssertionError:
            If the shapes of rt60_profile and frequency_bands do not match.
    """
    assert (
        rt60_profile.shape == frequency_bands.shape
    ), "RT60 profile and frequency bands must have the same shape."

    linear_freqs = torch.linspace(0, frequency_bands[-1], n_fft // 2 + 1)
    interpolator = PchipInterpolator(frequency_bands.numpy(), rt60_profile.numpy())
    linear_rt_profile = interpolator(linear_freqs)
    return torch.from_numpy(linear_rt_profile)


def rt_profile_to_alpha(linear_rt_profile: Tensor) -> Tensor:
    """
    Convert a linear RT60 profile to SWFT parameters.

    Args:
        linear_rt_profile (Tensor):
            The linear RT60 profile tensor of shape (n_fft).
    Returns:
        Tensor:
            The tensor $alpha(f)$ of shape (n_fft) containing the SWFT parameters for each FFT point.
    """

    return 3 * torch.log(torch.tensor(10.0)) / (linear_rt_profile)


def alpha_to_p_fourier(alpha: Tensor, sr: int) -> Tensor:
    """
    Convert the SWFT parameters to the frequency-domain decay operator.

    Args:
        alpha (Tensor):
            The tensor $alpha(f)$ of shape (n_fft) containing the SWFT parameters for each FFT point.
        sr (int):
            The sample rate in Hz.
    Returns:
        Tensor:
            The tensor $P(f)$ of shape (n_fft) containing the frequency-domain decay operator for each FFT point.
    """
    return torch.exp(-alpha / (sr))


def init_taylor_mul(Lh: int) -> tuple[int, Tensor, Tensor]:
    """
    Initialize the Taylor multiplication operator.

    Args:
        Lh (int):
            The length of the RIR.
    Returns:
        int:
            nfft, the number of FFT points for the Taylor multiplication operator.
        Tensor:
            arange(Lh) used in the Taylor multiplication operator.
        Tensor:
            omega, the Fourier basis of size (nfft)
    """
    nfft = int(1.1 * Lh)
    taylor_arange = torch.arange(Lh, dtype=torch.get_default_dtype())
    omega = torch.exp(-2j * torch.pi * taylor_arange[: nfft // 2 + 1] / nfft)
    return nfft, taylor_arange, omega


def taylor_mul(
    epsilon: Tensor,
    taylor_arange: Tensor,
    omega: Tensor,
    p_fourier: Tensor,
    nfft: int,
    taylor_order: int = 500,
) -> Tensor:
    """
    Perform the product $P epsilon$ using the Taylor multiplication operator.

    Args:
        epsilon (Tensor):
            The input tensor of shape (Lh) to be multiplied by the decay operator.
        taylor_arange (Tensor):
            arange(Lh) used in the Taylor multiplication operator.
        p_fourier (Tensor):
            The tensor of shape (nfft//2+1) containing the frequency-domain decay operator
        omega (Tensor):
            The tensor of shape (nfft//2+1) containing the Fourier basis
        nfft (int):
            The number of FFT points for the Taylor multiplication operator.
        taylor_order (int, optional):
            The maximum order of the Taylor series expansion. Defaults to 500.
    Returns:
        Tensor:
            The output tensor $P epsilon$ of shape (Lh).
    Warnings:

    """
    Lh = epsilon.shape[0]
    delta_alpha_bar = 1e-4

    # Stabilization parameter
    pf_min = p_fourier.min()
    pf_max = p_fourier.max()
    # pf_bar = exp(-alpha_bar)
    pf_bar = (pf_min + pf_max) / 2 - delta_alpha_bar

    # Evaluate the Taylor series at the stabilization point
    decay = torch.exp(taylor_arange * torch.log(pf_bar))
    epsilon = epsilon * decay

    # Initialize the result tensor
    y_fourier = cast(Tensor, torch.fft.rfft(epsilon, nfft))

    # Initialize the Taylor series coefficients
    diff = p_fourier / pf_bar - 1
    diff = diff * omega
    diff_max = diff.abs().max()
    diff = diff / diff_max

    diff_power = torch.ones_like(diff)

    for k in range(1, min(taylor_order, Lh)):
        epsilon = (epsilon * taylor_arange[: Lh - k + 1])[1:] / k * diff_max
        diff_power = diff_power * diff
        next_term = cast(Tensor, torch.fft.rfft(epsilon, nfft)) * diff_power

        # Early stopping
        if next_term.abs().max() < constants.ALMOST_ZERO:
            break

        # Sanity check
        has_inf = torch.isinf(next_term).any()
        has_nan = torch.isnan(next_term).any()
        if has_inf or has_nan:
            warn(
                f"Taylor series diverged at order {k}. Stopping iteration. "
                f"Max abs of next term: {next_term.abs().max().item()}. "
                f"Has inf: {has_inf}. Has NaN: {has_nan}."
            )
            break

        # Update the result tensor
        y_fourier = y_fourier + next_term

    y = cast(Tensor, torch.fft.irfft(y_fourier, nfft))[:Lh]
    return y.real
