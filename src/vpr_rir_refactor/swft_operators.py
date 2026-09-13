from matplotlib.axes import Axes
from scipy.interpolate import PchipInterpolator
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter
from typing import cast
from warnings import warn
import matplotlib.pyplot as plt
import torch

from . import constants
from .torch_pchip import cubic_interp1d as torch_pchip


def plot_rt60_operator(alpha: Tensor, sr: int, ax: Axes):
    # RT60 profile
    rt_profile = alpha_to_rt_profile(alpha)
    freqs = torch.linspace(0, sr // 2, rt_profile.shape[0])
    ax.plot(rt_profile, freqs)
    ax.set_xlabel("RT60 (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("RT60 Operator $e^{-\\alpha(f)}/SR$")


def plot_mode_filter(g: Tensor, sr: int, ax: Axes):
    nfft = 2 ** (int(g.shape[0]).bit_length() + 1)
    modes = cast(Tensor, torch.fft.rfft(g, nfft))
    freqs = torch.linspace(0, sr // 2, modes.shape[0])
    ax.plot(modes.abs(), freqs)
    ax.set_xlabel("Modes amplitude")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Mode Filter $g(f)$")


def log_swft_parameters(
    logger: SummaryWriter,
    alpha: Tensor,
    g: Tensor,
    current_step: int,
    sr: int,
    tag: str = "SWFT_Parameters",
):

    fig = plt.figure(figsize=(12, 6))
    axs: list[Axes] = fig.subplots(1, 2)
    plot_rt60_operator(alpha, sr, axs[0])
    plot_mode_filter(g, sr, axs[1])

    logger.add_figure(tag=tag, figure=fig, global_step=current_step)


def plot_rirs(rir_ref: Tensor, rir_other: Tensor, sr: int, ax: Axes):
    L1 = rir_ref.shape[0]
    L2 = rir_other.shape[0]
    time = torch.arange(max(L1, L2)) / sr

    ax.plot(time[:L1], rir_ref, color="red", alpha=0.5, label="Reference")
    ax.plot(time[:L2], rir_other, label="Estimation")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.set_title("Estimated vs Reference RIR")


def plot_rir_diff(rir_ref: Tensor, rir_other: Tensor, sr: int, ax: Axes):
    L1 = rir_ref.shape[0]
    L2 = rir_other.shape[0]
    Lmin = min(L1, L2)
    time = torch.arange(Lmin) / sr

    diff = rir_ref[:Lmin] - rir_other[:Lmin]

    ax.plot(time, diff.abs())
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Difference between Estimated and Reference RIR")


def log_rirs(
    logger: SummaryWriter,
    rir_ref: Tensor,
    rir_other: Tensor,
    sr: int,
    current_step: int,
):
    fig = plt.figure(figsize=(12, 6))
    axs: list[Axes] = fig.subplots(1, 2)

    plot_rirs(rir_ref, rir_other, sr, axs[0])
    plot_rir_diff(rir_ref, rir_other, sr, axs[1])
    logger.add_figure(tag="RIRS", figure=fig, global_step=current_step)


def plot_epsilon(
    mu_epsilon: Tensor,
    r_epsilon: Tensor,
    sr: int,
    ax: Axes,
):
    Lh = mu_epsilon.shape[0]
    time = torch.arange(Lh) / sr

    ax.plot(time, mu_epsilon, label="Mean")
    ax.fill_between(
        time,
        mu_epsilon - r_epsilon.sqrt(),
        mu_epsilon + r_epsilon.sqrt(),
        alpha=0.5,
        label="Std Dev",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.set_title("Epsilon Distribution $\\epsilon(t)$")


def log_epsilon(
    logger: SummaryWriter,
    mu_epsilon: Tensor,
    r_epsilon: Tensor,
    current_step: int,
    sr: int,
    tag: str = "Epsilon",
):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca()
    plot_epsilon(mu_epsilon, r_epsilon, sr, ax)
    logger.add_figure(tag=tag, figure=fig, global_step=current_step)


def interpolate(x: Tensor, y: Tensor, x_new: Tensor, method: str = "pchip") -> Tensor:
    """
    A differentialble interpolation function.

    Args:
        x (Tensor):
            The input tensor of shape (n,) containing the x-coordinates of the data points.
        y (Tensor):
            The input tensor of shape (n,) containing the y-coordinates of the data points.
        x_new (Tensor):
            The input tensor of shape (m,) containing the x-coordinates of the interpolated points.
        method (str, optional):
            The interpolation method to use. Defaults to "pchip". "linear" is also supported.
    Returns:
        Tensor:
            The interpolated tensor of shape (m,) containing the y-coordinates of the interpolated points.
    Raises:
        ValueError:
            If the specified interpolation method is not supported.
    """

    match method:
        case "pchip":
            return torch_pchip(x, y, x_new)
        case "linear":
            return torch.nn.functional.interpolate(
                y.unsqueeze(0).unsqueeze(0),
                size=x_new.shape[0],
                mode="linear",
                align_corners=True,
            ).squeeze()
        case _:
            raise ValueError(f"Unsupported interpolation method '{method}'.")


def create_linear_RT_profile(
    rt60_profile: Tensor,
    center_freqs: Tensor,
    sr: int,
    n_fft: int,
    interpolation_method: str = "pchip",
) -> Tensor:
    """
    Create a linear RT60 profile.

    Args:
        rt60_profile (Tensor):
            The RT60 profile tensor of shape (n_f_bands) in seconds.
        center_freqs (Tensor):
            The center frequencies tensor of shape (n_f_bands) containing the center frequencies of the bands in Hz.
        sr (int):
            The sample rate in Hz.
        n_fft (int):
            The number of FFT points for the interpolation.
    Returns:
        Tensor:
            A tensor of shape (n_fft) containing the interpolated RT60 values for each FFT point.
    Raises:
        AssertionError:
            If the shapes of rt60_profile and center_freqs do not match.
    """
    assert (
        rt60_profile.shape == center_freqs.shape
    ), "RT60 profile and center frequencies must have the same shape."

    linear_freqs = torch.linspace(
        0,
        sr // 2,
        n_fft // 2 + 1,
        dtype=center_freqs.dtype,
        device=center_freqs.device,
    )
    linear_rt_profile = interpolate(
        center_freqs,
        rt60_profile,
        linear_freqs,
        method=interpolation_method,
    )
    return linear_rt_profile.to(dtype=rt60_profile.dtype)


def create_log_RT_profile(
    rt60_profile: Tensor,
    center_freqs: Tensor,
    start_freq: float,
    sr: int,
    n_fft: int,
    interpolation_method: str = "pchip",
) -> Tensor:
    """
    Create a logarithmic RT60 profile.

    Args:
        rt60_profile (Tensor):
            The RT60 profile tensor of shape (n_f_bands) in seconds.
        center_freqs (Tensor):
            The center frequencies tensor of shape (n_f_bands) containing the center frequencies of the bands in Hz.
        sr (int):
            The sample rate in Hz.
        start_freq (float):
            The starting frequency in Hz.
        n_fft (int):
            The number of FFT points for the interpolation.
    Returns:
        Tensor:
            A tensor of shape (n_fft) containing the interpolated RT60 values for each FFT point.
    Raises:
        AssertionError:
            If the shapes of rt60_profile and center_freqs do not match.
    """
    assert (
        rt60_profile.shape == center_freqs.shape
    ), "RT60 profile and center frequencies must have the same shape."

    log_freqs = torch.logspace(
        torch.log10(torch.tensor(start_freq)),
        torch.log10(torch.tensor(sr // 2)),
        n_fft // 2 + 1,
        dtype=center_freqs.dtype,
        device=center_freqs.device,
    )
    linear_rt_profile = interpolate(
        center_freqs,
        rt60_profile,
        log_freqs,
        method=interpolation_method,
    )
    return linear_rt_profile.to(dtype=rt60_profile.dtype)


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


def alpha_to_rt_profile(alpha: Tensor) -> Tensor:
    """
    Convert SWFT parameters to a linear RT60 profile.

    Args:
        alpha (Tensor):
            The tensor $alpha(f)$ of shape (n_fft) containing the SWFT parameters for each FFT point.
    Returns:
        Tensor:
            The linear RT60 profile tensor of shape (n_fft).
    """

    return 3 * torch.log(torch.tensor(10.0)) / (alpha)


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


def init_taylor_mul(
    Lh: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[int, Tensor, Tensor]:
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
            omega, the Fourier basis of size (nfft // 2 + 1) for real signals
    """
    nfft = int(1.1 * Lh)
    taylor_arange = torch.arange(Lh, dtype=dtype, device=device)
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


def naive_taylor_mul(
    epsilon: Tensor,
    p_fourier: Tensor,
    omega: Tensor,
    nfft: int,
):
    """
    (P epsilon)(z) = E(z / P(z)) = sum epsilon[n] P(z)^n z^(-n)
    """
    Lh = epsilon.shape[0]
    # Can be computed as a polynomial evaluation
    eval_points = p_fourier * omega
    coeffs = epsilon.flip(0)
    h_bar_fourier = torch_polyval(coeffs, eval_points)

    return cast(Tensor, torch.fft.irfft(h_bar_fourier, nfft))[:Lh].real


def torch_polyval(coeffs: Tensor, z: Tensor) -> Tensor:
    """
    Evaluate a polynomial using Horner's method.

    P(z) = coeffs[0] * z^(n-1) + coeffs[1] * z^(n-2) + ... + coeffs[-1]

    Args:
        coeffs: 1D tensor of polynomial coefficients, highest degree first.
                Complex or real dtype supported.
        z: Tensor of evaluation points (any shape), may be complex.

    Returns:
        Tensor of the same shape as z (broadcast with nothing else needed,
        since coeffs is a vector), with promoted dtype.
    """

    dtype = torch.promote_types(coeffs.dtype, z.dtype)
    coeffs = coeffs.to(dtype)
    z = z.to(dtype)

    result = torch.zeros_like(z)
    for c in coeffs:
        result = result * z + c
    return result
