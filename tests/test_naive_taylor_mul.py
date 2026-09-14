from vpr_rir_refactor.swft_operators import (
    naive_taylor_mul,
    rt_profile_to_alpha,
    alpha_to_p_fourier,
    create_linear_RT_profile,
    taylor_mul,
)
import torch


def test_naive_implementation():
    Lh = 64000
    nfft = 2**16
    sr = 8000
    omega = torch.exp(-2j * torch.pi * torch.arange(nfft // 2 + 1) / nfft)
    epsilon = torch.randn(Lh)

    rt_profile = torch.rand(10) + 0.1
    center_freqs = torch.linspace(0, sr / 2, len(rt_profile))
    linear_rt = create_linear_RT_profile(
        rt_profile,
        center_freqs,
        sr,
        nfft,
    )
    alpha = rt_profile_to_alpha(linear_rt)
    p_fourier = alpha_to_p_fourier(alpha, sr)
    taylor_arange = torch.arange(Lh)

    h_bar = naive_taylor_mul(
        epsilon,
        p_fourier,
        omega,
        nfft,
    )

    h_bar_taylor = taylor_mul(
        epsilon,
        taylor_arange,
        omega,
        p_fourier,
        nfft,
    )

    assert torch.allclose(h_bar, h_bar_taylor, atol=1e-4)
