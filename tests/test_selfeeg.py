from selfeeg.utils.utils import torch_pchip
from scipy.interpolate import pchip_interpolate
import numpy as np
import torch


def test_pchip():
    frequency_bands = np.array([0, 1, 2, 4, 8, 16, 32], dtype=np.float32)
    rt_profile = np.random.randn(len(frequency_bands)).astype(np.float32)
    linear_freqs = np.linspace(0, frequency_bands[-1], 64, dtype=np.float32)

    scipy_interpolated = pchip_interpolate(frequency_bands, rt_profile, linear_freqs)
    scipy_interpolated = torch.from_numpy(scipy_interpolated).float()
    torch_interpolated = torch_pchip(
        torch.from_numpy(frequency_bands),
        torch.from_numpy(rt_profile),
        torch.from_numpy(linear_freqs),
        save_memory=False,
    )

    assert torch.allclose(torch_interpolated, scipy_interpolated)
