from vpr_rir_refactor.torch_pchip import cubic_interp1d as torch_pchip
import pytest
import torch


@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
def test_device(device: torch.device):
    x = torch.logspace(0, 1, 10, device=device)
    y = torch.randn_like(x)
    xs = torch.linspace(x[0], x[-1], 100, device=device)
    ys = torch_pchip(x, y, xs)

    assert ys.device.type == device.type


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype(dtype: torch.dtype):
    x = torch.logspace(0, 1, 10, dtype=dtype)
    y = torch.randn_like(x)
    xs = torch.linspace(x[0], x[-1], 100, dtype=dtype)
    ys = torch_pchip(x, y, xs)

    assert ys.dtype == dtype
