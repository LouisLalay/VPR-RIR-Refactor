# Verbose copy of https://github.com/slanthaler/torch-pchip_interp/blob/main/pchip_interp.py
# which does not provide a python package

# It comes with the following license:
# MIT License

# Copyright (c) 2024 slanthaler

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from torch import Tensor


def h_poly(values: Tensor) -> Tensor:
    tt = values[None, :] ** torch.arange(4, device=values.device)[:, None]
    A = torch.tensor(
        [[1, 0, -3, 2], [0, 1, -2, 1], [0, 0, 3, -2], [0, 0, -1, 1]],
        dtype=values.dtype,
        device=values.device,
    )
    return A @ tt


def cubic_interp1d(
    input_positions: Tensor,
    input_values: Tensor,
    target_positions: Tensor,
) -> Tensor:
    """
    Cubic Hermite spline interpolation in 1d.

    Input: input_positions - shape (N,)
           input_values - shape (..., N)
           target_positions - shape (Ns,)
    Output: target_values - shape (..., Ns)
    """
    assert (
        input_positions.max() >= target_positions.max()
        and input_positions.min() <= target_positions.min()
    ), "target_positions is out of bounds"
    m = (input_values[..., 1:] - input_values[..., :-1]) / (
        input_positions[1:] - input_positions[:-1]
    )
    m = torch.cat([m[..., [0]], (m[..., 1:] + m[..., :-1]) / 2, m[..., [-1]]], dim=-1)
    idxs = torch.searchsorted(input_positions[1:], target_positions)
    dx = input_positions[idxs + 1] - input_positions[idxs]
    hh = h_poly((target_positions - input_positions[idxs]) / dx)
    ys = (
        hh[[0]] * input_values[..., idxs]
        + hh[[1]] * m[..., idxs] * dx
        + hh[[2]] * input_values[..., idxs + 1]
        + hh[[3]] * m[..., idxs + 1] * dx
    )
    return ys.squeeze()
