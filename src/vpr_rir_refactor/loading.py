from __future__ import annotations
from numpy import float64, float32
from pathlib import Path
from torch import Tensor
from typing import Literal, NamedTuple
import hashlib
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

_DTYPE_TO_NP = {torch.float64: float64, torch.float32: float32}


def _check_dtype(dtype: torch.dtype) -> None:
    """
    Validate that dtype is a supported floating point type.

    Args:
        dtype (torch.dtype):
            Candidate dtype.
    Returns:
        None
    Raises:
        ValueError:
            If dtype is not torch.float32 or torch.float64.
    """
    if dtype not in _DTYPE_TO_NP:
        raise ValueError(f"dtype must be float32 or float64, got {dtype}.")


def _load_mono(
    path: Path,
    target_sr: int,
    dtype: torch.dtype,
    frames: int = -1,
    channel: int | Literal["mean"] = 0,
) -> Tensor:
    """
    Load a single channel from an audio file, resampling if needed.

    Args:
        path (Path):
            Path to the audio file.
        target_sr (int):
            Desired sample rate. The file is resampled if its native
            sample rate differs.
        dtype (torch.dtype):
            Output dtype, torch.float32 or torch.float64.
        frames (int, optional):
            Number of frames to read from the start of the file.
            Defaults to -1 (read all).
        channel (int | "mean", optional):
            Channel index to keep, or "mean" to average all channels
            to mono. Defaults to 0.
    Returns:
        Tensor:
            1-D tensor of shape (frames,).
    Raises:
        ValueError:
            If `channel` is an out-of-range index.
    """
    info = sf.info(str(path))
    if isinstance(channel, int) and not (-info.channels <= channel < info.channels):
        raise ValueError(
            f"{Path(path).name}: channel {channel} out of range for "
            f"{info.channels}-channel file."
        )

    read_frames = frames if info.samplerate == target_sr else -1
    x, native_sr = sf.read(
        str(path),
        frames=read_frames,
        dtype=_DTYPE_TO_NP[dtype],
        always_2d=True,
    )  # (frames, channels)

    x = x.mean(axis=1) if channel == "mean" else x[:, channel]

    if not x.flags["C_CONTIGUOUS"]:
        x = np.ascontiguousarray(x)

    x = torch.from_numpy(x)

    if native_sr != target_sr:
        x = AF.resample(x, native_sr, target_sr)

    if frames > 0 and read_frames == -1:
        x = x[:frames]

    return x


def _trim_trailing_silence(x: Tensor, thresh: float = 1e-6) -> Tensor:
    """
    Remove trailing near-zero samples.

    Args:
        x (Tensor):
            1-D signal.
        thresh (float, optional):
            Amplitude threshold below which a sample is considered
            silent. Defaults to 1e-6.
    Returns:
        Tensor:
            `x` truncated after its last sample with |x| > thresh.
            Returns `x` unchanged if no such sample exists or the
            last sample already exceeds the threshold.
    """
    above = (x.abs() > thresh).nonzero()
    if above.numel() == 0:
        return x
    last = int(above[-1])
    return x[: last + 1]


def _tile_to(x: Tensor, n: int) -> Tensor:
    """
    Repeat a signal cyclically from its start until it has n samples.

    Args:
        x (Tensor):
            1-D signal to tile.
        n (int):
            Target length.
    Returns:
        Tensor:
            Tensor of shape (n,).
    Raises:
        ValueError:
            If x is empty.
    """
    if x.shape[0] == 0:
        raise ValueError("Cannot tile an empty signal.")
    reps = -(-n // x.shape[0])
    return x.repeat(reps)[:n]


class Signals(NamedTuple):
    """
    Validated inputs for `reverberant_mix`.

    Invariants:
        - source, rir, noise are 1-D and share `dtype`.
        - source.shape[0] + rir.shape[0] - 1 == n_samples.
        - noise.shape[0] == n_samples.

    Attributes:
        source (Tensor):
            Dry source signal, shape (n_samples - L_rir + 1,).
        rir (Tensor):
            Room impulse response, shape (L_rir,).
        noise (Tensor):
            Noise signal, shape (n_samples,).
        sr (int):
            Sample rate shared by all signals.
        n_samples (int):
            Length of the full reverberant signal (source * rir),
            i.e. `source.shape[0] + rir.shape[0] - 1`.
        key (str):
            Hash of all parameters used to produce this instance.
    """

    source: Tensor
    rir: Tensor
    noise: Tensor
    sr: int
    n_samples: int
    key: str


def prepare_signals(
    source_path: Path,
    rir_path: Path,
    sr: int,
    n_samples: int,
    seed: int,
    noise_path: Path,
    noise_type: Literal["gaussian", "real"] = "gaussian",
    dtype: torch.dtype = torch.float64,
    rir_norm: Literal["peak", "energy", "none"] = "peak",
    rir_silence_thresh: float = 1e-6,
) -> Signals:
    """
    Load and condition source / RIR / noise so that `reverberant_mix`
    receives length- and dtype-consistent inputs.

    `n_samples` is the desired length of the full reverberant signal
    (source convolved with rir). The source length is deduced from it
    once the RIR length is known: `L_source = n_samples - L_rir + 1`.
    The RIR is read in full and only truncated to remove trailing
    near-zero samples; it is never cropped otherwise. Real noise is
    read in full and tiled cyclically from its start if it is shorter
    than `n_samples`; Gaussian noise is sampled directly at the
    required length.

    Args:
        source_path (Path):
            Path to the dry source file.
        rir_path (Path):
            Path to the RIR file.
        sr (int):
            Target sample rate. Files at a different native rate are
            resampled.
        n_samples (int):
            Desired length, in samples, of the full reverberant
            signal `source * rir`.
        noise_type ("gaussian" | "real", optional):
            "gaussian" samples i.i.d. standard normal noise, "real"
            loads noise from `noise_path`. Defaults to "gaussian".
        noise_path (Path):
            Path to the noise file. Unused if `noise_type` is not "real".
        dtype (torch.dtype, optional):
            torch.float32 or torch.float64. Defaults to torch.float64.
        rir_norm ("peak" | "energy" | "none", optional):
            "peak" divides the RIR by max|h|, "energy" divides by
            ||h||_2, "none" leaves it untouched. Defaults to "peak".
        rir_silence_thresh (float, optional):
            Threshold for trailing-silence trimming of the RIR.
            Defaults to 1e-6.
        seed (int):
            Seed for Gaussian noise generation. Unused for real noise.
    Returns:
        Signals:
            Loaded and conditioned signals, ready for
            `reverberant_mix`.
    Raises:
        ValueError:
            If dtype is unsupported, `noise_type` is "real" without
            `noise_path`, or `n_samples` is too small to fit the
            (trimmed) RIR with a nonempty source.
    """
    _check_dtype(dtype)
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}.")
    if noise_type == "real" and noise_path is None:
        raise ValueError("noise_path is required when noise_type='real'.")

    # --- RIR ---------------------------------------------------------
    rir = _load_mono(rir_path, sr, dtype)
    rir = _trim_trailing_silence(rir, rir_silence_thresh)
    if rir_norm == "peak":
        rir = rir / rir.abs().max()
    elif rir_norm == "energy":
        rir = rir / rir.pow(2).sum().sqrt()

    L_source = n_samples - rir.shape[0] + 1
    if L_source <= 0:
        raise ValueError(
            f"n_samples={n_samples} is too small for RIR of length "
            f"{rir.shape[0]}; need n_samples >= {rir.shape[0]}."
        )

    # --- source ------------------------------------------------------
    source = _load_mono(source_path, sr, dtype, frames=L_source)
    if source.shape[0] < L_source:
        source = torch.nn.functional.pad(source, (0, L_source - source.shape[0]))
    else:
        source = source[:L_source]

    # --- noise ---------------------------------------------------------
    if noise_type == "gaussian":
        generator = torch.Generator().manual_seed(seed)
        noise = torch.randn(n_samples, dtype=dtype, generator=generator)
    else:
        noise = _load_mono(noise_path, sr, dtype)
        noise = _tile_to(noise, n_samples)

    # --- key -------------------------------------------------------------
    spec = "|".join(
        str(v)
        for v in (
            source_path.name,
            rir_path.name,
            sr,
            n_samples,
            noise_type,
            noise_path.name if noise_path is not None else "",
            str(dtype),
            rir_norm,
            rir_silence_thresh,
            seed,
        )
    )
    key = hashlib.sha256(spec.encode()).hexdigest()

    assert source.shape[0] + rir.shape[0] - 1 == n_samples
    assert noise.shape[0] == n_samples
    assert source.dtype == rir.dtype == noise.dtype == dtype

    return Signals(source, rir, noise, sr, n_samples, key)
