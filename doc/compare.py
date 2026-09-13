# bench.py
import json
import statistics
from pathlib import Path
from time import perf_counter

import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from vpr_rir_refactor.swft_operators import (
    naive_taylor_mul,
    taylor_mul,
    rt_profile_to_alpha,
    alpha_to_p_fourier,
    create_linear_RT_profile,
)

DB = Path("doc/bench.json")


def build(Lh: int, sr: int, device: torch.device, seed=0):
    """All tensors on `device`, nothing hidden on CPU."""
    g = torch.Generator().manual_seed(seed)
    nfft = 2 ** (Lh - 1).bit_length()

    epsilon = torch.randn(Lh, generator=g).to(device)
    omega = torch.exp(
        -2j * torch.pi * torch.arange(nfft // 2 + 1, device=device) / nfft
    )
    rt_profile = torch.rand(10, generator=g) + 0.1
    center_freqs = torch.linspace(0, sr / 2, 10)
    alpha = rt_profile_to_alpha(
        create_linear_RT_profile(rt_profile, center_freqs, sr, nfft)
    )
    p_fourier = alpha_to_p_fourier(alpha, sr).to(device)
    arange = torch.arange(Lh, device=device)

    return epsilon, p_fourier, omega, arange, nfft


def time_one(method, Lh, sr, device, seed=0, warmup=5, min_total=0.5):
    """Median seconds per call. Syncs CUDA properly."""
    dev = torch.device(device)
    eps, pf, om, ar, nfft = build(Lh, sr, dev, seed)

    if method == "naive":
        fn = lambda: naive_taylor_mul(eps, pf, om, nfft)
    else:
        fn = lambda: taylor_mul(eps, ar, om, pf, nfft)

    sync = torch.cuda.synchronize if dev.type == "cuda" else lambda: None

    for _ in range(warmup):
        fn()
    sync()

    ts, total = [], 0.0
    while len(ts) < 200 and (len(ts) < 10 or total < min_total):
        sync()
        t0 = perf_counter()
        fn()
        sync()
        dt = perf_counter() - t0
        ts.append(dt)
        total += dt

    return statistics.median(ts)


def sweep(lengths, sr=16000, naive_max=20000, seed=0):
    """Run everything missing, cache to JSON. Safe to re-run."""
    db = json.loads(DB.read_text()) if DB.exists() else {}
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    todo = [
        (method, device, Lh)
        for method in ("naive", "taylor")
        for device in devices
        for Lh in lengths
        if not (method == "naive" and Lh > naive_max)
        and f"{method}|{device}|{Lh}|{sr}|{seed}" not in db
    ]
    print(f"{len(todo)} to run, {len(db)} cached")

    for m, d, L in tqdm(todo):
        try:
            db[f"{m}|{d}|{L}|{sr}|{seed}"] = time_one(m, L, sr, d, seed)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            tqdm.write(f"skip {m}/{d}/{L}: {e}")
            torch.cuda.empty_cache()
        DB.write_text(json.dumps(db, indent=1))

    return db


def plot(db=None, sr=16000, seed=0):
    db = db or json.loads(DB.read_text())
    fig, ax = plt.subplots(figsize=(7, 5))

    for m in ("naive", "taylor"):
        for d in ("cpu", "cuda"):
            pts = sorted(
                (int(k.split("|")[2]), v)
                for k, v in db.items()
                if k.startswith(f"{m}|{d}|") and k.endswith(f"|{sr}|{seed}")
            )
            if pts:
                ax.plot(*zip(*pts), marker="o", ms=3, label=f"{m}/{d}")

    ax.set(xscale="log", yscale="log", xlabel="$L_h$", ylabel="median time (s)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    lengths = torch.logspace(2, 20, 30, base=2, dtype=torch.int32).tolist()
    sweep(lengths)
    plot()
    plt.show()
