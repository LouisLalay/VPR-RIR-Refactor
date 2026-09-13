from __future__ import annotations
from .data import discover_BRAS, discover_librispeech, discover_Whamr
from .loading import prepare_signals, Signals
from pathlib import Path
from typing import Literal, NamedTuple
import csv
import hashlib
import random
import subprocess
import torch
import yaml

_DTYPE_FROM_STR = {"float32": torch.float32, "float64": torch.float64}


class ExperimentConfig(NamedTuple):
    """
    Parameters shared by every run of one experiment.

    Attributes:
        name (str):
            Experiment identifier, used for logging/output paths.
        sr (int):
            Target sample rate.
        n_samples (int):
            Length, in samples, of the full reverberant signal.
        noise_type ("gaussian" | "real"):
            Noise generation mode, shared across all runs.
        snr_dB (float):
            Desired signal-to-noise ratio in dB.
        seed (int):
            Seed for Gaussian noise generation, shared across runs.
        dtype (torch.dtype):
            torch.float32 or torch.float64.
        rir_norm ("peak" | "energy" | "none"):
            RIR normalization mode.
        rir_silence_thresh (float):
            Threshold for trailing-silence trimming of the RIR.
    """

    name: str
    sr: int
    n_samples: int
    noise_type: Literal["gaussian", "real"]
    snr_dB: float
    seed: int
    dtype: torch.dtype
    rir_norm: Literal["peak", "energy", "none"]
    rir_silence_thresh: float


class RunFiles(NamedTuple):
    """
    File selection for a single run within an experiment.

    Attributes:
        run_id (int):
            Index of this run within the experiment.
        source_path (Path):
            Path to the dry source file.
        rir_path (Path):
            Path to the RIR file.
        noise_path (Path):
            Path to the noise file. Required (non-empty) iff the
            experiment's `noise_type` is "real", ignored otherwise.
    """

    run_id: int
    source_path: Path
    rir_path: Path
    noise_path: Path


def _load_experiment_config(config_path: Path) -> ExperimentConfig:
    """
    Parse the shared experiment parameters from a YAML file.

    Args:
        config_path (Path):
            Path to the YAML config file.
    Returns:
        ExperimentConfig:
            Validated, typed shared configuration.
    Raises:
        ValueError:
            If `noise_type` or `dtype` is unsupported.
    """
    with config_path.open("r") as f:
        raw = yaml.safe_load(f)

    if raw["noise_type"] not in ("gaussian", "real"):
        raise ValueError(
            f"noise_type must be 'gaussian' or 'real', got {raw['noise_type']!r}."
        )
    if raw["dtype"] not in _DTYPE_FROM_STR:
        raise ValueError(f"unsupported dtype {raw['dtype']!r}.")

    return ExperimentConfig(
        name=raw["name"],
        sr=int(raw["sr"]),
        n_samples=int(raw["n_samples"]),
        noise_type=raw["noise_type"],
        snr_dB=float(raw["snr_dB"]),
        seed=int(raw["seed"]),
        dtype=_DTYPE_FROM_STR[raw["dtype"]],
        rir_norm=raw["rir_norm"],
        rir_silence_thresh=float(raw["rir_silence_thresh"]),
    )


def _load_run_files(
    runs_path: Path,
    noise_type: Literal["gaussian", "real"],
) -> list[RunFiles]:
    """
    Parse the per-run file matrix from a CSV file.

    Args:
        runs_path (Path):
            Path to the CSV file with columns run_id, source_path,
            rir_path, noise_path.
        noise_type ("gaussian" | "real"):
            Experiment's noise type, used to validate `noise_path`.
    Returns:
        list[RunFiles]:
            One entry per row, in file order.
    Raises:
        ValueError:
            If `noise_type` is "real" and a row has an empty
            `noise_path`.
    """
    runs: list[RunFiles] = []
    with runs_path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            noise_path_str = row["noise_path"].strip()
            if noise_type == "real" and not noise_path_str:
                raise ValueError(
                    f"run {row['run_id']}: noise_path required for real noise."
                )

            runs.append(
                RunFiles(
                    run_id=int(row["run_id"]),
                    source_path=Path(row["source_path"]),
                    rir_path=Path(row["rir_path"]),
                    noise_path=Path(noise_path_str) if noise_path_str else Path(""),
                )
            )
    return runs


def _prepare_experiment(config_path: Path, files_path: Path) -> dict[int, Signals]:
    """
    Run `prepare_signals` for every run of one experiment.

    Args:
        config_path (Path):
            Path to the experiment's shared-parameter YAML file.
        files_path (Path):
            Path to the experiment's run-matrix CSV file.
    Returns:
        dict[int, Signals]:
            Mapping from `run_id` to its prepared `Signals`.
    """
    config = _load_experiment_config(config_path)
    runs = _load_run_files(files_path, config.noise_type)

    all_signals: dict[int, Signals] = {}
    for run in runs:
        all_signals[run.run_id] = prepare_signals(
            source_path=run.source_path,
            rir_path=run.rir_path,
            sr=config.sr,
            n_samples=config.n_samples,
            seed=config.seed,
            noise_path=run.noise_path,
            noise_type=config.noise_type,
            dtype=config.dtype,
            rir_norm=config.rir_norm,
            rir_silence_thresh=config.rir_silence_thresh,
        )
    return all_signals


def _plan_experiment(
    available_sources: list[Path],
    available_rirs: list[Path],
    available_noises: list[Path],
    n_runs: int,
    xp_path: Path,
    seed: int = 0,
) -> None:
    """
    Build a run matrix and write it to `runs_path` as CSV.

    RIR coverage is prioritized: RIRs are cycled through in shuffled
    order so that usage counts differ by at most one across all RIRs
    (every RIR is used at least once if `n_runs >= len(available_rirs)`).
    Sources and noises are less critical and are simply cycled through
    independently in their own shuffled order, so they vary from run to
    run without any uniformity guarantee.

    Args:
        available_sources (list[Path]):
            Pool of candidate source files.
        available_rirs (list[Path]):
            Pool of candidate RIR files. Coverage of this pool is
            prioritized over the other two.
        available_noises (list[Path]):
            Pool of candidate noise files. Ignored downstream if the
            experiment's noise_type is "gaussian" (the column is still
            written, just unused).
        n_runs (int):
            Number of rows (runs) to generate.
        runs_path (Path):
            Output CSV path, with columns run_id, source_path,
            rir_path, noise_path.
        seed (int, optional):
            Seed controlling the shuffles, for reproducibility.
            Defaults to 0.
    Returns:
        None
    Raises:
        ValueError:
            If any of the three pools is empty.
    """
    if not available_sources or not available_rirs or not available_noises:
        raise ValueError(
            "available_sources, available_rirs and available_noises must be non-empty."
        )

    rng = random.Random(seed)

    def _cycle(pool: list[Path], n: int) -> list[Path]:
        """Repeat a shuffled copy of `pool` until length `n` is reached."""
        shuffled = pool.copy()
        rng.shuffle(shuffled)
        out: list[Path] = []
        while len(out) < n:
            if len(out) % len(pool) == 0:
                rng.shuffle(shuffled)
            out.extend(shuffled)
        return out[:n]

    rir_seq = _cycle(available_rirs, n_runs)
    source_seq = _cycle(available_sources, n_runs)
    noise_seq = _cycle(available_noises, n_runs)

    xp_path.parent.mkdir(parents=True, exist_ok=True)
    with xp_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "source_path", "rir_path", "noise_path"])
        for run_id, (source, rir, noise) in enumerate(
            zip(source_seq, rir_seq, noise_seq)
        ):
            writer.writerow([run_id, source, rir, noise])


def _get_git_commit_hash() -> str:
    """
    Get the current git commit hash.

    Returns:
        str:
            Commit hash as a string, or "unknown" if not in a git repo.
    Raises:
        ValueError:
            If the git command fails for any reason.
    """

    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
        return commit_hash
    except subprocess.CalledProcessError as e:
        raise ValueError("Not a git repository") from e


def create_xp_from_template(template_path: Path) -> Path:
    """
    Create an experiment directory populated with config.yaml and
    files.csv, from a template file.

    Args:
        template_path (Path):
            Path to the YAML template (seed, dtype, sr, n_samples,
            noise_type, snr_dB, rir_norm, rir_silence_thresh, n_runs,
            datasets paths, method spec).
    """
    with template_path.open("r") as f:
        template = yaml.safe_load(f)

    try:
        git_commit_hash = _get_git_commit_hash()
    except ValueError:
        from warnings import warn

        warn(
            "Not git repository found. "
            "We recommend using git to track your experiments for reproducibility."
        )
        git_commit_hash = ""

    # Create a unique experiment name bound to the current git commit hash and template
    template_hash = hashlib.sha256(
        yaml.safe_dump(template).encode("utf-8")
    ).hexdigest()[:8]
    exp_dir = template_path.parent / f"{template_hash}__{git_commit_hash}"
    if exp_dir.exists():
        return exp_dir

    exp_dir.mkdir(parents=True, exist_ok=True)

    # Load the available datasets
    sources = discover_librispeech(Path(template["datasets"]["sources"]))
    rirs = discover_BRAS(Path(template["datasets"]["rirs"]))
    noises = discover_Whamr(Path(template["datasets"]["noises"]))

    # Create a run matrix and write it to files.csv
    _plan_experiment(
        available_sources=sources,
        available_rirs=rirs,
        available_noises=noises,
        n_runs=template["n_runs"],
        xp_path=exp_dir / "files.csv",
        seed=template["seed"],
    )

    config = {
        "name": exp_dir.name,
        "sr": template["sr"],
        "n_samples": template["n_samples"],
        "noise_type": template["noise_type"],
        "snr_dB": template["snr_dB"],
        "seed": template["seed"],
        "dtype": template["dtype"],
        "rir_norm": template["rir_norm"],
        "rir_silence_thresh": template["rir_silence_thresh"],
    }
    with (exp_dir / "config.yaml").open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    # One entry per method, kept separate from the shared config
    with (exp_dir / "methods.yaml").open("w") as f:
        yaml.safe_dump(template["method"], f, sort_keys=False)

    return exp_dir
