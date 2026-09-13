from __future__ import annotations
from .experiments import _load_experiment_config
from .loading import _load_mono
from pathlib import Path
from taylor_swft.evaluation.metrics import distance, METRICS_UNITS
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import csv
import pandas as pd

METADATA_COLUMNS = [
    "run_id",  # Sort key
    "method",  # Method name
    "snr",  # Exp dependent
    "noise_type",  # Exp dependent
    "run_hash",  # Allows to match the filenames in the results.csv file
]

METRICS_COLUMNS = METADATA_COLUMNS + list(METRICS_UNITS.keys())


def log_metrics(
    logger: SummaryWriter,
    rir_ref: Tensor,
    rir_test: Tensor,
    sr: int,
    current_step: int,
):
    d = distance(rir_ref, rir_test, sr)
    for metric in d:
        if METRICS_UNITS[metric] != "":
            tag = f"{metric}_{METRICS_UNITS[metric]}"
        else:
            tag = metric
        logger.add_scalar(
            tag=f"METRIC/{tag}",
            scalar_value=d[metric],
            global_step=current_step,
        )


def _compute_metrics(
    rir_ref: Tensor,
    rir_test: Tensor,
    sr: int,
) -> dict[str, float]:
    """
    Compute all metrics between a reference and estimated RIR.

    Args:
        rir_ref (Tensor):
            Reference RIR.
        rir_test (Tensor):
            Estimated RIR.
        sr (int):
            Sampling rate.

    Returns:
        dict[str, float]:
            Computed metrics.
    """
    values = distance(rir_ref, rir_test, sr)

    return {
        name: float(value.item() if isinstance(value, Tensor) else value)
        for name, value in values.items()
    }


def _init_metrics(metrics_path: Path) -> None:
    """
    Create the metrics CSV with its header if it does not exist yet.
    """
    if metrics_path.is_file():
        return
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="") as f:
        csv.writer(f).writerow(METRICS_COLUMNS)


def _append_metrics(metrics_path: Path, row: dict) -> None:
    """
    Append one run to the metrics CSV.

    Args:
        metrics_path (Path):
            Path to the metrics CSV.
        row (dict):
            Mapping with the keys of METRICS_COLUMNS. Missing keys are
            written as empty strings.
    """
    _init_metrics(metrics_path)
    with metrics_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_COLUMNS, restval="")
        writer.writerow(row)


def compute_all_metrics(exp_dir: Path):
    results_file = exp_dir / "results.csv"
    results_df = pd.read_csv(results_file, index_col=0)

    exp_config = _load_experiment_config(exp_dir / "config.yaml")

    for k, row in tqdm(
        results_df.iterrows(),
        total=results_df.shape[0],
        desc="Computing metrics",
    ):
        metadata = {
            "run_id": row.name,
            "run_hash": row["run_hash"],
            "method": row["method"],
            "snr": exp_config.snr_dB,
            "noise_type": exp_config.noise_type,
        }
        rir_ref_path = Path(row["rir"])
        estimated_rir_path = Path(row["rir_estimate"])

        rir_ref = _load_mono(
            path=rir_ref_path,
            target_sr=exp_config.sr,
            dtype=exp_config.dtype,
        )
        estimated_rir = _load_mono(
            path=estimated_rir_path,
            target_sr=exp_config.sr,
            dtype=exp_config.dtype,
        )

        metrics = _compute_metrics(
            rir_ref=rir_ref,
            rir_test=estimated_rir,
            sr=exp_config.sr,
        )
        metrics.update(metadata)
        _append_metrics(exp_dir / "metrics.csv", metrics)
