from .baselines import Baseline1, validate_baseline1_kwargs
from .experiments import ExperimentConfig, _load_experiment_config
from .loading import prepare_signals, Signals
from .mixing import reverberant_mix
from .vpr_rir import validate_vpr_kwargs, VPR_RIR2

from pathlib import Path
from soundfile import write
from typing import NamedTuple, Callable
import csv
import hashlib
import json
import pandas as pd
import yaml

RESULTS_COLUMNS = [
    "run_id",
    "run_hash",
    "method",
    "source",
    "rir",
    "noise",
    "rir_estimate",
    "parameters",
]


class MethodSpec(NamedTuple):
    """
    One entry of the method registry.

    Attributes:
        cls (type):
            Class implementing the method. Constructed with
            source, reference_rir, reverberant_signal and the
            validated kwargs, and exposing fit() and get_estimates().
        validate (Callable[[dict], dict]):
            Converts raw YAML kwargs to the types the class expects.
    """

    cls: type
    validate: Callable[[dict], dict]


KNOWN_METHODS = {
    "vpr_rir": MethodSpec(cls=VPR_RIR2, validate=validate_vpr_kwargs),
    "baseline_1": MethodSpec(cls=Baseline1, validate=validate_baseline1_kwargs),
}


def _parse_method_file(method_file: Path, config_file: Path) -> dict:
    """
    Parse the method file and config file, validate the methods and their parameters.

    Args:
        method_file (Path):
            Path to the method file.
        config_file (Path):
            Path to the config file.

    Returns:
        dict: A dictionary containing the validated methods and parameters.
    """
    with method_file.open("r") as f:
        method_dict: dict = yaml.safe_load(f)

    with config_file.open("r") as f:
        config_dict: dict = yaml.safe_load(f)

    valid_methods = {}

    for name, spec in method_dict.items():
        if name not in KNOWN_METHODS:
            raise ValueError(f"Unknown method '{name}'.")
        method_kwargs = {**spec, **config_dict}
        valid_methods[name] = KNOWN_METHODS[name].validate(method_kwargs)

    return valid_methods


def _parse_files(files_path: Path) -> pd.DataFrame:
    """
    Parse the files given for each runs

    Args:
        files_path (Path):
            Path to the directory containing the method and config files.
    Returns:
        pd.DataFrame: A DataFrame containing all the files required for the experiment.
    """
    return pd.read_csv(files_path, index_col=0)


def _kwargs_fingerprint(method_name: str, kwargs: dict) -> str:
    """
    Stable string representation of a method and its kwargs.

    Args:
        method_name (str):
            Name of the method, as in KNOWN_METHODS.
        kwargs (dict):
            Validated kwargs for that method.
    Returns:
        str:
            Deterministic string, stable across processes and dict
            insertion order.
    """
    items = {k: str(v) for k, v in sorted(kwargs.items())}
    return method_name + "|" + json.dumps(items, sort_keys=True)


def _run_hash(signals_key: str, method_name: str, kwargs: dict) -> str:
    """
    Hash uniquely identifying one (inputs, method, kwargs) triplet.

    Args:
        signals_key (str):
            `Signals.key`, already covering all input signals and
            preprocessing parameters.
        method_name (str):
            Name of the method.
        kwargs (dict):
            Validated kwargs for that method.
    Returns:
        str:
            Hex digest identifying this run.
    """
    spec = signals_key + "|" + _kwargs_fingerprint(method_name, kwargs)
    return hashlib.sha256(spec.encode()).hexdigest()


def _run_dir(exp_path: Path, run_hash: str) -> Path:
    """
    Directory holding the outputs of one run.
    """
    return exp_path / run_hash


def _rir_estimate_path(exp_path: Path, run_hash: str, rir_path: Path) -> Path:
    """
    Path of the estimated RIR, named after the input RIR.
    """
    return _run_dir(exp_path, run_hash) / f"{rir_path.stem}.wav"


def _is_already_run(exp_path: Path, run_hash: str, rir_path: Path) -> bool:
    """
    Whether this exact run already produced an estimate on disk.
    """
    return _rir_estimate_path(exp_path, run_hash, rir_path).is_file()


def _init_results(results_path: Path) -> None:
    """
    Create the results CSV with its header if it does not exist yet.
    """
    if results_path.is_file():
        return
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", newline="") as f:
        csv.writer(f).writerow(RESULTS_COLUMNS)


def _append_result(results_path: Path, row: dict) -> None:
    """
    Append one run to the results CSV.

    Args:
        results_path (Path):
            Path to the results CSV.
        row (dict):
            Mapping with the keys of RESULTS_COLUMNS. Missing keys are
            written as empty strings.
    """
    _init_results(results_path)
    with results_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS, restval="")
        writer.writerow(row)


def _call_method(
    method_name: str,
    valid_methods: dict,
    signals: Signals,
    config: ExperimentConfig,
    run_dir: Path,
    estimated_rir_path: Path,
) -> None:
    """
    Build the noisy mixture, run one method on it, and save its outputs.

    Args:
        method_name (str):
            Name of the method, as in KNOWN_METHODS.
        valid_methods (dict):
            Mapping from method name to its validated kwargs.
        signals (Signals):
            Prepared source, RIR and noise for this run.
        config (ExperimentConfig):
            Shared experiment parameters.
        run_dir (Path):
            Directory holding this run's outputs.
        estimated_rir_path (Path):
            Where to write the estimated RIR.
    """
    kwargs = valid_methods[method_name]
    method_cls = KNOWN_METHODS[method_name].cls

    mix = reverberant_mix(
        source=signals.source,
        ref_rir=signals.rir,
        noise=signals.noise,
        snr_dB=config.snr_dB,
    )

    model = method_cls(
        source=signals.source,
        reference_rir=signals.rir,
        reverberant_signal=mix.noisy,
        **kwargs,
    )
    model.fit_experiment()
    rir_estimate, parameters = model.get_estimates()

    write(estimated_rir_path, rir_estimate, config.sr)

    with (run_dir / "parameters.json").open("w") as f:
        json.dump(parameters, f, indent=2)


def _run_one_xp(
    files_row: pd.Series,
    valid_methods: dict,
    config: ExperimentConfig,
    exp_path: Path,
) -> None:
    """
    Run every method on the files of one run, skipping cached runs.

    Args:
        files_row (pd.Series):
            One row of files.csv, with source_path, rir_path,
            noise_path, and run_id as its name.
        valid_methods (dict):
            Mapping from method name to its validated kwargs.
        config (ExperimentConfig):
            Shared experiment parameters.
        exp_path (Path):
            Experiment directory, holding per-run subdirectories and
            results.csv.
    """
    source_path = Path(files_row["source_path"])
    rir_path = Path(files_row["rir_path"])
    noise_path_str = str(files_row["noise_path"]).strip()
    noise_path = Path(noise_path_str)

    signals = prepare_signals(
        source_path=source_path,
        rir_path=rir_path,
        sr=config.sr,
        n_samples=config.n_samples,
        seed=config.seed,
        noise_path=noise_path,
        noise_type=config.noise_type,
        dtype=config.dtype,
        rir_norm=config.rir_norm,
        rir_silence_thresh=config.rir_silence_thresh,
    )

    results_path = exp_path / "results.csv"

    for method_name, kwargs in valid_methods.items():
        run_hash = _run_hash(signals.key, method_name, kwargs)
        if _is_already_run(exp_path, run_hash, rir_path):
            continue
        run_dir = _run_dir(exp_path, run_hash)
        run_dir.mkdir(parents=True, exist_ok=True)

        estimated_rir_path = _rir_estimate_path(exp_path, run_hash, rir_path)
        _call_method(
            method_name=method_name,
            valid_methods=valid_methods,
            signals=signals,
            config=config,
            run_dir=run_dir,
            estimated_rir_path=estimated_rir_path,
        )
        parameters_path = run_dir / "parameters.json"
        _append_result(
            results_path,
            {
                "run_hash": run_hash,
                "run_id": files_row.name,
                "method": method_name,
                "source": str(source_path),
                "rir": str(rir_path),
                "noise": noise_path_str,
                "rir_estimate": str(estimated_rir_path),
                "parameters": str(parameters_path) if parameters_path.is_file() else "",
            },
        )


def run_all_xp(exp_path: Path) -> None:
    """
    Run every method on every row of an experiment's files.csv.

    Reads config.yaml, methods.yaml and files.csv from `exp_path`, and
    writes per-run outputs and results.csv back into it. Runs whose
    estimate already exists on disk are skipped.

    Args:
        exp_path (Path):
            Experiment directory, as produced by
            `create_xp_from_template`.
    """
    config = _load_experiment_config(exp_path / "config.yaml")
    valid_methods = _parse_method_file(
        exp_path / "methods.yaml", exp_path / "config.yaml"
    )
    files = _parse_files(exp_path / "files.csv")

    for _, files_row in files.iterrows():
        _run_one_xp(
            files_row=files_row,
            valid_methods=valid_methods,
            config=config,
            exp_path=exp_path,
        )
