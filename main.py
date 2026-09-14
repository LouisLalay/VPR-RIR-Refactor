from vpr_rir_refactor import (
    create_xp_from_template,
    run_all_xp,
    compute_all_metrics,
    plot_metrics,
)
from pathlib import Path


def main():
    template_path = Path("experiments/template.yaml")
    exp_path = create_xp_from_template(template_path)
    run_all_xp(exp_path)
    compute_all_metrics(exp_path)
    plot_metrics(exp_path / "metrics.csv")


def run_multiple_xp(template_dir: Path):
    # Takes multiple templates from a directory and runs all experiments
    for template_file in template_dir.glob("*.yaml"):
        exp_path = create_xp_from_template(template_file)
        run_all_xp(exp_path)
        compute_all_metrics(exp_path)


if __name__ == "__main__":
    run_multiple_xp(Path("experiments/2026-09-14__First-serious-run"))
