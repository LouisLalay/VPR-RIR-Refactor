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


if __name__ == "__main__":
    main()
