from vpr_rir_refactor import run_all_xp, compute_all_metrics, create_xp_from_template
from pathlib import Path


def main():
    template_path = Path("experiments/template.yaml")
    exp_path = create_xp_from_template(template_path)
    run_all_xp(exp_path)
    compute_all_metrics(exp_path)


if __name__ == "__main__":
    main()
