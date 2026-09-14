from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_metrics(metrics_path: Path) -> None:
    """
    Plot an overview of all metrics grouped by method.

    Args:
        metrics_path (Path):
            Path to the experiment's metrics.csv file.
    """
    metrics = pd.read_csv(metrics_path)

    metric_columns = [
        column
        for column in metrics.columns
        if column
        not in {
            "run_id",
            "method",
            "snr",
            "noise_type",
            "run_hash",
        }
    ]

    noise_type = metrics["noise_type"].iloc[0]
    snr = metrics["snr"].iloc[0]

    plot_data = metrics.melt(
        id_vars=["method"],
        value_vars=metric_columns,
        var_name="metric",
        value_name="value",
    )

    sns.set_theme(style="whitegrid")

    g = sns.catplot(
        data=plot_data,
        x="method",
        y="value",
        col="metric",
        kind="box",
        col_wrap=3,
        sharey=False,
        height=4,
        aspect=1.2,
    )

    g.set_titles("{col_name}")
    g.set_axis_labels("Method", "Value")
    g.figure.suptitle(f"Metrics Overview (Noise: {noise_type}, SNR: {snr} dB)")

    plt.tight_layout()
    plt.show()
