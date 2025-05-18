import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np


def load_and_merge_data(base_path="results/class_imbalance"):
    base_df = pd.read_csv(os.path.join(base_path, "evaluation_data.csv"))
    modified_df = pd.read_csv(os.path.join(base_path, "newevaluation_data.csv"))
    merged_df = pd.merge(
        base_df, modified_df, on="name", suffixes=("_base", "_modified")
    )
    return merged_df


def plot_metric_comparison(merged_data, metric, show=True, save_path=None):
    # Calculate the difference (Modified - Base)
    delta = merged_data[f"{metric}_modified"] - merged_data[f"{metric}_base"]
    names = merged_data["name"].str.replace("dataset_", "")

    # Colors based on positive/negative delta (green for positive, red for negative)
    colors = ["#4CAF50" if val >= 0 else "#F44336" for val in delta]

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(names, delta, color=colors)

    # Add a horizontal line at zero
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    # Labels and title
    ax.set_title(f"Difference in {metric.capitalize()} (Modified - Base)", fontsize=14)
    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel(f"Δ {metric.capitalize()}", fontsize=12)

    # Rotate x-axis labels and adjust layout
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    ax.tick_params(axis="y", labelsize=10)

    # Dynamically adjust y-axis limits based on the max delta magnitude
    max_delta = max(delta.max(), abs(delta.min()))
    buffer = max(0.1, max_delta * 0.1)  # Ensure some buffer space
    ax.set_ylim(-max_delta - buffer, max_delta + buffer)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_improvement_summary(merged_data, show=True, save_path=None):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    summary_data = []

    for metric in metrics:
        delta = merged_data[f"{metric}_modified"] - merged_data[f"{metric}_base"]
        summary_data.append(
            {
                "metric": metric,
                "mean_improvement": delta.mean(),
                "improved": sum(delta > 0),
                "worsened": sum(delta < 0),
                "neutral": sum(delta == 0),
            }
        )

    df = pd.DataFrame(summary_data)
    df = df.sort_values("mean_improvement", ascending=False)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [2, 1]}
    )

    # Mean improvement plot
    colors = ["#4CAF50" if x >= 0 else "#F44336" for x in df["mean_improvement"]]
    bars = ax1.bar(df["metric"], df["mean_improvement"], color=colors)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_title("Average Metric Improvement (Modified - Base)", fontsize=14)
    ax1.set_ylabel("Mean Δ", fontsize=12)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
        )

    # Count comparison plot
    bar_width = 0.35
    x = np.arange(len(df))
    ax2.bar(
        x - bar_width / 2, df["improved"], bar_width, label="Improved", color="#4CAF50"
    )
    ax2.bar(
        x + bar_width / 2, df["worsened"], bar_width, label="Worsened", color="#F44336"
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["metric"])
    ax2.set_title("Number of Datasets with Improvement/Worsening", fontsize=14)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_all_comparisons(merged_data, show=True, save_dir=None):
    # Plot individual metrics
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    for metric in metrics:
        save_path = (
            os.path.join(save_dir, f"{metric}_comparison.png") if save_dir else None
        )
        plot_metric_comparison(
            merged_data, metric=metric, show=show, save_path=save_path
        )

    # Plot summary
    summary_path = (
        os.path.join(save_dir, "improvement_summary.png") if save_dir else None
    )
    plot_improvement_summary(merged_data, show=show, save_path=summary_path)


if __name__ == "__main__":
    results_base_path = "results/class_imbalance"
    merged_data = load_and_merge_data(results_base_path)
    comparisons_base_path = "results/class_imbalance/comparisons"
    os.makedirs(comparisons_base_path, exist_ok=True)
    plot_all_comparisons(merged_data, show=False, save_dir=comparisons_base_path)
