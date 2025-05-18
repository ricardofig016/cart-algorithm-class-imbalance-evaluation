import matplotlib.pyplot as plt
import pandas as pd
import os


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


def plot_all_comparisons(merged_data, show=True, save_dir=None):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    for metric in metrics:
        save_path = (
            os.path.join(save_dir, f"{metric}_comparison.png") if save_dir else None
        )
        plot_metric_comparison(
            merged_data, metric=metric, show=show, save_path=save_path
        )


if __name__ == "__main__":
    results_base_path = "results/class_imbalance"
    merged_data = load_and_merge_data(results_base_path)
    comparisons_base_path = "results/class_imbalance/comparisons"
    os.makedirs(comparisons_base_path, exist_ok=True)
    plot_all_comparisons(merged_data, show=False, save_dir=comparisons_base_path)
