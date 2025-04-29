import matplotlib.pyplot as plt
import pandas as pd


def plot_metric(data, metric, show=True, save_path=None):
    """
    Plot the specified metric.

    Parameters:
        data (pd.DataFrame or convertible): Data containing 'name' and the specified metric column.
        metric (str): The metric to plot (e.g., 'f1', 'accuracy', 'roc_auc').
        show (bool): If True, calls plt.show() to display the plot.
        save_path (str or None): If provided, saves the plot to this filepath.
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(data["name"], data[metric], color="skyblue")
    ax.set_title(f"{metric.capitalize()} Score")
    ax.set_xlabel("Dataset")
    ax.set_ylabel(metric.capitalize())
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()

    plt.close(fig)


def plot_all_metrics(data, show=True, save_dir=None):
    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    for metric in metrics:
        plot_metric(
            data,
            metric=metric,
            show=show,
            save_path=save_dir and os.path.join(save_dir, f"{metric}_plot.png"),
        )


if __name__ == "__main__":
    import os

    results_base_path = "../results/class_imbalance"
    os.makedirs(results_base_path, exist_ok=True)
    data = pd.read_csv(os.path.join(results_base_path, "evaluation_data.csv"))

    plot_all_metrics(data, show=False, save_dir=results_base_path)
