import matplotlib.pyplot as plt
import pandas as pd


def plot_accuracy(data, show=True, save_path=None):
    """
    Plot the accuracy metric.

    Parameters:
        data (pd.DataFrame or convertible): Data containing 'name' and 'accuracy' columns.
        show (bool): If True, calls plt.show() to display the plot.
        save_path (str or None): If provided, saves the plot to this filepath.
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(data["name"], data["accuracy"], color="skyblue")
    ax.set_title("Accuracy")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()

    plt.close(fig)


def plot_f1(data, show=True, save_path=None):
    """
    Plot the F1 score metric.

    Parameters:
        data (pd.DataFrame or convertible): Data containing 'name' and 'f1' columns.
        show (bool): If True, calls plt.show() to display the plot.
        save_path (str or None): If provided, saves the plot to this filepath.
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(data["name"], data["f1"], color="skyblue")
    ax.set_title("F1 Score")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()

    plt.close(fig)


def plot_roc_auc(data, show=True, save_path=None):
    """
    Plot the ROC AUC metric.

    Parameters:
        data (pd.DataFrame or convertible): Data containing 'name' and 'roc_auc' columns.
        show (bool): If True, calls plt.show() to display the plot.
        save_path (str or None): If provided, saves the plot to this filepath.
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(data["name"], data["roc_auc"], color="skyblue")
    ax.set_title("ROC AUC")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("ROC AUC")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    import os

    results_base_path = "../results/class_imbalance"
    os.makedirs(results_base_path, exist_ok=True)
    data = pd.read_csv(os.path.join(results_base_path, "evaluation_data.csv"))

    plot_accuracy(
        data, show=False, save_path=os.path.join(results_base_path, "accuracy_plot.png")
    )
    plot_f1(data, show=False, save_path=os.path.join(results_base_path, "f1_plot.png"))
    plot_roc_auc(
        data, show=False, save_path=os.path.join(results_base_path, "roc_auc_plot.png")
    )
