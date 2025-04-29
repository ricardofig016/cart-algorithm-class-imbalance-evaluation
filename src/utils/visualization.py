import matplotlib.pyplot as plt
import pandas as pd


def plot_evaluation_results(data):
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    # plot accuracy, f1 and roc_auc
    metrics = ["accuracy", "f1", "roc_auc"]
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    for idx, metric in enumerate(metrics):
        axs[idx].bar(data["name"], data[metric], color="skyblue")
        axs[idx].set_title(metric.capitalize())
        axs[idx].set_xlabel("Dataset")
        axs[idx].set_ylabel(metric.capitalize())
        axs[idx].set_ylim(0, 1)
        axs[idx].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv("../results/class_imbalance_evaluation.csv")
    plot_evaluation_results(data)
