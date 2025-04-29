import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from cart.cart import DecisionTree


def evaluate(data_dir="../data/processed/class_imbalance", max_datasets=-1):
    results = []
    datasets = (
        os.listdir(data_dir)
        if max_datasets == -1
        else os.listdir(data_dir)[:max_datasets]
    )
    for dataset in datasets:
        print(f"Evaluating for dataset: {dataset}")
        dataset_object = {"name": dataset}
        dataset_path = os.path.join(data_dir, dataset)

        # Load preprocessed data
        X_train = pd.read_csv(os.path.join(dataset_path, "X_train.csv")).values
        y_train = pd.read_csv(
            os.path.join(dataset_path, "y_train.csv")
        ).values.flatten()

        # Initialize and train model
        tree = DecisionTree(max_depth=5, criterion="gini")  # Hyperparameter Tuning
        tree.fit(X_train, y_train)

        # Make predictions
        X_test = pd.read_csv(os.path.join(dataset_path, "X_test.csv")).values
        predictions = tree.predict(X_test)

        # Evaluate model performance
        y_test = pd.read_csv(os.path.join(dataset_path, "y_test.csv")).values.flatten()
        accuracy = np.mean(predictions == y_test)
        f1 = f1_score(y_test, predictions)
        roc_auc = roc_auc_score(y_test, predictions)

        dataset_object["accuracy"] = accuracy
        dataset_object["f1"] = f1
        dataset_object["roc_auc"] = roc_auc
        results.append(dataset_object)

    return results


def save_results(results, output_path="../results/class_imbalance/evaluation_data.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame.from_records(results)
    df.to_csv(output_path, index=True)


if __name__ == "__main__":
    results = evaluate()
    save_results(results)
