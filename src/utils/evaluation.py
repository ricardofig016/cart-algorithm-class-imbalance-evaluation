import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from cart.cart import DecisionTree


def evaluate(data_dir="../data/processed/class_imbalance"):
    results = {}
    for dataset in os.listdir(data_dir):
        print(f"\nProcessing dataset: {dataset}")

        results[dataset] = []
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
        roc = roc_auc_score(y_test, predictions)
        results[dataset].append(accuracy)
        results[dataset].append(f1)
        results[dataset].append(roc)
    return results


if __name__ == "__main__":
    evaluate()
