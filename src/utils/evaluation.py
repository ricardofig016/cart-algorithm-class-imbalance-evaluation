import os
import numpy as np
import pandas as pd
from cart.cart import DecisionTree


def basic_accuracy_eval(data_dir="../data/processed/class_imbalance"):
    for dataset in os.listdir(data_dir):
        # print(f"{data_dir}")
        print(f"\nProcessing dataset: {dataset}")
        dataset_path = os.path.join(data_dir, dataset)

        # Load preprocessed data
        X_train = pd.read_csv(os.path.join(dataset_path, "X_train.csv")).values
        y_train = pd.read_csv(
            os.path.join(dataset_path, "y_train.csv")
        ).values.flatten()

        # Initialize and train model
        tree = DecisionTree(max_depth=5, criterion="gini")
        tree.fit(X_train, y_train)

        # Make predictions
        X_test = pd.read_csv(os.path.join(dataset_path, "X_test.csv")).values
        predictions = tree.predict(X_test)

        # Test
        y_test = pd.read_csv(os.path.join(dataset_path, "y_test.csv")).values.flatten()
        accuracy = np.mean(predictions == y_test)
        print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    basic_accuracy_eval()
