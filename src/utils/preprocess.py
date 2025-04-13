import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def preprocess_datasets(
    raw_dir="data/raw",
    processed_dir="data/processed",
    test_size=0.2,
    random_state=42,
    imputation_strategy="mean",
):
    """
    Preprocess datasets for class imbalance challenge

    Parameters:
    - raw_dir: Input directory with raw CSV files
    - processed_dir: Output directory for processed datasets
    - test_size: Proportion for test split (default 0.2)
    - random_state: Random seed for reproducibility
    - imputation_strategy: Strategy for handling missing values ('mean' or 'drop')
    """

    os.makedirs(processed_dir, exist_ok=True)

    for filename in os.listdir(raw_dir):
        if not filename.endswith(".csv"):
            continue

        print(f"\nProcessing {filename}...")
        file_path = os.path.join(raw_dir, filename)

        # Load and prepare data
        df = pd.read_csv(file_path)
        X = df.iloc[:, :-1]  # Features
        y = df.iloc[:, -1]  # Target

        # Handle missing values in target
        valid_indices = y.notna()
        X = X[valid_indices]
        y = y[valid_indices]

        # Handle missing values in features
        if imputation_strategy == "drop":
            # Drop rows with any missing values
            valid_rows = X.notna().all(axis=1)
            X = X[valid_rows]
            y = y[valid_rows]
        else:
            # Impute missing values
            numerical_cols = X.select_dtypes(include=np.number).columns
            categorical_cols = X.select_dtypes(exclude=np.number).columns

            # Impute numerical features
            if len(numerical_cols) > 0:
                X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].mean())

            # Impute categorical features
            for col in categorical_cols:
                X[col] = X[col].fillna(
                    X[col].mode()[0] if len(X[col].mode()) > 0 else "missing"
                )

        # Convert categorical variables to one-hot encoding
        if len(categorical_cols) > 0:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

        # Normalize numerical features
        if len(numerical_cols) > 0:
            scaler = MinMaxScaler()
            X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

        # Stratified train-test split (maintain class imbalance)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        # Save processed data
        base_name = os.path.splitext(filename)[0]
        final_dir = os.path.join(processed_dir, base_name)
        os.makedirs(final_dir, exist_ok=True)
        X_train.to_csv(os.path.join(final_dir, "X_train.csv"), index=False)
        X_test.to_csv(os.path.join(final_dir, "X_test.csv"), index=False)
        y_train.to_csv(os.path.join(final_dir, "y_train.csv"), index=False)
        y_test.to_csv(os.path.join(final_dir, "y_test.csv"), index=False)

        print(f"Saved processed data for {base_name}")


if __name__ == "__main__":
    # Example usage with default parameters
    preprocess_datasets(
        raw_dir="data/raw",
        processed_dir="data/processed",
        imputation_strategy="mean",  # Change to 'drop' to remove rows with missing values
    )
