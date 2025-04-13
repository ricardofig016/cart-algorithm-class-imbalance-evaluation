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

        # Load and clean data
        df = pd.read_csv(file_path)
        X = df.iloc[:, :-1]  # Features
        y = df.iloc[:, -1]  # Target

        # Remove rows with missing target values
        valid_indices = y.notna()
        X = X[valid_indices].reset_index(drop=True)
        y = y[valid_indices].reset_index(drop=True)

        # Feature cleaning pipeline
        if imputation_strategy == "drop":
            # Remove rows with any missing features
            valid_rows = X.notna().all(axis=1)
            X = X[valid_rows].reset_index(drop=True)
            y = y[valid_rows].reset_index(drop=True)
        else:
            # Handle numerical features
            numerical_cols = X.select_dtypes(include=np.number).columns
            if len(numerical_cols) > 0:
                # Two-stage imputation for all-NaN columns
                col_means = X[numerical_cols].mean()
                X[numerical_cols] = X[numerical_cols].fillna(col_means)
                X[numerical_cols] = X[numerical_cols].fillna(
                    0
                )  # Fallback for all-NaN cols

            # Handle categorical features
            categorical_cols = X.select_dtypes(exclude=np.number).columns
            for col in categorical_cols:
                if X[col].isna().any():
                    mode = X[col].mode()[0] if len(X[col].mode()) > 0 else "missing"
                    X[col] = X[col].fillna(mode)

        # Convert categoricals to numeric
        if len(categorical_cols) > 0:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

        # Final NaN safeguard
        X = X.fillna(0)

        # Normalize numerical features
        numerical_cols = X.select_dtypes(include=np.number).columns
        if len(numerical_cols) > 0:
            scaler = MinMaxScaler()
            X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

        # Validate data integrity
        assert not X.isna().any().any(), "NaN values detected after preprocessing"
        assert not y.isna().any(), "NaN target values after preprocessing"

        # Stratified split
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

        print(f"Successfully processed {base_name}")


if __name__ == "__main__":
    preprocess_datasets(raw_dir="data/raw", processed_dir="data/processed")
