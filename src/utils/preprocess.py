import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def preprocess_datasets(
    raw_dir="data/raw/class_imbalance",
    processed_dir="data/processed/class_imbalance",
    test_size=0.3,
    random_state=42,
):
    """
    Preprocess datasets with robust label encoding and validation
    """
    os.makedirs(processed_dir, exist_ok=True)
    label_encoder = LabelEncoder()

    for filename in os.listdir(raw_dir):
        # if filename != "dataset_1000_hypothyroid.csv":
        #     continue

        if not filename.endswith(".csv"):
            continue

        print(f"{processed_dir}:")
        print(f"Processing {filename}...")
        file_path = os.path.join(raw_dir, filename)

        try:
            # Load and validate data
            df = pd.read_csv(file_path)
            if df.shape[1] < 2:
                print(f"Skipping {filename}: Insufficient columns")
                continue

            # Separate features and target
            X = df.iloc[:, :-1].copy()
            y_raw = df.iloc[:, -1].copy()

            # Clean and encode labels
            valid_mask = y_raw.notna()
            X, y_raw = X[valid_mask], y_raw[valid_mask]
            y_raw_sorted = np.sort(y_raw)
            if len(y_raw_sorted) == 0:
                print(f"Skipping {filename}: No valid targets")
                continue

            # Convert labels to 0-indexed integers
            y = pd.Series(label_encoder.fit_transform(y_raw_sorted), name=y_raw.name)
            classes = label_encoder.classes_
            if len(classes) < 2:
                print(f"Skipping {filename}: Only one class present")
                continue

            # Numerical imputation
            num_cols = X.select_dtypes(include=np.number).columns
            if len(num_cols) > 0:
                X[num_cols] = X[num_cols].fillna(X[num_cols].mean()).fillna(0)

            # Categorical imputation
            cat_cols = X.select_dtypes(exclude=np.number).columns
            for col in cat_cols:
                mode = X[col].mode()[0] if not X[col].mode().empty else "missing"
                X[col] = X[col].fillna(mode)

            # Convert categoricals to dummies
            if len(cat_cols) > 0:
                X = pd.get_dummies(X, columns=cat_cols, drop_first=False)

            # Final validation
            X = X.fillna(0).infer_objects()
            assert not X.isna().any().any(), "NaN values in features"
            assert X.shape[0] > 0, "Empty dataset after preprocessing"

            # Normalize numericals
            num_cols = X.select_dtypes(include=np.number).columns
            if len(num_cols) > 0:
                # print("First few datapoints (numerical columns) before normalization:")
                # print(X[num_cols].head(10))
                # print("Datapoint with maximum age:")
                # print(X.loc[X["age"] == X["age"].max()][num_cols])
                # print("Datapoint with minimum age:")
                # print(X.loc[X["age"] == X["age"].min()][num_cols].head(1))

                X[num_cols] = MinMaxScaler().fit_transform(X[num_cols])

                # print("\n--------------------------------------------------\n")
                # print("Formula: (value - column_min) / (column_max - column_min)")
                # print("First few datapoints (numerical columns) after normalization:")
                # print(X[num_cols].head(10))
                # print("Datapoint with maximum age:")
                # print(X.loc[X["age"] == X["age"].max()][num_cols])
                # print("Datapoint with minimum age:")
                # print(X.loc[X["age"] == X["age"].min()][num_cols].head(1))

            # Stratified split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=random_state
            )

            # Save processed data
            base_name = os.path.splitext(filename)[0]
            output_dir = os.path.join(processed_dir, base_name)
            os.makedirs(output_dir, exist_ok=True)

            X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
            X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
            y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
            y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

            print(f"Classes: {list(classes)} → {list(y.unique())}\n")

        except Exception as e:
            print(f"Failed processing {filename}: {str(e)}\n")


if __name__ == "__main__":
    preprocess_datasets()
