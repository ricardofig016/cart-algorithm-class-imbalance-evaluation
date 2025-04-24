import numpy as np
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from mla.cart import DecisionTreeClassifier, DecisionTreeRegressor
import os
import pandas as pd 
import numpy as np

def classification():
    base_dir = "data/raw/class_imbalance"

    for filename in os.listdir(base_dir):
        file_path = os.path.join(base_dir, filename)
        df = pd.read_csv(file_path)
        newdf = df.dropna(axis=0, how="any")
        #print(filename)
        #print("data length", len(df))
        #print("new data length", len(newdf))
        X = newdf[newdf.columns[:-1]]
        y = newdf.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = DecisionTreeClassifier(max_depth=3)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print("classification accuracy", accuracy_score(y_test, predictions))

def regression():
    # Generate a random regression problem.
    X, y = make_regression(n_samples=1000, n_features=20, n_informative=10, noise=0.1, random_state=42)
    rng = np.random.RandomState(1)
    X = np.sort(5 * rng.rand(80, 1), axis = 0)
    y = np.sin(X).ravel()
    y[::5] += 3 * (0.5 - rng.rand(16))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = DecisionTreeRegressor(max_depth=4)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("classification accuracy", roc_auc_score(y_test, predictions))

if __name__ == '__main__':
    classification()
    #regression()