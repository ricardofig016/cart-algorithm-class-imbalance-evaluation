import numpy as np
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from mla.cart import DTClassifier, DTRegressor

def classification():
    # Generate a random binary classification problem.
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=10, n_classes=2, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1111)

    model = DTClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("classification accuracy", accuracy_score(y_test, predictions))

def regression():
    # Generate a random regression problem
    X, y = make_regression(n_samples=1000, n_features=20, n_informative=10, noise=0.1, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1111)

    model = DTRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("classification accuracy", roc_auc_score(y_test, predictions))

if __name__ == '__main__':
    classification()
    regression()