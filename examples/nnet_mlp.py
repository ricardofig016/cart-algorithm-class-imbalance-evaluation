import logging

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.metrics import roc_auc_score

from mla.metrics.metrics import mean_squared_error
from mla.neuralnet import NeuralNet
from mla.neuralnet.constraints import MaxNorm
from mla.neuralnet.layers import Activation, Dense, Dropout
from mla.neuralnet.optimizers import Adadelta, Adam
from mla.neuralnet.parameters import Parameters
from mla.neuralnet.regularizers import L2
from mla.utils import one_hot
import os 
import pandas as pd 
import numpy as np

logging.basicConfig(level=logging.DEBUG)


def classification():
    base_dir = "multiclass_classification"

    for filename in os.listdir(base_dir):
        file_path = os.path.join(base_dir, filename)
        df = pd.read_csv(file_path)
        newdf = df.dropna(axis=0, how="any")
        print(filename)
        #print("data length", len(df))
        #print("new data length", len(newdf))
        X = newdf.iloc[:, :-1]
        y = newdf.iloc[:, -1]
        y = one_hot(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1111)

        model = NeuralNet(
        layers=[
            Dense(256, Parameters(init="uniform", regularizers={"W": L2(0.05)})),
            Activation("relu"),
            Dropout(0.5),
            Dense(128, Parameters(init="normal", constraints={"W": MaxNorm()})),
            Activation("relu"),
            Dense(2),
            Activation("softmax"),
        ],
        loss="categorical_crossentropy",
        optimizer=Adadelta(),
        metric="accuracy",
        batch_size=64,
        max_epochs=25,
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print("classification accuracy", roc_auc_score(y_test[:, 0], predictions[:, 0]))


def regression():
    # Generate a random regression problem
    X, y = make_regression(n_samples=5000, n_features=25, n_informative=25, n_targets=1, random_state=100, noise=0.05)
    y *= 0.01
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1111)

    model = NeuralNet(
        layers=[
            Dense(64, Parameters(init="normal")),
            Activation("linear"),
            Dense(32, Parameters(init="normal")),
            Activation("linear"),
            Dense(1),
        ],
        loss="mse",
        optimizer=Adam(),
        metric="mse",
        batch_size=256,
        max_epochs=15,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("regression mse", mean_squared_error(y_test, predictions.flatten()))


if __name__ == "__main__":
    classification()
    #regression()
