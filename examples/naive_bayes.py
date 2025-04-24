from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import os 
import pandas as pd
import numpy as np
from mla.naive_bayes import NaiveBayesClassifier
from sklearn.preprocessing import LabelEncoder

def classification():
    base_dir = "data/raw/class_imbalance"

    for filename in os.listdir(base_dir):
        file_path = os.path.join(base_dir, filename)
        df = pd.read_csv(file_path)
        newdf = df.dropna()
        X = newdf.iloc[:, :-1]
        y = newdf.iloc[:, -1]
        if (list(np.unique(y)) != [0, 1]):
            le = LabelEncoder()
            label = le.fit_transform(newdf.iloc[:, -1])
            newdf.drop(newdf.iloc[:, -1], axis=1, inplace=True)
            newdf.iloc[:, -1] = label
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
        tree = NaiveBayesClassifier()
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)[:,1]
        print("Accuracy:", roc_auc_score(y_test, y_pred))


if __name__ == "__main__":
    classification()