import numpy as np

from mla.base import BaseEstimator

class DTBase(BaseEstimator):
    def __init__(self, min_samples_split=2, max_depth=None, min_impurity_decrease=0.0):
        """
        Initializes the Decision Tree Base class with the specified parameters.

        Parameters:
        min_samples_split (int, optional): The minimum number of samples required to split an internal node. Defaults to 2.
        max_depth (int, optional): The maximum depth of the decision tree. If None, the tree will grow until all leaves are pure or until all leaves contain less than min_samples_split samples. Defaults to None.
        min_impurity_decrease (float, optional): The minimum impurity decrease required for a split to be considered. Defaults to 0.0.
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease

    def fit(self, X, y=None):
        self.tree_ = self._build_tree(X, y)
        return self

    def _predict(self, X, y=None):
        return self.tree_.predict(X)
    
    def _gini_impurity(self, y):
        """
        Calculates the Gini impurity of a given target variable.

        Parameters:
        y (numpy.ndarray): A 1D array-like object containing the target variable values.

        Returns:
        float: The Gini impurity of the target variable.
        """
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)
    
    def _entropy(self, y):
        """
        Calculates the entropy of a given target variable.

        Parameters:
        y (numpy.ndarray): A 1D array-like object containing the target variable values.

        Returns:
        float: The entropy of the target variable. The entropy is a measure of the uncertainty or randomness in the data.
        """
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities))
    
    def _impurity(self, y):
        """
        Calculates the combined Gini impurity and entropy of a given target variable.

        Parameters:
        y (numpy.ndarray): A 1D array-like object containing the target variable values.

        Returns:
        float: The combined Gini impurity and entropy of the target variable.
        """
        gini = self._gini_impurity(y)
        entropy = self._entropy(y)
        return gini + entropy
    
    def _split_node(self, X, y, indices, feature_idx, split_value):
        """
        Splits a node in the decision tree based on a given feature and split value.

        Parameters:
        X (numpy.ndarray): A 2D array-like object containing the feature matrix.
        y (numpy.ndarray): A 1D array-like object containing the target variable values.
        indices (numpy.ndarray): A 1D array-like object containing the indices of the samples to split.
        feature_idx (int): The index of the feature to split on.
        split_value (float): The value to split the feature on.

        Returns:
        tuple: A tuple containing the indices of the samples in the left child node, the indices of the samples in the right child node, and the split value.
        """
        indices_left = np.where(X[indices, feature_idx] <= split_value)[0]
        indices_right = np.where(X[indices, feature_idx] > split_value)[0]

        return indices_left, indices_right, split_value
    
class DTClassifier(DTBase):
    def _predict(self, X, node, depth=0):
        if depth >= self.max_depth or node.childs is None:
            return node.value
        
class DTRegressor(DTBase):
    def _predict(self, X, node, depth=0):
        if depth >= self.max_depth or node.childs is None:
            return node.value
        
        
                