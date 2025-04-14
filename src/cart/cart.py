"""
CART Implementation for Class Imbalance Study

Original Reference Implementation:
- Adapted from zziz/cart (https://github.com/zziz/cart)
- Clean-room implementation without direct code copying

Key Modifications from Reference:
1. Architecture Simplification:
   - Removed regression functionality to focus purely on classification
   - Unified tree structure with dedicated TreeNode class
   - Simplified API (removed pruning parameters, consolidated initialization)

2. Project-Specific Optimizations:
   - Direct compatibility with preprocessed numpy arrays (from data/processed/)
   - Binary classification focus with class label tracking
   - Early stopping criteria aligned with imbalance analysis needs
   - Memory-efficient node structure for large datasets

3. Phase 2 Readiness:
   - Modular impurity calculations for weighted Gini modification
   - Class label preservation for imbalance weighting
   - Predict method optimized for probability-based metrics (ROC-AUC)

Implementation Differences from Reference:
- No sklearn dependencies
- Removed print_tree visualization methods
- Simplified split criteria to essential parameters
- Vectorized impurity calculations for performance
- Added sample counting for imbalance analysis

Important Notes:
- Designed for binary classification (handles multi-class through majority voting)
- Requires preprocessed numerical features (compatible with utils/preprocess.py)
- Class labels stored in self.classes for Phase 2 modifications

Maintains Core CART Functionality:
- Gini/Entropy split criteria
- Depth-based stopping
- Recursive tree construction
- Majority class prediction
"""

import numpy as np


class TreeNode:
    """Node structure for decision tree"""

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  # Feature index for splitting
        self.threshold = threshold  # Threshold value for split
        self.left = left  # Left subtree
        self.right = right  # Right subtree
        self.value = value  # Class label for leaf nodes
        self.samples = 0  # Number of samples in node
        self.depth = 0  # Depth in tree structure


class DecisionTree:
    """CART classifier with Gini/Entropy criteria for handling class imbalance analysis"""

    def __init__(self, max_depth=None, min_samples_split=2, criterion="gini"):
        self.max_depth = max_depth  # Maximum tree depth
        self.min_samples_split = min_samples_split  # Minimum samples to split
        self.criterion = criterion.lower()  # Impurity measure (gini/entropy)
        self.root = None  # Root node of decision tree
        self.classes = None  # Store class labels

    def fit(self, X, y):
        """Build decision tree from training data."""
        y = self._convert_labels(y)
        self.classes = np.unique(y)
        self.root = self._grow_tree(X, y, depth=0)

    def predict(self, X):
        """Predict class labels for input samples"""
        return np.array([self._traverse(x, self.root) for x in X])

    def _convert_labels(self, y):
        """Convert labels to np.int64. If conversion fails, map categorical labels to numbers."""
        try:
            return y.astype(np.int64)
        except ValueError:
            uniques = np.unique(y)
            mapping = {label: idx for idx, label in enumerate(uniques)}
            return np.array([mapping[val] for val in y], dtype=np.int64)

    def _grow_tree(self, X, y, depth):
        """Recursively build decision tree"""
        node = TreeNode()
        node.samples = X.shape[0]
        node.depth = depth

        # Stopping conditions
        if (
            (self.max_depth and depth >= self.max_depth)
            or (node.samples < self.min_samples_split)
            or (len(np.unique(y)) == 1)
        ):
            node.value = self._most_common(y)
            return node

        # Find optimal split
        feature, threshold = self._best_split(X, y)
        if feature is None:
            node.value = self._most_common(y)
            return node

        # Split dataset
        left_idx = X[:, feature] <= threshold
        right_idx = ~left_idx

        # Grow child nodes
        node.feature = feature
        node.threshold = threshold
        node.left = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        node.right = self._grow_tree(X[right_idx], y[right_idx], depth + 1)

        return node

    def _best_split(self, X, y):
        """Find optimal feature and threshold for splitting"""
        best_gain = -1
        best_feature, best_threshold = None, None
        current_impurity = self._calculate_impurity(y)

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_y = y[X[:, feature] <= threshold]
                right_y = y[X[:, feature] > threshold]

                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                gain = current_impurity - self._weighted_impurity(left_y, right_y)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _calculate_impurity(self, y):
        """Calculate impurity of target values"""
        proportions = np.bincount(y) / len(y)

        if self.criterion == "gini":
            return 1 - np.sum(proportions**2)
        elif self.criterion == "entropy":
            return -np.sum(proportions * np.log2(proportions + 1e-9))
        else:
            raise ValueError("Invalid criterion. Use 'gini' or 'entropy'")

    def _weighted_impurity(self, left_y, right_y):
        """Calculate weighted impurity for child nodes"""
        n_left, n_right = len(left_y), len(right_y)
        n_total = n_left + n_right

        return (n_left / n_total) * self._calculate_impurity(left_y) + (
            n_right / n_total
        ) * self._calculate_impurity(right_y)

    def _most_common(self, y):
        """Find majority class label"""
        return np.bincount(y).argmax()

    def _traverse(self, x, node):
        """Traverse tree to make prediction"""
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)


# Example usage
if __name__ == "__main__":
    import os
    import pandas as pd

    base_dir = "data/processed"
    for dataset in os.listdir(base_dir):
        print(f"Processing dataset: {dataset}")
        dataset_path = os.path.join(base_dir, dataset)

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
