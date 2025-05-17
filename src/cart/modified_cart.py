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
    def __init__(self, max_depth=None, min_samples_split=2, criterion="gini", class_weight="balanced"):
        self.max_depth = max_depth  # Maximum tree depth
        self.min_samples_split = min_samples_split  # Minimum samples to split
        self.criterion = criterion.lower()  # Impurity measure (gini/entropy)
        self.class_weight = class_weight  # Class weight: 'balanced', None, or dictionary
        self.root = None  # Root node of decision tree
        self.classes = None  # Store class labels
        self.n_classes = 0  # Number of classes
        self.weights = None  # Class weights computed during fitting

    def fit(self, X, y):
        """Build decision tree from training data."""
        y = self._convert_labels(y)
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        # Compute class weights before growing the tree
        self.weights = self._compute_class_weights(y)
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
        
        # Calculate parent node impurity
        parent_impurity = self._calculate_weighted_impurity(y)
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = ~left_idx
                
                left_y = y[left_idx]
                right_y = y[right_idx]
                
                if len(left_y) == 0 or len(right_y) == 0:
                    continue
                
                # Calculate weighted impurity and information gain
                n_left, n_right = len(left_y), len(right_y)
                n_total = n_left + n_right
                
                left_impurity = self._calculate_weighted_impurity(left_y)
                right_impurity = self._calculate_weighted_impurity(right_y)
                
                # Weighted average of child impurities
                child_impurity = (n_left / n_total) * left_impurity + (n_right / n_total) * right_impurity
                
                # Information gain
                gain = parent_impurity - child_impurity
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _calculate_impurity(self, y):
        """Calculate traditional (unweighted) impurity of target values"""
        proportions = np.bincount(y, minlength=self.n_classes) / len(y)
        proportions = proportions[proportions > 0]  # Remove zero proportions to avoid log(0)
        
        if self.criterion == "gini":
            return 1 - np.sum(proportions**2)
        elif self.criterion == "entropy":
            return -np.sum(proportions * np.log2(proportions))
        else:
            raise ValueError("Invalid criterion. Use 'gini' or 'entropy'")
            
    def _calculate_weighted_impurity(self, y):
        """Calculate impurity using class weights"""
        if len(y) == 0:
            return 0
            
        # Get counts and normalize to get proportions
        counts = np.bincount(y, minlength=self.n_classes)
        total_samples = len(y)
        
        # Apply weights to the proportions
        weighted_proportions = np.zeros(self.n_classes)
        for cls in range(self.n_classes):
            if counts[cls] > 0:
                # Weight the proportion by the class weight
                weighted_proportions[cls] = (counts[cls] / total_samples) * self.weights[cls]
                
        # Normalize the weighted proportions
        if np.sum(weighted_proportions) > 0:
            weighted_proportions = weighted_proportions / np.sum(weighted_proportions)
        
        # Filter out zero proportions
        non_zero_props = weighted_proportions[weighted_proportions > 0]
        
        if self.criterion == "gini":
            return 1 - np.sum(non_zero_props**2)
        elif self.criterion == "entropy":
            return -np.sum(non_zero_props * np.log2(non_zero_props))
        else:
            raise ValueError("Invalid criterion. Use 'gini' or 'entropy'")

    def _weighted_impurity(self, left_y, right_y):
        """Calculate weighted impurity for child nodes"""
        n_left, n_right = len(left_y), len(right_y)
        n_total = n_left + n_right
        
        if n_total == 0:
            return 0
            
        return (n_left / n_total) * self._calculate_weighted_impurity(left_y) + (
            n_right / n_total
        ) * self._calculate_weighted_impurity(right_y)
    
    def _compute_class_weights(self, y):
        """Compute class weights based on the specified weighting strategy"""
        if self.class_weight is None:
            # No weighting - use uniform weights
            return np.ones(self.n_classes)
        elif self.class_weight == "balanced":
            # Balanced weighting - inverse frequency
            class_counts = np.bincount(y, minlength=self.n_classes)
            n_samples = len(y)
            weights = np.zeros(self.n_classes)
            for cls in range(self.n_classes):
                if class_counts[cls] > 0:
                    weights[cls] = n_samples / (self.n_classes * class_counts[cls])
            return weights
        elif isinstance(self.class_weight, dict):
            # Manual weighting - user-specified weights
            weights = np.ones(self.n_classes)
            for cls, weight in self.class_weight.items():
                if isinstance(cls, str) and not cls.isdigit():
                    # Try to find class in self.classes if it's a string
                    # This handles cases where class labels are non-numeric
                    continue  # Skip if class mapping isn't set up yet
                elif int(cls) < self.n_classes:
                    weights[int(cls)] = weight
            return weights
        else:
            raise ValueError("class_weight must be 'balanced', None, or a dictionary")
    
    def _most_common(self, y):
        """Find prediction for a leaf node, considering class weights"""
        if len(y) == 0:
            return 0
            
        # Get class counts
        counts = np.bincount(y, minlength=self.n_classes)
        
        # Apply class weights to the counts
        weighted_counts = counts * self.weights
        
        # Return the class with highest weighted count
        return np.argmax(weighted_counts)

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

    base_dir = "data/processed/class_imbalance"
    for dataset in os.listdir(base_dir):
        print(f"{base_dir}")
        print(f"Processing dataset: {dataset}")
        dataset_path = os.path.join(base_dir, dataset)

        # Load preprocessed data
        X_train = pd.read_csv(os.path.join(dataset_path, "X_train.csv")).values
        y_train = pd.read_csv(os.path.join(dataset_path, "y_train.csv")).values.flatten()

        # Initialize and train model
        tree = DecisionTree(max_depth=5, criterion="gini", class_weight="balanced")
        tree.fit(X_train, y_train)

        # Make predictions
        X_test = pd.read_csv(os.path.join(dataset_path, "X_test.csv")).values
        predictions = tree.predict(X_test)

        # Test
        y_test = pd.read_csv(os.path.join(dataset_path, "y_test.csv")).values.flatten()
        accuracy = np.mean(predictions == y_test)
        print(f"Accuracy: {accuracy * 100:.2f}%") 
        
        # Calculate per-class metrics
        #nique_classes = np.unique(y_test)
        #for cls in unique_classes:
         #   cls_mask = y_test == cls
          #  if np.sum(cls_mask) > 0:  # If class exists in test set
           #     cls_accuracy = np.mean(predictions[cls_mask] == y_test[cls_mask])
            #    print(f"Class {cls} accuracy: {cls_accuracy * 100:.2f}% (samples: {np.sum(cls_mask)})")