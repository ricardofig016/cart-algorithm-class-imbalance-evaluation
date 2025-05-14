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
        self.class_distribution = None  # Probability distribution for classes


class DecisionTree:
    def __init__(
            self, max_depth=None, min_samples_split=2, criterion="gini", class_weight="balanced", 
            min_weight_fraction_leaf=0.0, min_impurity_decrease=0.0, smoothing_factor=1e-6,
            prediction_confidence_threshold=0.5
            ):
        self.max_depth = max_depth  # Maximum tree depth
        self.min_samples_split = min_samples_split  # Minimum samples to split
        self.criterion = criterion.lower()  # Impurity measure (gini/entropy)
        self.class_weight = class_weight  # Class weight: 'balanced', None, or dictionary
        self.min_weight_fraction_leaf = min_weight_fraction_leaf  # Minimum weighted fraction of samples in leaf
        self.min_impurity_decrease = min_impurity_decrease  # Minimum impurity decrease for split
        self.smoothing_factor = smoothing_factor  # Smoothing factor for numerical stability
        self.prediction_confidence_threshold = prediction_confidence_threshold  # Minimum confidence for predictions
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
        # Input validation
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            X = np.array(X)
            y = np.array(y)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y have incompatible shapes: {X.shape[0]} vs {y.shape[0]}")
            
        if X.shape[0] == 0:
            raise ValueError("Cannot train on empty dataset")
            
        self.root = self._grow_tree(X, y, depth=0)

    def predict(self, X):
        """Predict class labels for input samples"""
        return np.array([self._traverse(x, self.root)[0] for x in X])
        
    def predict_proba(self, X):
        """Predict class probabilities for input samples"""
        return np.array([self._get_leaf_distribution(x, self.root) for x in X])

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
        
        # Calculate weighted sample count (for min_weight_fraction_leaf check)
        sample_weights = np.ones(len(y))
        for i, label in enumerate(y):
            sample_weights[i] = self.weights[label]
        weighted_n_samples = np.sum(sample_weights)
        
        # Stopping conditions
        if (
            (self.max_depth and depth >= self.max_depth)
            or (node.samples < self.min_samples_split)
            or (len(np.unique(y)) == 1)
            or (weighted_n_samples < self.min_weight_fraction_leaf * np.sum(self.weights))
        ):
            node.value = self._most_common(y)
            # Also store class distribution for probabilistic predictions
            node.class_distribution = self._get_class_distribution(y)
            return node

        # Find optimal split
        feature, threshold = self._best_split(X, y)
        if feature is None:
            node.value = self._most_common(y)
            # Also store class distribution for probabilistic predictions
            node.class_distribution = self._get_class_distribution(y)
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
        
        # Total weighted samples in parent
        parent_weights = np.zeros(len(y))
        for i, label in enumerate(y):
            parent_weights[i] = self.weights[label]
        total_weighted_samples = np.sum(parent_weights)
        
        min_weight_leaf = self.min_weight_fraction_leaf * total_weighted_samples
        
        for feature in range(X.shape[1]):
            # Use faster approach to find potential thresholds
            feature_values = X[:, feature]
            sorted_idx = np.argsort(feature_values)
            sorted_y = y[sorted_idx]
            sorted_weights = parent_weights[sorted_idx]
            
            # Get unique values in sorted order
            thresholds = np.unique(feature_values)
            
            for threshold in thresholds:
                left_idx = feature_values <= threshold
                right_idx = ~left_idx
                
                left_y = y[left_idx]
                right_y = y[right_idx]
                
                if len(left_y) == 0 or len(right_y) == 0:
                    continue
                
                # Calculate weighted sample counts for each child node
                left_weights = parent_weights[left_idx]
                right_weights = parent_weights[right_idx]
                left_weight_sum = np.sum(left_weights)
                right_weight_sum = np.sum(right_weights)
                
                # Check minimum weighted leaf samples
                if left_weight_sum < min_weight_leaf or right_weight_sum < min_weight_leaf:
                    continue
                
                # Calculate weighted impurity and information gain
                n_left, n_right = len(left_y), len(right_y)
                n_total = n_left + n_right
                
                left_impurity = self._calculate_weighted_impurity(left_y)
                right_impurity = self._calculate_weighted_impurity(right_y)
                
                # Weighted average of child impurities using sample weights
                child_impurity = (left_weight_sum / total_weighted_samples) * left_impurity + \
                                (right_weight_sum / total_weighted_samples) * right_impurity
                
                # Information gain with regularization
                gain = parent_impurity - child_impurity
                
                # Apply minimum impurity decrease constraint
                if gain < self.min_impurity_decrease:
                    continue
                
                # Regularization: slightly penalize imbalanced splits
                balance_factor = 4 * (left_weight_sum * right_weight_sum) / (total_weighted_samples ** 2)
                regularized_gain = gain * balance_factor
                
                if regularized_gain > best_gain:
                    best_gain = regularized_gain
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
        """Calculate impurity using class weights with smoothing for stability"""
        if len(y) == 0:
            return 0
            
        # Get counts and normalize to get proportions
        counts = np.bincount(y, minlength=self.n_classes)
        total_samples = len(y)
        
        # Apply weights to the proportions with smoothing factor
        weighted_proportions = np.zeros(self.n_classes)
        for cls in range(self.n_classes):
            # Weight the proportion by the class weight (add smoothing to avoid division by zero)
            weighted_proportions[cls] = ((counts[cls] + self.smoothing_factor) / 
                                        (total_samples + self.n_classes * self.smoothing_factor)) * self.weights[cls]
        
        # Normalize the weighted proportions
        sum_weighted_props = np.sum(weighted_proportions)
        if sum_weighted_props > 0:
            weighted_proportions = weighted_proportions / sum_weighted_props
        
        # Small sample bias correction - increase impurity estimate for small samples
        # Miller's bias correction factor inversely proportional to sample size
        bias_correction = 1.0
        if total_samples < 100:  # Only apply correction for small samples
            bias_correction = 1.0 + (10.0 / total_samples)
            
        if self.criterion == "gini":
            # Compute Gini impurity with small sample bias correction
            impurity = 1 - np.sum(weighted_proportions**2)
            # Apply bias correction (higher correction = higher impurity)
            impurity = min(1.0, impurity * bias_correction)
            return impurity
            
        elif self.criterion == "entropy":
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            weighted_proportions = np.clip(weighted_proportions, epsilon, 1.0)
            # Compute entropy with small sample bias correction
            impurity = -np.sum(weighted_proportions * np.log2(weighted_proportions))
            # Apply bias correction (bounded by max entropy)
            max_entropy = np.log2(self.n_classes)
            impurity = min(max_entropy, impurity * bias_correction)
            return impurity
            
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
    
    def _get_class_distribution(self, y):
        """Get weighted probability distribution of classes"""
        if len(y) == 0:
            # Default to uniform distribution with small smoothing factor
            return np.ones(self.n_classes) / self.n_classes
            
        # Get counts for each class
        counts = np.bincount(y, minlength=self.n_classes)
        total_samples = len(y)
        
        # Apply smoothing and class weights
        weighted_distribution = np.zeros(self.n_classes)
        
        for cls in range(self.n_classes):
            # Apply Laplace smoothing
            smoothed_count = counts[cls] + self.smoothing_factor
            smoothed_total = total_samples + self.n_classes * self.smoothing_factor
            
            # Weight by class weight
            weighted_distribution[cls] = (smoothed_count / smoothed_total) * self.weights[cls]
        
        # Normalize to get probabilities
        if np.sum(weighted_distribution) > 0:
            weighted_distribution = weighted_distribution / np.sum(weighted_distribution)
        else:
            weighted_distribution = np.ones(self.n_classes) / self.n_classes
            
        return weighted_distribution
        
    def _most_common(self, y):
        """Find prediction for a leaf node, considering class weights"""
        if len(y) == 0:
            return 0
            
        # Use the class distribution to determine the prediction
        distribution = self._get_class_distribution(y)
        return np.argmax(distribution)
    def _traverse(self, x, node):
        """Traverse tree to make prediction with confidence
        
        Returns:
            tuple: (prediction, confidence) where prediction is the class label
                  and confidence is the probability for that class
        """
        # If at a leaf node, return prediction and confidence
        if node.value is not None:
            distribution = node.class_distribution
            confidence = distribution[node.value]
            
            # Use confidence threshold to potentially adjust prediction
            if confidence < self.prediction_confidence_threshold:
                # If confidence is too low and we have a clear second choice with significant probability
                sorted_probs = np.argsort(distribution)[::-1]  # Sort in descending order
                if len(sorted_probs) > 1 and distribution[sorted_probs[1]] > 0.3:  # If second class has >30% probability
                    # Consider class weights when choosing between top classes
                    weighted_distribution = distribution * self.weights
                    # Normalize
                    if np.sum(weighted_distribution) > 0:
                        weighted_distribution = weighted_distribution / np.sum(weighted_distribution)
                    # Take the highest weighted class
                    prediction = np.argmax(weighted_distribution)
                    confidence = distribution[prediction]
                else:
                    # Otherwise stick with the original prediction
                    prediction = node.value
            else:
                prediction = node.value
                
            return (prediction, confidence)
        
        # Otherwise continue traversing
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)
        
    def _get_leaf_distribution(self, x, node):
        """Get probability distribution at leaf node for a sample"""
        # If at a leaf node, return the distribution
        if node.value is not None:
            return node.class_distribution
            
        # Otherwise continue traversing
        if x[node.feature] <= node.threshold:
            return self._get_leaf_distribution(x, node.left)
        return self._get_leaf_distribution(x, node.right)

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
        tree = DecisionTree(
            max_depth=5, 
            criterion="gini", 
            class_weight="balanced",
            min_weight_fraction_leaf=0.01,
            min_impurity_decrease=0.001,
            smoothing_factor=1e-6,
            prediction_confidence_threshold=0.6
        )
        tree.fit(X_train, y_train)

        # Make predictions
        X_test = pd.read_csv(os.path.join(dataset_path, "X_test.csv")).values
        predictions = tree.predict(X_test)

        # Test
        y_test = pd.read_csv(os.path.join(dataset_path, "y_test.csv")).values.flatten()
        accuracy = np.mean(predictions == y_test)
        print(f"Accuracy: {accuracy * 100:.2f}%")

