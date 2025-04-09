import numpy as np

from mla.base import BaseEstimator

class CART(BaseEstimator):
    def __init__(self, criterion = 'gini', prune = 'depth', max_depth = 4, min_criterion = 0.05):
        self.x = None
        self.label = None
        self.n_samples = None
        self.gain = None
        self.left = None
        self.right = None
        self.threshold = None
        self.depth = 0

        self.root = None
        self.criterion = criterion
        self.prune = prune
        self.max_depth = max_depth
        self.min_criterion = min_criterion

    def predict(self, X):
        return np.array([self.root._predict(x) for x in X])
    
    def _grow_tree(self, X, y=None, criterion = 'gini'):
        self.n_samples = X.shape[0] 

        if len(np.unique(y)) == 1:
            self.label = y[0]
            return

        best_gain = 0
        best_x = None
        best_threshold = None

        if criterion in {'gini', 'entropy'}:
            self.label = max([(c, len(y[y == c])) for c in np.unique(y)], key = lambda k : k[1])[0]
        else:
            self.label = np.mean(y)

        impurity_node = self._impurity(criterion, y)
        
        for col in range(X.shape[1]):
            x_level = np.unique(X[:,col])
            thresholds = (x_level[:-1] + x_level[1:]) / 2

            for threshold in thresholds:
                y_l = y[X[:,col] <= threshold]
                impurity_l = self._impurity(criterion, y_l)
                n_l = len(y_l) / self.n_samples

                y_r = y[X[:,col] > threshold]
                impurity_r = self._impurity(criterion, y_r)
                n_r = len(y_r) / self.n_samples

                impurity_gain = impurity_node - (n_l * impurity_l + n_r * impurity_r)
                if impurity_gain > best_gain:
                    best_gain = impurity_gain
                    best_x = col
                    best_threshold = threshold

        self.x = best_x
        self.gain = best_gain
        self.threshold = best_threshold
        self._split_tree(X, y, criterion)

    def _split_tree(self, X, y, criterion):
        X_l = X[X[:, self.x] <= self.threshold]
        y_l = y[X[:, self.x] <= self.threshold]
        self.left = CART()
        self.left.depth = self.depth + 1
        self.left._grow_tree(X_l, y_l, criterion)

        X_r = X[X[:, self.x] > self.threshold]
        y_r = y[X[:, self.x] > self.threshold]
        self.right = CART()
        self.right.depth = self.depth + 1
        self.right._grow_tree(X_r, y_r, criterion)
    
    def _impurity(self, criterion, y=None):
        if criterion == 'gini':
            _, counts = np.unique(y, return_counts=True)
            probabilities = counts / len(y)
            return 1 - np.sum(probabilities ** 2)
        elif criterion == 'mse':
            return np.mean((y - np.mean(y)) ** 2)
        elif criterion == 'entropy':
            _, counts = np.unique(y, return_counts=True)
            probabilities = counts / len(y)
            return -np.sum(probabilities * np.log2(probabilities)) 
        
    def _predict(self, d):
        if self.x != None:
            if d[self.x] <= self.threshold:
                return self.left._predict(d)
            else:
                return self.right._predict(d)
        else: 
            return self.label
    
    def _prune(self, method, max_depth, min_criterion, n_samples):
        if self.x is None:
            return

        self.left._prune(method, max_depth, min_criterion, n_samples)
        self.right._prune(method, max_depth, min_criterion, n_samples)

        pruning = False

        if method == 'impurity' and self.left.x is None and self.right.x is None: 
            if ((self.gain * self.n_samples) / n_samples) < min_criterion:
                pruning = True
        elif method == 'depth' and self.depth >= max_depth:
            pruning = True

        if pruning is True:
            self.left = None
            self.right = None
            self.x = None

    def print_tree(self):
        self.root._show_tree(0, ' ')

    def _show_tree(self, depth, cond):
        base = '    ' * depth + cond
        if self.x != None:
            print(base + 'if X[' + str(self.x) + '] <= ' + str(self.threshold))
            self.left._show_tree(depth+1, 'then ')
            self.right._show_tree(depth+1, 'else ')
        else:
            print(base + '{value: ' + str(self.label) + ', samples: ' + str(self.n_samples) + '}')

class DecisionTreeClassifier(CART):
    def fit(self, X, y=None):
        self.root = CART()
        self.root._grow_tree(X, y, self.criterion)
        self.root._prune(self.prune, self.max_depth, self.min_criterion, self.root.n_samples)

class DecisionTreeRegressor(CART):
    def fit(self, X, y=None):
        self.root = CART()
        self.root._grow_tree(X, y, 'mse')
        self.root._prune(self.prune, self.max_depth, self.min_criterion, self.root.n_samples)
        
        
                