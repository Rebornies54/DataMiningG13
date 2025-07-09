import numpy as np
from collections import Counter
import random
from config import CUSTOM_RF_PARAMS

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_impurity_decrease=0.0):
        """
        Initialize a Decision Tree classifier.
        
        Parameters
        ----------
        max_depth : int or None, default=None
            Maximum depth of the tree. None means unlimited depth.
        min_samples_split : int, default=2
            Minimum number of samples required to split an internal node.
        min_samples_leaf : int, default=1
            Minimum number of samples required to be at a leaf node.
        min_impurity_decrease : float, default=0.0
            Minimum impurity decrease required for a split.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.root = None
        self.feature_importances_ = None
        self.n_classes_ = None
        self.classes_ = None
        self.class_weights_ = None
        self.feature_indices = None

    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, impurity=None, n_samples=None, probabilities=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value
            self.impurity = impurity
            self.n_samples = n_samples
            self.probabilities = probabilities

    def _gini_impurity(self, y):
        """Calculate weighted Gini impurity for a set of labels"""
        counter = Counter(y)
        impurity = 1
        total = len(y)
        
        # Apply class weights if available
        if self.class_weights_ is not None:
            for label, count in counter.items():
                weight = self.class_weights_.get(label, 1.0)
                prob = (count * weight) / total
                impurity -= prob ** 2
        else:
            for count in counter.values():
                prob = count / total
                impurity -= prob ** 2
                
        return impurity

    def _information_gain(self, X, y, feature, threshold):
        """Calculate information gain using Gini impurity with class weights"""
        parent_impurity = self._gini_impurity(y)
        
        # Split data
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
            return 0
        
        # Calculate weighted impurity
        left_impurity = self._gini_impurity(y[left_mask])
        right_impurity = self._gini_impurity(y[right_mask])
        
        n = len(y)
        left_weight = len(y[left_mask]) / n
        right_weight = len(y[right_mask]) / n
        
        # Calculate weighted impurity reduction
        weighted_impurity = left_weight * left_impurity + right_weight * right_impurity
        gain = parent_impurity - weighted_impurity
        
        # Apply minimum impurity decrease threshold
        if gain < self.min_impurity_decrease:
            return 0
            
        return gain

    def _best_split(self, X, y):
        """Find the best split for the data using multiple criteria"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        feature_importances = np.zeros(n_features)
        
        # Try multiple thresholds for each feature
        for feature in range(n_features):
            # Get unique values for the feature
            unique_values = np.unique(X[:, feature])
            
            # For numerical features, use percentiles
            if len(unique_values) > 10:
                # Use more percentiles for better splits
                percentiles = np.percentile(X[:, feature], [10, 20, 30, 40, 50, 60, 70, 80, 90])
                thresholds = np.unique(percentiles)
            else:
                # For categorical-like features, use all unique values
                thresholds = unique_values
            
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature, threshold)
                feature_importances[feature] += gain
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        # Normalize feature importances
        if np.sum(feature_importances) > 0:
            feature_importances /= np.sum(feature_importances)
        self.feature_importances_ = feature_importances
        
        return best_feature, best_threshold

    def _calculate_node_probabilities(self, y):
        """Calculate class probabilities"""
        class_counts = Counter(y)
        total = sum(class_counts.values())
        probabilities = np.zeros(self.n_classes_)
        
        for i, cls in enumerate(self.classes_):
            count = class_counts.get(cls, 0)
            probabilities[i] = count / total
        
        return probabilities

    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree with pruning"""
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # Store classes
        if self.classes_ is None:
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
        
        # Calculate current node impurity
        current_impurity = self._gini_impurity(y)
        
        # Calculate node probabilities
        probabilities = self._calculate_node_probabilities(y)
        
        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           n_labels == 1 or \
           current_impurity == 0:
            leaf_value = np.argmax(probabilities)
            return self.Node(value=leaf_value, impurity=current_impurity, 
                           n_samples=n_samples, probabilities=probabilities)
        
        # Find best split
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            leaf_value = np.argmax(probabilities)
            return self.Node(value=leaf_value, impurity=current_impurity, 
                           n_samples=n_samples, probabilities=probabilities)
        
        # Create child nodes
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Check minimum samples in leaves
        if len(y[left_mask]) < self.min_samples_leaf or len(y[right_mask]) < self.min_samples_leaf:
            leaf_value = np.argmax(probabilities)
            return self.Node(value=leaf_value, impurity=current_impurity, 
                           n_samples=n_samples, probabilities=probabilities)
        
        # Recursively build subtrees
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return self.Node(
            best_feature, 
            best_threshold, 
            left_subtree, 
            right_subtree,
            impurity=current_impurity,
            n_samples=n_samples,
            probabilities=probabilities
        )

    def _traverse_tree(self, x, node):
        """Traverse the tree to make a prediction"""
        if node.value is not None:
            return node.value, node.probabilities
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def fit(self, X, y):
        """Train the decision tree"""
        # Compute class weights if needed
        if hasattr(self, 'class_weight') and self.class_weight == 'balanced':
            classes = np.unique(y)
            n_samples = len(y)
            weights = np.ones(len(classes))
            for i, c in enumerate(classes):
                weights[i] = n_samples / (len(classes) * np.sum(y == c))
            self.class_weights_ = dict(zip(classes, weights))
        
        self.root = self._build_tree(X, y)
        return self

    def predict_proba(self, X):
        """Predict class probabilities for X"""
        predictions = []
        for x in X:
            _, proba = self._traverse_tree(x, self.root)
            predictions.append(proba)
        return np.array(predictions)

    def predict(self, X):
        """Make predictions for X"""
        predictions = []
        for x in X:
            pred, _ = self._traverse_tree(x, self.root)
            predictions.append(pred)
        return np.array(predictions)

class CustomRandomForest:
    def __init__(self, n_estimators=None, max_depth=None, min_samples_split=None, 
                 min_samples_leaf=None, max_features=None, random_state=None,
                 bootstrap=None, oob_score=None, class_weight=None):
        """
        Initialize a Custom Random Forest classifier.
        
        Parameters
        ----------
        n_estimators : int, default=250
            Number of trees in the forest. Higher values improve robustness but increase computation time.
        max_depth : int or None, default=20
            Maximum depth of trees. None means unlimited depth. Lower values prevent overfitting.
        min_samples_split : int, default=12
            Minimum number of samples required to split an internal node.
        min_samples_leaf : int, default=8
            Minimum number of samples required to be at a leaf node.
        max_features : {'sqrt', 'log2'} or int, default='sqrt'
            Number of features to consider when looking for the best split.
        random_state : int or None, default=42
            Controls the randomness of the bootstrapping and feature selection.
        bootstrap : bool, default=True
            Whether to use bootstrap samples when building trees.
        oob_score : bool, default=True
            Whether to calculate out-of-bag score.
        class_weight : {'balanced', None}, default='balanced'
            Weights associated with classes. 'balanced' adjusts weights inversely proportional to class frequencies.
        """
        # Use parameters from config if not explicitly provided
        self.n_estimators = n_estimators if n_estimators is not None else CUSTOM_RF_PARAMS['n_estimators']
        self.max_depth = max_depth if max_depth is not None else CUSTOM_RF_PARAMS['max_depth']
        self.min_samples_split = min_samples_split if min_samples_split is not None else CUSTOM_RF_PARAMS['min_samples_split']
        self.min_samples_leaf = min_samples_leaf if min_samples_leaf is not None else CUSTOM_RF_PARAMS['min_samples_leaf']
        self.max_features = max_features if max_features is not None else CUSTOM_RF_PARAMS['max_features']
        self.random_state = random_state if random_state is not None else CUSTOM_RF_PARAMS['random_state']
        self.bootstrap = bootstrap if bootstrap is not None else CUSTOM_RF_PARAMS['bootstrap']
        self.oob_score = oob_score if oob_score is not None else CUSTOM_RF_PARAMS['oob_score']
        self.class_weight = class_weight if class_weight is not None else CUSTOM_RF_PARAMS['class_weight']
        
        self.trees = []
        self.feature_importances_ = None
        self.feature_names_ = None
        self.oob_score_ = None
        self.n_classes_ = None
        self.classes_ = None
        
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)
            
        # Validate parameters
        self.validate_parameters()
        
    def validate_parameters(self):
        """Validate model parameters"""
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be greater than 0")
        if self.max_depth is not None and self.max_depth < 1:
            raise ValueError("max_depth must be greater than 0")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1")
        if self.max_features not in ['sqrt', 'log2'] and not isinstance(self.max_features, int):
            raise ValueError("max_features must be 'sqrt', 'log2', or an integer")

    def _bootstrap_sample(self, X, y):
        """Create a bootstrap sample of the data"""
        n_samples = X.shape[0]
        if self.bootstrap:
            indices = np.random.choice(n_samples, n_samples, replace=True)
            oob_indices = np.setdiff1d(np.arange(n_samples), indices)
            return X[indices], y[indices], indices, oob_indices
        else:
            return X, y, np.arange(n_samples), np.array([])

    def _get_feature_subset(self, n_features):
        """Get a random subset of features"""
        if isinstance(self.max_features, int):
            n_features_to_use = min(self.max_features, n_features)
        elif self.max_features == 'sqrt':
            n_features_to_use = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            n_features_to_use = int(np.log2(n_features))
        else:
            n_features_to_use = n_features
        
        return np.random.choice(n_features, n_features_to_use, replace=False)

    def _compute_class_weights(self, y):
        """Compute class weights if specified"""
        if self.class_weight is None:
            return None
        
        classes = np.unique(y)
        n_samples = len(y)
        weights = np.ones(len(classes))
        
        if self.class_weight == 'balanced':
            for i, c in enumerate(classes):
                weights[i] = n_samples / (len(classes) * np.sum(y == c))
        
        return dict(zip(classes, weights))

    def fit(self, X, y, progress_callback=None):
        """Train the random forest with progress tracking"""
        n_samples, n_features = X.shape
        self.trees = []
        feature_importances = np.zeros(n_features)
        
        # Store feature names if X is a DataFrame
        if hasattr(X, 'columns'):
            self.feature_names_ = X.columns
        
        # Store classes
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # Initialize OOB predictions
        if self.oob_score:
            oob_predictions = np.zeros((n_samples, self.n_classes_))
            oob_counts = np.zeros(n_samples)
        
        # Train trees with progress tracking
        for i in range(self.n_estimators):
            # Create bootstrap sample
            X_bootstrap, y_bootstrap, indices, oob_indices = self._bootstrap_sample(X, y)
            
            # Get feature subset
            feature_indices = self._get_feature_subset(n_features)
            X_subset = X_bootstrap[:, feature_indices]
            
            # Create and train decision tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=0.0
            )
            
            # Set class weights and feature indices
            tree.class_weight = self.class_weight
            tree.feature_indices = feature_indices
            
            # Train tree on bootstrap sample with feature subset
            tree.fit(X_subset, y_bootstrap)
            
            # Accumulate feature importances
            tree_importances = np.zeros(n_features)
            tree_importances[feature_indices] = tree.feature_importances_
            feature_importances += tree_importances
            
            # Update OOB predictions
            if self.oob_score and len(oob_indices) > 0:
                X_oob = X[oob_indices][:, feature_indices]
                oob_pred = tree.predict_proba(X_oob)
                oob_predictions[oob_indices] += oob_pred
                oob_counts[oob_indices] += 1
            
            self.trees.append(tree)
            
            # Report progress
            if progress_callback and (i + 1) % 10 == 0:
                progress = (i + 1) / self.n_estimators
                progress_callback(progress)
        
        # Average feature importances across all trees
        self.feature_importances_ = feature_importances / self.n_estimators
        
        # Compute OOB score
        if self.oob_score:
            oob_predictions /= oob_counts[:, np.newaxis]
            oob_pred_classes = np.argmax(oob_predictions, axis=1)
            self.oob_score_ = np.mean(oob_pred_classes == y)
        
        return self

    def predict_proba(self, X):
        """Predict class probabilities for X"""
        # Get predictions from all trees
        predictions = []
        for tree in self.trees:
            X_subset = X[:, tree.feature_indices]
            tree_pred = tree.predict_proba(X_subset)
            predictions.append(tree_pred)
        
        # Average probabilities across all trees
        predictions = np.array(predictions)
        final_probs = np.mean(predictions, axis=0)
        
        # Normalize probabilities to sum to 1
        row_sums = final_probs.sum(axis=1)
        final_probs = final_probs / row_sums[:, np.newaxis]
        
        return final_probs

    def predict(self, X):
        """Make predictions for X"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def score(self, X, y):
        """Return the accuracy score on the given test data and labels"""
        return np.mean(self.predict(X) == y) 