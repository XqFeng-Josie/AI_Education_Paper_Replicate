"""
Model implementation module
Implement Naive Baseline, Decision Tree, and Random Forest models
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class NaiveBaselineClassifier(BaseEstimator, ClassifierMixin):
    """
    Naive Baseline Classifier
    
    Setup A: use G2 pass/fail label (G2 >= 10 is pass)
    Setup C: predict the most common class in the training set
    """
    
    def __init__(self, setup='A', use_g2=True):
        """
        Args:
            setup: 'A' or 'C'
            use_g2: whether to use G2 (only for Setup A)
        """
        self.setup = setup
        self.use_g2 = use_g2
        self.majority_class = None
        self.g2_data = None
        
    def fit(self, X, y, g2_data=None):
        """
        Train model
        
        Args:
            X: feature matrix
            y: target variable
            g2_data: G2 data (for Setup A)
        """
        if self.setup == 'A' and self.use_g2:
            # Setup A: use G2, no training needed
            if g2_data is None:
                raise ValueError("Setup A需要提供G2数据")
            self.g2_data = g2_data
        else:
            # Setup C: use majority class
            self.majority_class = np.bincount(y).argmax()
        
        return self
    
    def predict(self, X, g2_data=None):
        """
        Predict
        
        Args:
            X: feature matrix
            g2_data: G2 data (for Setup A)
            
        Returns:
            predictions: prediction results
        """
        if self.setup == 'A' and self.use_g2:
            # Setup A: use G2 pass/fail label
            if g2_data is None:
                raise ValueError("Setup A needs to provide G2 data")
            predictions = (g2_data >= 10).astype(int)
        else:
            # Setup C: predict majority class
            n_samples = X.shape[0]
            predictions = np.full(n_samples, self.majority_class)
        
        return predictions


class NaiveBaselineRegressor(BaseEstimator, RegressorMixin):
    """
    Naive Baseline Regressor
    
    Setup A: use G2 as prediction value
    Setup C: use average G3 value in the training set
    """
    
    def __init__(self, setup='A', use_g2=True):
        """
        Args:
            setup: 'A' or 'C'
            use_g2: whether to use G2 (only for Setup A)
        """
        self.setup = setup
        self.use_g2 = use_g2
        self.mean_value = None
        self.g2_data = None
        
    def fit(self, X, y, g2_data=None):
        """
        Train model
        
        Args:
            X: feature matrix
            y: target variable (G3)
            g2_data: G2 data (for Setup A)
        """
        if self.setup == 'A' and self.use_g2:
            # Setup A: use G2, no training needed
            if g2_data is None:
                raise ValueError("Setup A needs to provide G2 data")
            self.g2_data = g2_data
        else:
            # Setup C: use average value
            self.mean_value = np.mean(y)
        
        return self
    
    def predict(self, X, g2_data=None):
        """
        Predict
        
        Args:
            X: feature matrix
            g2_data: G2 data (for Setup A)
            
        Returns:
            predictions: prediction results
        """
        if self.setup == 'A' and self.use_g2:
            # Setup A: use G2
            if g2_data is None:
                raise ValueError("Setup A needs to provide G2 data")
            predictions = g2_data.copy()
        else:
            # Setup C: predict average value
            n_samples = X.shape[0]
            predictions = np.full(n_samples, self.mean_value)
        
        return predictions


def get_decision_tree_classifier():
    """Get decision tree classifier"""
    return DecisionTreeClassifier(random_state=42)


def get_decision_tree_regressor():
    """Get decision tree regressor"""
    return DecisionTreeRegressor(random_state=42)


def get_random_forest_classifier():
    """Get random forest classifier"""
    return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)


def get_random_forest_regressor():
    """Get random forest regressor"""
    return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

