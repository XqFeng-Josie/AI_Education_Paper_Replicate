"""
data_preprocessing.py
data loading and preprocessing, build feature set (Setup A and Setup C), create target variable
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """Data preprocessor"""
    
    def __init__(self, data_path='data/student-por.csv'):
        """
        Initialize data preprocessor
        
        Args:
            data_path: data file path
        """
        self.data_path = data_path
        self.data = None
        self.label_encoders = {}
        
    def load_data(self):
        """Load data"""
        print(f"Loading data: {self.data_path}")
        self.data = pd.read_csv(self.data_path, sep=';')
        print(f"Data shape: {self.data.shape}")
        print(f"Column names: {list(self.data.columns)}")
        return self.data
    
    def create_binary_target(self, threshold=10):
        """
        Create binary target variable (pass/fail)
        
        Args:
            threshold: pass threshold, default 10
            
        Returns:
            binary_target: binary label (1=pass, 0=fail)
        """
        if self.data is None:
            raise ValueError("Please load data first")
        
        binary_target = (self.data['G3'] >= threshold).astype(int)
        print(f"\nBinary target statistics:")
        print(f"Pass (1): {binary_target.sum()} ({binary_target.mean()*100:.2f}%)")
        print(f"Fail (0): {(1-binary_target).sum()} ({(1-binary_target.mean())*100:.2f}%)")
        
        return binary_target
    
    def get_regression_target(self):
        """
        Get regression target variable (G3)
        
        Returns:
            G3: final grade
        """
        if self.data is None:
            raise ValueError("Please load data first")
        
        regression_target = self.data['G3'].copy()
        print(f"\nRegression target statistics:")
        print(f"Mean: {regression_target.mean():.2f}")
        print(f"Standard deviation: {regression_target.std():.2f}")
        print(f"Range: [{regression_target.min()}, {regression_target.max()}]")
        
        return regression_target
    
    def prepare_features_setup_a(self):
        """
        Prepare Setup A features (include G1 and G2, exclude G3)
        
        Returns:
            X: feature matrix
            feature_names: feature name list
        """
        if self.data is None:
            raise ValueError("Please load data first")
        
        # exclude target variable G3
        X = self.data.drop(columns=['G3']).copy()
        
        # encode categorical variables
        X_encoded = self._encode_categorical_features(X)
        
        print(f"\nSetup A features:")
        print(f"Number of features: {X_encoded.shape[1]}")
        print(f"Include G1 and G2: {'G1' in X.columns and 'G2' in X.columns}")
        
        return X_encoded, list(X.columns)
    
    def prepare_features_setup_c(self):
        """
        Prepare Setup C features (exclude G1, G2 and G3)
        
        Returns:
            X: feature matrix
            feature_names: feature name list
        """
        if self.data is None:
            raise ValueError("Please load data first")
        
        # exclude all grade variables
        X = self.data.drop(columns=['G1', 'G2', 'G3']).copy()
        
        # encode categorical variables
        X_encoded = self._encode_categorical_features(X)
        
        print(f"\nSetup C features:")
        print(f"Number of features: {X_encoded.shape[1]}")
        print(f"Exclude G1, G2 and G3: {'G1' not in X.columns and 'G2' not in X.columns}")
        
        return X_encoded, list(X.columns)
    
    def _encode_categorical_features(self, X):
        """
        Encode categorical features
        
        Args:
            X: original feature DataFrame
            
        Returns:
            X_encoded: encoded feature matrix
        """
        X_encoded = X.copy()
        
        # identify categorical columns (non-numeric columns)
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X_encoded[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                # if already fitted, directly convert
                # handle possible new classes (in test set)
                unique_values = set(X[col].astype(str).unique())
                known_classes = set(self.label_encoders[col].classes_)
                if unique_values.issubset(known_classes):
                    X_encoded[col] = self.label_encoders[col].transform(X[col].astype(str))
                else:
                    # if there are new classes, need to refit (should not happen in cross-validation)
                    self.label_encoders[col] = LabelEncoder()
                    X_encoded[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
        
        return X_encoded
    
    def get_setup_data(self, setup='A'):
        """
        Get features and targets for specified setup
        
        Args:
            setup: 'A' 或 'C'
            
        Returns:
            X: feature matrix
            y_binary: binary target
            y_regression: regression target
            feature_names: feature name list
        """
        if setup == 'A':
            X, feature_names = self.prepare_features_setup_a()
        elif setup == 'C':
            X, feature_names = self.prepare_features_setup_c()
        else:
            raise ValueError(f"Unknown setup: {setup}, please use 'A' or 'C'")
        
        y_binary = self.create_binary_target()
        y_regression = self.get_regression_target()
        
        return X, y_binary, y_regression, feature_names

