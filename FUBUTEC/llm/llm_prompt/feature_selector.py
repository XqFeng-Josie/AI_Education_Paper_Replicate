"""
Feature selection based on importance from traditional ML models (RF/XGBoost)
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import logging

# Optional xgboost import
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    xgb = None

logger = logging.getLogger(__name__)


class FeatureSelector:
    """Feature selector based on model importance"""
    
    def __init__(
        self,
        model_type: str = 'rf',
        n_features: int = 10,
        random_state: int = 42
    ):
        """
        Initialize feature selector
        
        Args:
            model_type: 'rf' for Random Forest, 'xgb' for XGBoost
            n_features: Number of top features to select (top N)
            random_state: Random seed
        """
        self.model_type = model_type.lower()
        self.n_features = n_features
        self.random_state = random_state
        self.selected_features = None
        self.feature_importance = None
        self.model = None
        
        if self.model_type not in ['rf', 'xgb']:
            raise ValueError(f"model_type must be 'rf' or 'xgb', got {model_type}")
        
        if self.model_type == 'xgb' and not HAS_XGBOOST:
            raise ImportError(
                "XGBoost is not installed. Please install it with: pip install xgboost\n"
                "Or use 'rf' (Random Forest) instead."
            )
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task: str = 'classification'
    ) -> List[str]:
        """
        Fit model and select top N important features
        
        Args:
            X: Feature DataFrame (encoded)
            y: Target Series
            task: 'classification' or 'regression'
            
        Returns:
            List of selected feature names (top N)
        """
        print(f"\n{'=' * 80}")
        print("Feature importance analysis")
        print(f"{'=' * 80}")
        print(f"Model type: {self.model_type.upper()}")
        print(f"Task type: {task}")
        print(f"Selecting top {self.n_features} important features")
        print(f"Total feature count: {len(X.columns)}")
        
        # Train model
        if task == 'classification':
            if self.model_type == 'rf':
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            else:  # xgb
                self.model = xgb.XGBClassifier(
                    n_estimators=100,
                    random_state=self.random_state,
                    n_jobs=-1,
                    eval_metric='logloss'
                )
        else:  # regression
            if self.model_type == 'rf':
                self.model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            else:  # xgb
                self.model = xgb.XGBRegressor(
                    n_estimators=100,
                    random_state=self.random_state,
                    n_jobs=-1
                )
        
        # Fit model
        print(f"\nTraining {self.model_type.upper()} model...")
        self.model.fit(X, y)
        
        # Get feature importance
        if self.model_type == 'rf':
            importances = self.model.feature_importances_
        else:  # xgb
            importances = self.model.feature_importances_
        
        # Create importance dictionary
        feature_names = list(X.columns)
        self.feature_importance = dict(zip(feature_names, importances))
        
        # Sort by importance
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Select top N
        self.selected_features = [feat[0] for feat in sorted_features[:self.n_features]]
        
        # Print results
        print(f"\nTop {self.n_features} important features:")
        print("-" * 80)
        for i, (feat_name, importance) in enumerate(sorted_features, 1):
            print(f"{i:2d}. {feat_name:20s} : {importance:.6f}")
        
        print(f"\n{'=' * 80}\n")
        
        return self.selected_features
    
    def get_selected_features(self) -> List[str]:
        """Get list of selected feature names"""
        if self.selected_features is None:
            raise ValueError("Feature selector not fitted yet. Call fit() first.")
        return self.selected_features.copy()
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance dictionary"""
        if self.feature_importance is None:
            raise ValueError("Feature selector not fitted yet. Call fit() first.")
        return self.feature_importance.copy()
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Select features from DataFrame
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with only selected features
        """
        if self.selected_features is None:
            raise ValueError("Feature selector not fitted yet. Call fit() first.")
        
        # Check if all selected features exist
        missing_features = set(self.selected_features) - set(X.columns)
        if missing_features:
            logger.warning(f"Missing features in X: {missing_features}")
            # Only select features that exist
            available_features = [f for f in self.selected_features if f in X.columns]
            return X[available_features]
        
        return X[self.selected_features]
    
    def filter_student_dict(self, student: Dict) -> Dict:
        """
        Filter student dictionary to only include selected features
        
        Args:
            student: Student data dictionary
            
        Returns:
            Filtered student dictionary with only selected features
        """
        if self.selected_features is None:
            raise ValueError("Feature selector not fitted yet. Call fit() first.")
        
        filtered = {k: v for k, v in student.items() if k in self.selected_features}
        return filtered

