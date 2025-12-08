"""
Data validation module
Validate quality of generated synthetic data
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class DataValidator:
    """Validate quality of synthetic data"""
    
    def __init__(self, original_data: pd.DataFrame):
        """
        Initialize validator
        
        Args:
            original_data: Original data (for comparison)
        """
        self.original_data = original_data
    
    def validate(self, synthetic_data: pd.DataFrame) -> Dict:
        """
        Validate synthetic data
        
        Args:
            synthetic_data: Synthetic data
            
        Returns:
            Validation results dictionary
        """
        results = {
            'basic_stats': self._check_basic_stats(synthetic_data),
            'value_ranges': self._check_value_ranges(synthetic_data),
            'categorical_distributions': self._check_categorical_distributions(synthetic_data),
            'numeric_distributions': self._check_numeric_distributions(synthetic_data),
            'correlations': self._check_correlations(synthetic_data),
            'missing_values': self._check_missing_values(synthetic_data)
        }
        
        # Calculate overall quality score
        results['quality_score'] = self._calculate_quality_score(results)
        
        return results
    
    def _check_basic_stats(self, synthetic_data: pd.DataFrame) -> Dict:
        """Check basic statistics"""
        return {
            'original_count': len(self.original_data),
            'synthetic_count': len(synthetic_data),
            'original_columns': len(self.original_data.columns),
            'synthetic_columns': len(synthetic_data.columns),
            'columns_match': set(self.original_data.columns) == set(synthetic_data.columns)
        }
    
    def _check_value_ranges(self, synthetic_data: pd.DataFrame) -> Dict:
        """Check value ranges"""
        issues = []
        
        # Check categorical variables
        categorical_cols = self.original_data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in synthetic_data.columns:
                original_values = set(self.original_data[col].unique())
                synthetic_values = set(synthetic_data[col].unique())
                invalid_values = synthetic_values - original_values
                if invalid_values:
                    issues.append(f"{col}: Invalid values {invalid_values}")
        
        # Check numeric variable ranges
        numeric_cols = self.original_data.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if col in synthetic_data.columns:
                orig_min = self.original_data[col].min()
                orig_max = self.original_data[col].max()
                synth_min = synthetic_data[col].min()
                synth_max = synthetic_data[col].max()
                
                if synth_min < orig_min or synth_max > orig_max:
                    issues.append(f"{col}: Range exceeded [{orig_min}, {orig_max}]")
        
        return {
            'issues': issues,
            'is_valid': len(issues) == 0
        }
    
    def _check_categorical_distributions(self, synthetic_data: pd.DataFrame) -> Dict:
        """Check categorical variable distributions"""
        comparisons = {}
        categorical_cols = self.original_data.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col in synthetic_data.columns:
                orig_dist = self.original_data[col].value_counts(normalize=True).sort_index()
                synth_dist = synthetic_data[col].value_counts(normalize=True).sort_index()
                
                # Calculate KL divergence (simplified version)
                all_values = set(orig_dist.index) | set(synth_dist.index)
                kl_div = 0
                for val in all_values:
                    p = orig_dist.get(val, 0.001)
                    q = synth_dist.get(val, 0.001)
                    kl_div += p * np.log(p / q) if p > 0 else 0
                
                comparisons[col] = {
                    'kl_divergence': kl_div,
                    'original_dist': orig_dist.to_dict(),
                    'synthetic_dist': synth_dist.to_dict()
                }
        
        return comparisons
    
    def _check_numeric_distributions(self, synthetic_data: pd.DataFrame) -> Dict:
        """Check numeric variable distributions"""
        comparisons = {}
        numeric_cols = self.original_data.select_dtypes(include=['int64', 'float64']).columns
        
        for col in numeric_cols:
            if col in synthetic_data.columns:
                orig_mean = self.original_data[col].mean()
                orig_std = self.original_data[col].std()
                synth_mean = synthetic_data[col].mean()
                synth_std = synthetic_data[col].std()
                
                mean_diff = abs(orig_mean - synth_mean) / (orig_std + 1e-6)
                std_diff = abs(orig_std - synth_std) / (orig_std + 1e-6)
                
                comparisons[col] = {
                    'original_mean': orig_mean,
                    'original_std': orig_std,
                    'synthetic_mean': synth_mean,
                    'synthetic_std': synth_std,
                    'mean_diff_ratio': mean_diff,
                    'std_diff_ratio': std_diff
                }
        
        return comparisons
    
    def _check_correlations(self, synthetic_data: pd.DataFrame) -> Dict:
        """Check correlations between variables"""
        numeric_cols = self.original_data.select_dtypes(include=['int64', 'float64']).columns
        
        # Check key correlations (e.g., between grades)
        key_pairs = [
            ('G1', 'G2'),
            ('G2', 'G3'),
            ('G1', 'G3'),
            ('studytime', 'G3'),
            ('failures', 'G3')
        ]
        
        correlations = {}
        for col1, col2 in key_pairs:
            if col1 in numeric_cols and col2 in numeric_cols:
                if col1 in synthetic_data.columns and col2 in synthetic_data.columns:
                    orig_corr = self.original_data[col1].corr(self.original_data[col2])
                    synth_corr = synthetic_data[col1].corr(synthetic_data[col2])
                    
                    correlations[f"{col1}-{col2}"] = {
                        'original': orig_corr,
                        'synthetic': synth_corr,
                        'difference': abs(orig_corr - synth_corr)
                    }
        
        return correlations
    
    def _check_missing_values(self, synthetic_data: pd.DataFrame) -> Dict:
        """Check missing values"""
        orig_missing = self.original_data.isnull().sum().sum()
        synth_missing = synthetic_data.isnull().sum().sum()
        
        return {
            'original_missing': orig_missing,
            'synthetic_missing': synth_missing,
            'has_missing': synth_missing > 0
        }
    
    def _calculate_quality_score(self, results: Dict) -> float:
        """Calculate overall quality score (0-1)"""
        score = 1.0
        
        # Basic statistics (20%)
        if not results['basic_stats']['columns_match']:
            score -= 0.2
        
        # Value ranges (30%)
        if not results['value_ranges']['is_valid']:
            score -= 0.3
        
        # Categorical distributions (20%)
        cat_scores = []
        for col, comp in results['categorical_distributions'].items():
            kl = comp['kl_divergence']
            # Smaller KL divergence is better, convert to 0-1 score
            cat_score = max(0, 1 - min(kl, 2) / 2)
            cat_scores.append(cat_score)
        if cat_scores:
            avg_cat_score = np.mean(cat_scores)
            score = score * 0.8 + avg_cat_score * 0.2
        
        # Numeric distributions (20%)
        num_scores = []
        for col, comp in results['numeric_distributions'].items():
            mean_diff = comp['mean_diff_ratio']
            std_diff = comp['std_diff_ratio']
            # Smaller difference is better
            num_score = max(0, 1 - min(mean_diff + std_diff, 2) / 2)
            num_scores.append(num_score)
        if num_scores:
            avg_num_score = np.mean(num_scores)
            score = score * 0.8 + avg_num_score * 0.2
        
        # Missing values (10%)
        if results['missing_values']['has_missing']:
            score -= 0.1
        
        return max(0, min(1, score))
    
    def print_validation_report(self, results: Dict):
        """Print validation report"""
        print("\n" + "="*60)
        print("Data Validation Report")
        print("="*60)
        
        # Basic statistics
        print("\n[Basic Statistics]")
        stats = results['basic_stats']
        print(f"Original data: {stats['original_count']} records")
        print(f"Synthetic data: {stats['synthetic_count']} records")
        print(f"Column match: {'✓' if stats['columns_match'] else '✗'}")
        
        # Value ranges
        print("\n[Value Range Check]")
        if results['value_ranges']['is_valid']:
            print("✓ All values are within valid range")
        else:
            print("✗ Found the following issues:")
            for issue in results['value_ranges']['issues']:
                print(f"  - {issue}")
        
        # Categorical distributions
        print("\n[Categorical Variable Distributions]")
        for col, comp in list(results['categorical_distributions'].items())[:5]:
            kl = comp['kl_divergence']
            print(f"{col}: KL divergence = {kl:.4f} {'✓' if kl < 0.5 else '⚠'}")
        
        # Numeric distributions
        print("\n[Numeric Variable Distributions]")
        for col, comp in list(results['numeric_distributions'].items())[:5]:
            mean_diff = comp['mean_diff_ratio']
            print(f"{col}: Mean difference = {mean_diff:.2f} {'✓' if mean_diff < 0.5 else '⚠'}")
        
        # Correlations
        print("\n[Key Correlations]")
        for pair, corr in list(results['correlations'].items())[:3]:
            diff = corr['difference']
            print(f"{pair}: Correlation difference = {diff:.3f} {'✓' if diff < 0.2 else '⚠'}")
        
        # Quality score
        print("\n[Overall Quality Score]")
        score = results['quality_score']
        print(f"Quality score: {score:.2%}")
        if score >= 0.8:
            print("✓ Data quality is excellent")
        elif score >= 0.6:
            print("⚠ Data quality is good but can be improved")
        else:
            print("✗ Data quality needs improvement")
        
        print("="*60 + "\n")

