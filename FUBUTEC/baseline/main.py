"""
Paper replication experiment main script
Run all paper replication experiments
"""
import pandas as pd
import numpy as np
import json
from datetime import datetime

# support running as script directly
try:
    from .data_preprocessing import DataPreprocessor
    from .models import (
        NaiveBaselineClassifier, NaiveBaselineRegressor,
        get_decision_tree_classifier, get_decision_tree_regressor,
        get_random_forest_classifier, get_random_forest_regressor
    )
    from .evaluation import evaluate_model
except ImportError:
    from data_preprocessing import DataPreprocessor
    from models import (
        NaiveBaselineClassifier, NaiveBaselineRegressor,
        get_decision_tree_classifier, get_decision_tree_regressor,
        get_random_forest_classifier, get_random_forest_regressor
    )
    from evaluation import evaluate_model


def run_replication_experiments():
    """Run complete paper replication experiment"""
    
    print("=" * 80)
    print("FUBUTEC 2008 Paper Replication Experiment")
    print("=" * 80)
    
    # initialize data preprocessor
    preprocessor = DataPreprocessor('../data/student-por.csv')
    preprocessor.load_data()
    
    # store all results
    all_results = []
    
    # run experiment for each setup
    for setup in ['A', 'C']:
        print(f"\n{'=' * 80}")
        print(f"Setup {setup} Experiment")
        print(f"{'=' * 80}")
        
        # get data
        X, y_binary, y_regression, feature_names = preprocessor.get_setup_data(setup)
        
        # get G2 data (for Naive Baseline in Setup A)
        g2_data = None
        if setup == 'A':
            g2_data = preprocessor.data['G2']
        
        # 1. Naive Baseline Model
        print(f"\n{'-' * 80}")
        print("1. Naive Baseline Model (Naive Baseline)")
        print(f"{'-' * 80}")
        
        # classification task
        nv_clf = NaiveBaselineClassifier(setup=setup, use_g2=(setup == 'A'))
        result_nv_clf = evaluate_model(
            nv_clf, X, y_binary, setup=setup, task='classification',
            g2_data=g2_data, model_name='NV (Classification)'
        )
        all_results.append(result_nv_clf)
        
        # regression task
        nv_reg = NaiveBaselineRegressor(setup=setup, use_g2=(setup == 'A'))
        result_nv_reg = evaluate_model(
            nv_reg, X, y_regression, setup=setup, task='regression',
            g2_data=g2_data, model_name='NV (Regression)'
        )
        all_results.append(result_nv_reg)
        
        # 2. Decision Tree Model
        print(f"\n{'-' * 80}")
        print("2. Decision Tree Model (Decision Tree)")
        print(f"{'-' * 80}")
        
        # classification task
        dt_clf = get_decision_tree_classifier()
        result_dt_clf = evaluate_model(
            dt_clf, X, y_binary, setup=setup, task='classification',
            model_name='DT (Classification)'
        )
        all_results.append(result_dt_clf)
        
        # regression task
        dt_reg = get_decision_tree_regressor()
        result_dt_reg = evaluate_model(
            dt_reg, X, y_regression, setup=setup, task='regression',
            model_name='DT (Regression)'
        )
        all_results.append(result_dt_reg)
        
        # 3. Random Forest Model
        print(f"\n{'-' * 80}")
        print("3. Random Forest Model (Random Forest)")
        print(f"{'-' * 80}")
        
        # classification task
        rf_clf = get_random_forest_classifier()
        result_rf_clf = evaluate_model(
            rf_clf, X, y_binary, setup=setup, task='classification',
            model_name='RF (Classification)'
        )
        all_results.append(result_rf_clf)
        
        # regression task
        rf_reg = get_random_forest_regressor()
        result_rf_reg = evaluate_model(
            rf_reg, X, y_regression, setup=setup, task='regression',
            model_name='RF (Regression)'
        )
        all_results.append(result_rf_reg)
    
    # summarize results
    print(f"\n{'=' * 80}")
    print("Experiment Results Summary")
    print(f"{'=' * 80}")
    
    # create results table
    results_summary = []
    for result in all_results:
        results_summary.append({
            'Model': result['model_name'],
            'Setup': f"Setup {result['setup']}",
            'Task': result['task'],
            'Mean': f"{result['mean_score']:.4f}",
            'Standard Deviation': f"{result['std_score']:.4f}"
        })
    
    results_df = pd.DataFrame(results_summary)
    print("\nResults Table:")
    print(results_df.to_string(index=False))
    
    # compare with paper results
    print(f"\n{'=' * 80}")
    print("Compare with paper results")
    print(f"{'=' * 80}")
    
    # paper results (from instruction.md)
    paper_results = {
        ('NV', 'A', 'classification'): 89.7,
        ('DT', 'A', 'classification'): 93.0,
        ('RF', 'A', 'classification'): 92.6,
        ('NV', 'C', 'classification'): 84.6,
        ('DT', 'C', 'classification'): 84.4,
        ('RF', 'C', 'classification'): 85.0,
        ('NV', 'A', 'regression'): 1.32,
        ('DT', 'A', 'regression'): 1.46,
        ('RF', 'A', 'regression'): 1.32,
        ('NV', 'C', 'regression'): 3.23,
        ('DT', 'C', 'regression'): 2.93,
        ('RF', 'C', 'regression'): 2.67,
    }
    
    print("\nComparison Table:")
    print(f"{'Model':<10} {'Setup':<8} {'Task':<15} {'Paper Results':<12} {'Replication Results':<12} {'Difference':<10}")
    print("-" * 70)
    
    for result in all_results:
        model_key = result['model_name'].split()[0]  # extract model name (NV, DT, RF)
        setup = result['setup']
        task = result['task']
        
        key = (model_key, setup, task)
        if key in paper_results:
            paper_value = paper_results[key]
            our_value = result['mean_score']
            if task == 'classification':
                # classification result is percentage
                our_value_pct = our_value * 100
                diff = our_value_pct - paper_value
                print(f"{model_key:<10} Setup {setup:<4} {task:<15} {paper_value:<12.2f} {our_value_pct:<12.2f} {diff:+.2f}")
            else:
                # regression result is RMSE
                diff = our_value - paper_value
                print(f"{model_key:<10} Setup {setup:<4} {task:<15} {paper_value:<12.2f} {our_value:<12.2f} {diff:+.2f}")
    
    # save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"../results/results_replication_{timestamp}.json"
    
    # convert numpy array to list for JSON serialization
    results_for_json = []
    for result in all_results:
        result_copy = result.copy()
        result_copy['all_results'] = result_copy['all_results'].tolist()
        results_for_json.append(result_copy)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_for_json, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {results_file}")
    
    # save summary table
    summary_file = f"../results/results_summary_{timestamp}.csv"
    results_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f"Summary table saved to: {summary_file}")
    
    return all_results, results_df


if __name__ == '__main__':
    run_replication_experiments()

