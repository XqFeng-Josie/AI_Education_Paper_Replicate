"""
Evaluation module
Implement 20×10-fold cross-validation and result evaluation
"""
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, mean_squared_error
from tqdm import tqdm


def cross_validate(model, X, y, setup='A', task='classification', 
                   n_runs=20, n_folds=10, random_state=42, g2_data=None):
    """
    Perform 20×10-fold cross-validation
    
    Args:
        model: model object (needs to implement fit and predict methods)
        X: feature matrix
        y: target variable
        setup: 'A' or 'C'
        task: 'classification' or 'regression'
        n_runs: number of repetitions (default 20)
        n_folds: number of folds (default 10)
        random_state: random seed
        g2_data: G2 data (for Naive Baseline in Setup A)
        
    Returns:
        results: list of results for all folds
        mean_score: mean score
        std_score: standard deviation
    """
    results = []
    
    print(f"\nStarting cross-validation: {n_runs} runs × {n_folds} folds")
    print(f"Task: {task}, Setup: Setup {setup}")
    
    for run in tqdm(range(n_runs), desc="Number of runs"):
        # each run uses a different random seed
        run_seed = random_state + run
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=run_seed)
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            # split training and test sets
            if hasattr(X, 'iloc'):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            else:
                X_train, X_test = X[train_idx], X[test_idx]
            
            if hasattr(y, 'iloc'):
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            else:
                y_train, y_test = y[train_idx], y[test_idx]
            
            # prepare G2 data (if needed)
            g2_train = None
            g2_test = None
            if g2_data is not None:
                if hasattr(g2_data, 'iloc'):
                    g2_train = g2_data.iloc[train_idx]
                    g2_test = g2_data.iloc[test_idx]
                else:
                    g2_train = g2_data[train_idx]
                    g2_test = g2_data[test_idx]
            
            # train model
            if hasattr(model, 'fit'):
                if g2_train is not None:
                    model.fit(X_train, y_train, g2_data=g2_train)
                else:
                    model.fit(X_train, y_train)
            
            # predict
            if g2_test is not None:
                y_pred = model.predict(X_test, g2_data=g2_test)
            else:
                y_pred = model.predict(X_test)
            
            # evaluate
            if task == 'classification':
                score = accuracy_score(y_test, y_pred)
            else:  # regression
                score = np.sqrt(mean_squared_error(y_test, y_pred))  # RMSE
            
            results.append(score)
    
    results = np.array(results)
    mean_score = np.mean(results)
    std_score = np.std(results)
    
    return results, mean_score, std_score


def evaluate_model(model, X, y, setup='A', task='classification',
                   n_runs=20, n_folds=10, random_state=42, g2_data=None,
                   model_name='Model'):
    """
    Evaluate model and return results
    
    Args:
        model: model object
        X: feature matrix
        y: target variable
        setup: 'A' or 'C'
        task: 'classification' or 'regression'
        n_runs: number of repetitions
        n_folds: number of folds
        random_state: random seed
        g2_data: G2 data
        model_name: model name
        
    Returns:
        dict: dictionary containing results
    """
    results, mean_score, std_score = cross_validate(
        model, X, y, setup=setup, task=task,
        n_runs=n_runs, n_folds=n_folds, random_state=random_state, g2_data=g2_data
    )
    
    result_dict = {
        'model_name': model_name,
        'setup': setup,
        'task': task,
        'mean_score': mean_score,
        'std_score': std_score,
        'all_results': results,
        'n_runs': n_runs,
        'n_folds': n_folds
    }
    
    # print results
    if task == 'classification':
        print(f"\n{model_name} (Setup {setup}) - Classification accuracy:")
    else:
        print(f"\n{model_name} (Setup {setup}) - Regression RMSE:")
    
    print(f"  Mean: {mean_score:.4f} ± {std_score:.4f}")
    
    return result_dict

