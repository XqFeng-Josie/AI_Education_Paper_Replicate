"""
Step 5: Evaluate and Visualize BERTopic Models
Evaluates trained models and generates visualizations
"""
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from bertopic import BERTopic
from utils_metrics import calculate_coherence_cv_bertopic, calculate_irbo

def load_best_params():
    """Load optimal n_neighbors from grid search"""
    best_params_path = 'results/best_params.json'
    if os.path.exists(best_params_path):
        with open(best_params_path, 'r') as f:
            return json.load(f)
    else:
        print("⚠️  Warning: best_params.json not found")
        exit(0)

def evaluate_models(groups, topics_dict, topic_models, best_params):
    """Evaluate all models"""
    print("\nCalculating metrics for all groups...")
    results = {}
    TOP_N = 10
    
    for name in groups.keys():
        print(f"  Evaluating {name}...")
        
        texts = groups[name]['cleaned_text'].tolist()
        topics_list = topics_dict[name]
        topic_model = topic_models[name]
        
        coherence = calculate_coherence_cv_bertopic(texts, topics_list, topic_model, top_n=TOP_N)
        irbo_score_val = calculate_irbo(topic_model, top_n=TOP_N)
        n_topics = len(set(topics_list)) - (1 if -1 in topics_list else 0)
        n_outliers = sum(1 for t in topics_list if t == -1)
        
        results[name] = {
            'n_topics': n_topics,
            'n_outliers': n_outliers,
            'coherence_cv': coherence,
            'irbo': irbo_score_val,
            'n_posts': len(texts),
            'n_neighbors': best_params.get(name, 15)
        }
        
        print(f"    Topics: {n_topics}, Coherence: {coherence:.4f}, IRBO: {irbo_score_val:.4f}")
    
    return results

def save_results(results):
    """Save evaluation results"""
    results_df = pd.DataFrame(results).T
    results_df = results_df.round(4)
    results_df.to_csv('results/bertopic_results_summary.csv')
    return results_df

def print_summary(results_df):
    """Print evaluation summary"""
    print("\n" + "="*80)
    print("BERTOPIC EVALUATION RESULTS")
    print("="*80)
    print(results_df.to_string())
    print("\n" + "="*80)
    print("Paper Reference (All group): Topics ~50-57, Coherence ~0.616")
    if 'All' in results_df.index:
        print(f"Our Results (All): Topics {int(results_df.loc['All', 'n_topics'])}, "
              f"Coherence {results_df.loc['All', 'coherence_cv']:.4f}")
    print("="*80)

def main():
    print("="*80)
    print("Step 5: Evaluate BERTopic Models")
    print("="*80)
    
    # Load best parameters
    print("\n[1/4] Loading optimal parameters...")
    best_params = load_best_params()
    
    # Load data
    print("\n[2/4] Loading data and models...")
    with open('data/groups_preprocessed.pkl', 'rb') as f:
        groups = pickle.load(f)
    with open('models/topics_dict.pkl', 'rb') as f:
        topics_dict = pickle.load(f)
    
    topic_models = {}
    for name in groups.keys():
        topic_models[name] = BERTopic.load(f'models/bertopic_{name.lower()}')
    
    # Evaluate
    print("\n[3/4] Evaluating models...")
    results = evaluate_models(groups, topics_dict, topic_models, best_params)
    
    # Save and print
    print("\n[4/4] Saving results...")
    results_df = save_results(results)
    print_summary(results_df)
    
    print("\n✓ Evaluation complete!")
    print("  - results/bertopic_results_summary.csv")

if __name__ == '__main__':
    main()

