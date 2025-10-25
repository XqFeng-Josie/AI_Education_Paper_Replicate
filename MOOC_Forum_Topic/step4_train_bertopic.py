"""
Step 4: Train BERTopic Models with Grid Search for n_neighbors
This script performs grid search over different n_neighbors values to find optimal parameters
"""
import pickle
import os
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN
from umap import UMAP
from sentence_transformers import SentenceTransformer
import json

# Import unified metrics utilities
from utils_metrics import calculate_coherence_cv_bertopic, calculate_irbo

def create_umap_model(n_samples, n_neighbors):
    """Create UMAP model with specified n_neighbors"""
    # Ensure n_neighbors is valid
    n_neighbors = min(n_neighbors, n_samples - 1)
    n_neighbors = max(2, n_neighbors)  # At least 2
    
    return UMAP(
        n_components=5,
        metric='cosine',
        min_dist=0.05,
        random_state=100,
        n_neighbors=n_neighbors
    )

def create_bertopic_model(n_samples, embedding_model, n_neighbors):
    """Create BERTopic model with specified n_neighbors"""
    vectorizer_model = CountVectorizer(ngram_range=(1, 3), stop_words='english')
    umap_model = create_umap_model(n_samples, n_neighbors)
    hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', prediction_data=True)
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,
        nr_topics='auto',
        verbose=False  # Set to False for grid search
    )
    
    return topic_model

def grid_search_n_neighbors(name, texts, embs, embedding_model, n_neighbors_range):
    """Perform grid search over n_neighbors values"""
    print(f"\n{'='*60}")
    print(f"Grid Search for {name} group ({len(texts)} posts)")
    print(f"{'='*60}")
    print(f"Testing n_neighbors: {n_neighbors_range}")
    
    results = []
    
    for n_neighbors in n_neighbors_range:
        print(f"\n  Testing n_neighbors={n_neighbors}...")
        
        try:
            # Create and train model
            topic_model = create_bertopic_model(len(texts), embedding_model, n_neighbors)
            topics, probs = topic_model.fit_transform(texts, embs)
            
            # Calculate metrics
            n_topics = len(set(topics)) - (1 if -1 in topics else 0)
            n_outliers = sum(1 for t in topics if t == -1)
            
            # Only calculate coherence and IRBO if we have topics
            if n_topics > 0:
                coherence = calculate_coherence_cv_bertopic(texts, topics, topic_model, top_n=10)
                irbo = calculate_irbo(topic_model, top_n=10)
            else:
                coherence = 0.0
                irbo = 0.0
            
            result = {
                'n_neighbors': n_neighbors,
                'n_topics': n_topics,
                'n_outliers': n_outliers,
                'coherence_cv': coherence,
                'irbo': irbo
            }
            results.append(result)
            
            print(f"    Topics: {n_topics}, Outliers: {n_outliers}, "
                  f"Coherence: {coherence:.4f}, IRBO: {irbo:.4f}")
        
        except Exception as e:
            print(f"    Error with n_neighbors={n_neighbors}: {str(e)}")
            continue
    
    return results

def select_best_n_neighbors(results):
    """Select best n_neighbors based on coherence score"""
    if not results:
        print("No results found")
        exit(0)
    
    # Filter out results with 0 or very few topics
    valid_results = [r for r in results if r['n_topics'] >= 2]
    
    if not valid_results:
        # If no valid results, use the one with most topics
        print("No valid results found")
        exit(0)
    
    # Select based on highest coherence score
    best = max(valid_results, key=lambda x: x['coherence_cv'])
    return best['n_neighbors']

def main():
    print("="*80)
    print("Step 4: Train BERTopic Models with Grid Search")
    print("="*80)
    
    # Configuration
    # Grid search range for n_neighbors
    # Paper mentions 0.7 but this should be integer values
    # We'll test a range of values from 5 to 30
    N_NEIGHBORS_RANGE = [5, 7, 10, 12, 15, 18, 20, 25, 30]
    
    # Load preprocessed groups and embeddings
    print("\n[1/4] Loading data...")
    with open('data/groups_preprocessed.pkl', 'rb') as f:
        groups = pickle.load(f)
    with open('data/embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    
    # Load embedding model
    print("\n[2/4] Loading embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Grid search for each group
    print("\n[3/4] Performing grid search for all groups...")
    all_grid_results = {}
    best_params = {}
    
    for name, group_df in groups.items():
        texts = group_df['cleaned_text'].tolist()
        embs = embeddings[name]
        
        # Perform grid search
        grid_results = grid_search_n_neighbors(
            name, texts, embs, embedding_model, N_NEIGHBORS_RANGE
        )
        all_grid_results[name] = grid_results
        
        # Select best n_neighbors
        best_n = select_best_n_neighbors(grid_results)
        best_params[name] = best_n
        
        print(f"\n  ✓ Best n_neighbors for {name}: {best_n}")
        
        # Save grid search results for this group
        df = pd.DataFrame(grid_results)
        df.to_csv(f'results/grid_search_bertopic_{name.lower()}.csv', index=False)
        print(f"    Grid search results saved to: results/grid_search_bertopic_{name.lower()}.csv")
    
    # Save all grid search results
    with open('results/grid_search_bertopic_all.json', 'w') as f:
        json.dump(all_grid_results, f, indent=2)
    
    with open('results/best_params_bertopic.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print("\n" + "="*80)
    print("BEST PARAMETERS SELECTED:")
    print("="*80)
    for name, n_neighbors in best_params.items():
        print(f"  {name}: n_neighbors={n_neighbors}")
    print("="*80)
    
    # Train final models with best parameters
    print("\n[4/4] Training final models with best parameters...")
    topic_models = {}
    topics_dict = {}
    probs_dict = {}
    results = {}
    
    for name, group_df in groups.items():
        print(f"\n  Training final {name} model...")
        
        texts = group_df['cleaned_text'].tolist()
        embs = embeddings[name]
        best_n = best_params[name]
        
        # Train model with best parameters
        topic_model = create_bertopic_model(len(texts), embedding_model, best_n)
        topic_model.verbose = True  # Enable verbose for final training
        topics, probs = topic_model.fit_transform(texts, embs)
        
        # Store results
        topic_models[name] = topic_model
        topics_dict[name] = topics
        probs_dict[name] = probs
        
        # Calculate metrics
        n_topics = len(set(topics)) - (1 if -1 in topics else 0)
        n_outliers = sum(1 for t in topics if t == -1)
        coherence = calculate_coherence_cv_bertopic(texts, topics, topic_model, top_n=10)
        irbo = calculate_irbo(topic_model, top_n=10)
        
        # Store metrics
        results[name] = {
            'n_topics': n_topics,
            'n_outliers': n_outliers,
            'coherence_cv': coherence,
            'irbo': irbo,
            'n_posts': len(texts),
            'n_neighbors': best_n
        }
        
        print(f"    Topics: {n_topics}, Outliers: {n_outliers}")
        print(f"    Coherence: {coherence:.4f}, IRBO: {irbo:.4f}")
        
        # Save model
        topic_model.save(f'models/bertopic_{name.lower()}')
        print(f"    Saved: models/bertopic_{name.lower()}")
    
    # Save topics and probs
    with open('models/topics_dict.pkl', 'wb') as f:
        pickle.dump(topics_dict, f)
    with open('models/probs_dict.pkl', 'wb') as f:
        pickle.dump(probs_dict, f)
    
    # Save results summary
    results_df = pd.DataFrame(results).T
    results_df = results_df.round(4)
    results_df.to_csv('results/bertopic_results_summary.csv')
    
    print(f"\n" + "="*80)
    print("BERTOPIC FINAL RESULTS")
    print("="*80)
    print(results_df.to_string())
    print("\n" + "="*80)
    print("Paper Reference (All group): Topics ~50-57, Coherence ~0.616")
    if 'All' in results_df.index:
        print(f"Our Results (All): Topics {int(results_df.loc['All', 'n_topics'])}, "
              f"Coherence {results_df.loc['All', 'coherence_cv']:.4f}")
    print("="*80)
    
    print(f"\n✓ All models trained and saved with optimized parameters")
    print(f"  - Models: models/bertopic_*")
    print(f"  - Results: results/bertopic_results_summary.csv")

if __name__ == '__main__':
    main()

