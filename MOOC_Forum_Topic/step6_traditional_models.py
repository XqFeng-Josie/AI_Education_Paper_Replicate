"""
Step 6: Train Traditional Topic Models (LDA, LSI, NMF) for All Groups
This prepares traditional models data for final comparison in Step 7
"""
import pickle
import pandas as pd
import os
import json
from gensim import corpora
from gensim.models import LdaModel, LsiModel
from sklearn.decomposition import NMF as SklearnNMF
from sklearn.feature_extraction.text import TfidfVectorizer

# Import unified metrics utilities
from utils_metrics import calculate_coherence_cv_traditional

def train_lda(texts, n_topics=10):
    """Train LDA model"""
    tokenized_texts = [text.split() for text in texts]
    dictionary = corpora.Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_topics,
        random_state=100,
        passes=10,
        per_word_topics=True
    )
    
    # Extract topic words
    topic_words = []
    for topic_id in range(n_topics):
        words = [word for word, _ in lda_model.show_topic(topic_id, topn=10)]
        topic_words.append(words)
    
    return lda_model, topic_words

def train_lsi(texts, n_topics=10):
    """Train LSI model"""
    tokenized_texts = [text.split() for text in texts]
    dictionary = corpora.Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    
    lsi_model = LsiModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_topics
    )
    
    # Extract topic words
    topic_words = []
    for topic_id in range(n_topics):
        words = [word for word, _ in lsi_model.show_topic(topic_id, topn=10)]
        topic_words.append(words)
    
    return lsi_model, topic_words

def train_nmf(texts, n_topics=10, verbose=False):
    """Train NMF model with detailed logging"""
    if verbose:
        print(f"    [NMF] Creating TF-IDF vectorizer (max_features=1000)...")
    
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    
    if verbose:
        print(f"    [NMF] Transforming {len(texts)} documents to TF-IDF...")
    tfidf = vectorizer.fit_transform(texts)
    
    if verbose:
        print(f"    [NMF] TF-IDF matrix shape: {tfidf.shape}")
        print(f"    [NMF] Training NMF model with {n_topics} topics...")
    
    nmf_model = SklearnNMF(
        n_components=n_topics,
        random_state=100,
        max_iter=200,
        init='nndsvda',  # Better initialization
        solver='cd',  # Coordinate Descent solver
        verbose=1 if verbose else 0
    )
    
    nmf_model.fit(tfidf)
    
    if verbose:
        print(f"    [NMF] Training completed. Reconstruction error: {nmf_model.reconstruction_err_:.4f}")
    
    # Extract topic words
    feature_names = vectorizer.get_feature_names_out()
    topic_words = []
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_indices = topic.argsort()[-10:][::-1]
        words = [feature_names[i] for i in top_indices]
        topic_words.append(words)
        if verbose and topic_idx < 3:  # Show first 3 topics
            print(f"    [NMF] Topic {topic_idx}: {', '.join(words[:5])}")
    
    return nmf_model, topic_words

def find_best_k(texts, model_type='lda', k_range=range(3, 21), verbose=False):
    """Grid search for optimal number of topics"""
    print(f"  Grid search for {model_type.upper()} (k={min(k_range)}-{max(k_range)})...")
    best_k = None
    best_coherence = -1
    results = []
    
    total_k = len(k_range)
    for idx, k in enumerate(k_range, 1):
        if verbose:
            print(f"    [{idx}/{total_k}] Testing k={k}...")
        
        try:
            if model_type == 'lda':
                model, topic_words = train_lda(texts, n_topics=k)
            elif model_type == 'lsi':
                model, topic_words = train_lsi(texts, n_topics=k)
            elif model_type == 'nmf':
                model, topic_words = train_nmf(texts, n_topics=k, verbose=verbose)
            
            if verbose:
                print(f"    [{idx}/{total_k}] Calculating coherence...")
            
            coherence = calculate_coherence_cv_traditional(texts, topic_words)
            results.append({'k': k, 'coherence': coherence})
            
            if verbose:
                print(f"    [{idx}/{total_k}] k={k}, coherence={coherence:.4f}")
            
            if coherence > best_coherence:
                best_coherence = coherence
                best_k = k
        except Exception as e:
            print(f"    [{idx}/{total_k}] Error with k={k}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            continue
    
    print(f"    Best: k={best_k}, coherence={best_coherence:.4f}")
    return best_k, results

# Note: Plotting functions have been moved to step7_final_comparison.py

def main():
    """
    Main function to train traditional models
    Supports: LDA, LSI, NMF
    Usage: Can be configured via MODEL_LIST variable or command line
    """
    import sys
    
    # Parse command line arguments or use default
    if len(sys.argv) > 1:
        model_list = [m.strip().upper() for m in sys.argv[1].split(',')]
    else:
        model_list = ['LDA', 'LSI', 'NMF']  # Default: train all models
    
    print("="*80)
    print("Step 6: Train Traditional Topic Models")
    print("="*80)
    print(f"Models to train: {', '.join(model_list)}")
    
    # Load preprocessed data
    print("\n[1/4] Loading preprocessed data...")
    with open('data/groups_preprocessed.pkl', 'rb') as f:
        groups = pickle.load(f)
    
    # Create directories if not exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Process all groups
    print("\n[2/4] Training traditional models for all groups...")
    all_results = {}  # Store all results
    best_params = {}  # Store best k for each group-model combination
    
    for group_name, group_df in groups.items():
        print(f"\n{'='*80}")
        print(f"Processing {group_name} Group ({len(group_df)} posts)")
        print(f"{'='*80}")
        
        texts = group_df['cleaned_text'].tolist()
        group_results = {}
        
        # Determine k_range based on group size
        max_k = min(30, len(texts) // 50)  # Reasonable upper limit
        max_k = max(10, max_k)  # At least 10
        k_range = range(3, max_k + 1)
        
        # Train traditional models
        for model_type in model_list:
            model_name = model_type.upper()
            print(f"\n{model_name} Model:")
            
            # Enable verbose logging for NMF
            verbose = True
            
            try:
                # Find best k
                best_k, search_results = find_best_k(
                    texts, 
                    model_type=model_name.lower(), 
                    k_range=k_range, 
                    verbose=verbose
                )
                
                # Train final model with best k
                if model_name == 'LDA':
                    model, topic_words = train_lda(texts, n_topics=best_k)
                elif model_name == 'LSI':
                    model, topic_words = train_lsi(texts, n_topics=best_k)
                elif model_name == 'NMF':
                    model, topic_words = train_nmf(texts, n_topics=best_k)
                
                coherence = calculate_coherence_cv_traditional(texts, topic_words)
                
                # Store results
                group_results[model_name] = {
                    'n_topics': best_k,
                    'coherence_cv': coherence,
                    'n_posts': len(texts)
                }
                
                # Store best params
                best_params[f"{group_name}_{model_name}"] = best_k
                
                # Save model
                model_path = f'models/traditional_{model_name.lower()}_{group_name.lower()}.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump({'model': model, 'topic_words': topic_words}, f)
                
                # Save grid search results
                grid_df = pd.DataFrame(search_results)
                grid_df.to_csv(f'results/grid_search_{model_name.lower()}_{group_name.lower()}.csv', index=False)
                
                print(f"  ✓ Best k={best_k}, coherence={coherence:.4f}")
                print(f"    Saved: {model_path}")
            
            except Exception as e:
                print(f"  ✗ Error training {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        all_results[group_name] = group_results
    
    # Save all results
    print("\n[3/4] Saving results...")
    
    # Save best parameters
    with open('results/traditional_best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    print("  ✓ Saved: results/traditional_best_params.json")
    
    # Save results summary (similar to bertopic_results_summary.csv)
    summary_data = []
    for group_name, models in all_results.items():
        for model_name, metrics in models.items():
            summary_data.append({
                'group': group_name,
                'model': model_name,
                'n_topics': metrics['n_topics'],
                'coherence_cv': metrics['coherence_cv'],
                'n_posts': metrics['n_posts']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.round(4)
    summary_df.to_csv('results/traditional_results_summary.csv', index=False)
    print("  ✓ Saved: results/traditional_results_summary.csv")
    
    # Save detailed results (for step7 comparison)
    with open('models/traditional_models_all_groups.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    print("  ✓ Saved: models/traditional_models_all_groups.pkl")
    
    # Print summary
    print("\n[4/4] Training Summary")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)
    
    print(f"\n✓ Traditional models trained successfully!")
    print(f"  - Models: models/traditional_*")
    print(f"  - Results: results/traditional_results_summary.csv")
    print(f"  - Best params: results/traditional_best_params.json")

if __name__ == '__main__':
    main()