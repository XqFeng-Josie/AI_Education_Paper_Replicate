"""
Step 5: Evaluate Models and Generate Results
"""
import pickle
import numpy as np
import pandas as pd
from bertopic import BERTopic
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from itertools import combinations

def rbo_score(list1, list2, p=0.9):
    """Calculate Rank-Biased Overlap"""
    if not list1 or not list2:
        return 0.0
    
    k = min(len(list1), len(list2))
    if k == 0:
        return 0.0
    
    overlap = 0.0
    for d in range(1, k + 1):
        set1 = set(list1[:d])
        set2 = set(list2[:d])
        overlap += (len(set1 & set2) / d) * (p ** (d - 1))
    
    rbo = ((1 - p) / p) * overlap
    return min(1.0, rbo)

def calculate_coherence_cv(texts, topics_list, topic_model, top_n=10):
    """Calculate C_v coherence score"""
    tokenized_texts = [text.split() for text in texts]
    dictionary = Dictionary(tokenized_texts)
    
    topic_words = []
    unique_topics = sorted(set(topics_list))
    if -1 in unique_topics:
        unique_topics.remove(-1)
    
    for topic_id in unique_topics:
        words = topic_model.get_topic(topic_id)
        if words:
            topic_words.append([word for word, _ in words[:top_n]])
    
    if not topic_words:
        return 0.0
    
    coherence_model = CoherenceModel(
        topics=topic_words,
        texts=tokenized_texts,
        dictionary=dictionary,
        coherence='c_v'
    )
    
    return coherence_model.get_coherence()

def calculate_irbo(topic_model, top_n=10, p=0.9):
    """Calculate IRBO for topic diversity"""
    topic_info = topic_model.get_topic_info()
    topic_ids = topic_info[topic_info['Topic'] != -1]['Topic'].tolist()
    
    if len(topic_ids) < 2:
        return 1.0
    
    topic_words_lists = []
    for topic_id in topic_ids:
        words = topic_model.get_topic(topic_id)
        if words:
            topic_words_lists.append([word for word, _ in words[:top_n]])
    
    similarities = []
    for list1, list2 in combinations(topic_words_lists, 2):
        sim = rbo_score(list1, list2, p=p)
        similarities.append(sim)
    
    if not similarities:
        return 1.0
    
    mean_similarity = np.mean(similarities)
    irbo = 1 - mean_similarity
    
    return irbo

def main():
    print("="*60)
    print("Step 5: Evaluate Models")
    print("="*60)
    
    # Load data
    print("\n[1/3] Loading data...")
    with open('data/groups_preprocessed.pkl', 'rb') as f:
        groups = pickle.load(f)
    with open('models/topics_dict.pkl', 'rb') as f:
        topics_dict = pickle.load(f)
    
    # Load models
    print("\n[2/3] Loading trained models...")
    topic_models = {}
    for name in groups.keys():
        model_path = f'models/bertopic_{name.lower()}'
        topic_models[name] = BERTopic.load(model_path)
        print(f"  Loaded: {model_path}")
    
    # Evaluate
    print("\n[3/3] Calculating metrics...")
    results = {}
    TOP_N = 10
    
    for name in groups.keys():
        print(f"\n  Evaluating {name}...")
        
        texts = groups[name]['cleaned_text'].tolist()
        topics_list = topics_dict[name]
        topic_model = topic_models[name]
        
        coherence = calculate_coherence_cv(texts, topics_list, topic_model, top_n=TOP_N)
        irbo_score_val = calculate_irbo(topic_model, top_n=TOP_N)
        n_topics = len(set(topics_list)) - (1 if -1 in topics_list else 0)
        
        results[name] = {
            'n_topics': n_topics,
            'coherence_cv': coherence,
            'irbo': irbo_score_val,
            'n_posts': len(texts)
        }
        
        print(f"    Topics: {n_topics}")
        print(f"    Coherence: {coherence:.4f}")
        print(f"    IRBO: {irbo_score_val:.4f}")
    
    # Save results
    results_df = pd.DataFrame(results).T
    results_df = results_df.round(4)
    results_df.to_csv('results_summary.csv')
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(results_df)
    print("\n" + "="*80)
    print("PAPER REFERENCE (All group):")
    print("  Topics: ~50-57")
    print("  Coherence: ~0.616")
    print("  IRBO: ~1.0")
    print("="*80)
    
    print(f"\n✓ Results saved to: results_summary.csv")

if __name__ == '__main__':
    main()

