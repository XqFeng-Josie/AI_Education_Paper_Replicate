"""
Unified Metrics Calculation Utilities
Consolidates all metric calculation functions to avoid code duplication
"""
import numpy as np
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from itertools import combinations


def rbo_score(list1, list2, p=0.9):
    """
    Calculate Rank-Biased Overlap (RBO) between two ranked lists
    
    Args:
        list1: First ranked list
        list2: Second ranked list
        p: Weight parameter (default: 0.9)
        
    Returns:
        float: RBO score between 0 and 1
    """
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


def calculate_coherence_cv_bertopic(texts, topics_list, topic_model, top_n=10):
    """
    Calculate C_v coherence score for BERTopic models
    
    Args:
        texts: List of text documents
        topics_list: List of topic assignments
        topic_model: Trained BERTopic model
        top_n: Number of top words to use (default: 10)
        
    Returns:
        float: Coherence score
    """
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
    
    try:
        coherence_model = CoherenceModel(
            topics=topic_words,
            texts=tokenized_texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        return coherence_model.get_coherence()
    except Exception as e:
        print(f"Warning: Could not calculate coherence: {e}")
        return 0.0


def calculate_coherence_cv_traditional(texts, topic_words, top_n=10):
    """
    Calculate C_v coherence score for traditional models (LDA, LSI, NMF)
    
    Args:
        texts: List of text documents
        topic_words: List of lists of topic words
        top_n: Number of top words to use (default: 10)
        
    Returns:
        float: Coherence score
    """
    tokenized_texts = [text.split() for text in texts]
    dictionary = Dictionary(tokenized_texts)
    
    # Ensure topic_words are lists of words
    topic_words_filtered = []
    for words in topic_words:
        if words:
            topic_words_filtered.append(words[:top_n])
    
    if not topic_words_filtered:
        return 0.0
    
    try:
        coherence_model = CoherenceModel(
            topics=topic_words_filtered,
            texts=tokenized_texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        return coherence_model.get_coherence()
    except Exception as e:
        print(f"Warning: Could not calculate coherence: {e}")
        return 0.0


def calculate_irbo(topic_model, top_n=10, p=0.9):
    """
    Calculate IRBO (Inverted Rank-Biased Overlap) for topic diversity
    
    Args:
        topic_model: Trained BERTopic model
        top_n: Number of top words to use (default: 10)
        p: RBO weight parameter (default: 0.9)
        
    Returns:
        float: IRBO score (1 - mean RBO similarity)
    """
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

