"""
Step 4: Train BERTopic Models
"""
import pickle
import os
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN
from umap import UMAP
from sentence_transformers import SentenceTransformer

def create_umap_model(n_samples):
    n_neighbors = min(15, n_samples - 1)
    return UMAP(
        n_components=5,
        metric='cosine',
        min_dist=0.05,
        random_state=100,
        n_neighbors=n_neighbors
    )

def create_bertopic_model(n_samples, embedding_model):
    vectorizer_model = CountVectorizer(ngram_range=(1, 3), stop_words='english')
    umap_model = create_umap_model(n_samples)
    hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', prediction_data=True)
    
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,
        nr_topics='auto',
        verbose=True
    )
    
    return topic_model

def main():
    print("="*60)
    print("Step 4: Train BERTopic Models")
    print("="*60)
    
    # Load preprocessed groups and embeddings
    print("\n[1/3] Loading data...")
    with open('data/groups_preprocessed.pkl', 'rb') as f:
        groups = pickle.load(f)
    with open('data/embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    
    # Load embedding model
    print("\n[2/3] Loading embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Train models
    print("\n[3/3] Training BERTopic models...")
    topic_models = {}
    topics_dict = {}
    probs_dict = {}
    
    os.makedirs('models', exist_ok=True)
    
    for name, group_df in groups.items():
        print(f"\n{'='*60}")
        print(f"Training {name} group ({len(group_df)} posts)")
        print(f"{'='*60}")
        
        texts = group_df['cleaned_text'].tolist()
        embs = embeddings[name]
        
        # Train model
        topic_model = create_bertopic_model(len(texts), embedding_model)
        topics, probs = topic_model.fit_transform(texts, embs)
        
        # Store results
        topic_models[name] = topic_model
        topics_dict[name] = topics
        probs_dict[name] = probs
        
        # Print info
        n_topics = len(set(topics)) - (1 if -1 in topics else 0)
        print(f"\n{name} Results:")
        print(f"  Topics: {n_topics}")
        print(f"  Outliers: {sum(1 for t in topics if t == -1)}")
        
        # Save model
        topic_model.save(f'models/bertopic_{name.lower()}')
        print(f"  Saved: models/bertopic_{name.lower()}")
    
    # Save topics and probs
    with open('models/topics_dict.pkl', 'wb') as f:
        pickle.dump(topics_dict, f)
    with open('models/probs_dict.pkl', 'wb') as f:
        pickle.dump(probs_dict, f)
    
    print(f"\n✓ All models trained and saved")

if __name__ == '__main__':
    main()

