"""
Step 6: Visualize Results
"""
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from bertopic import BERTopic

def main():
    print("="*60)
    print("Step 6: Visualize Results")
    print("="*60)
    
    # Load results
    print("\n[1/3] Loading results...")
    results_df = pd.read_csv('results_summary.csv', index_col=0)
    
    with open('data/groups_preprocessed.pkl', 'rb') as f:
        groups = pickle.load(f)
    
    # Load models
    print("\n[2/3] Loading models...")
    topic_models = {}
    for name in groups.keys():
        model_path = f'models/bertopic_{name.lower()}'
        topic_models[name] = BERTopic.load(model_path)
    
    # Create comparison plot
    print("\n[3/3] Creating visualizations...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Number of topics
    axes[0].bar(results_df.index, results_df['n_topics'], color='skyblue')
    axes[0].set_title('Number of Topics by Group')
    axes[0].set_ylabel('Number of Topics')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Coherence
    axes[1].bar(results_df.index, results_df['coherence_cv'], color='lightcoral')
    axes[1].axhline(y=0.616, color='red', linestyle='--', label='Paper: 0.616')
    axes[1].set_title('Topic Coherence (C_v)')
    axes[1].set_ylabel('Coherence Score')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend()
    
    # IRBO
    axes[2].bar(results_df.index, results_df['irbo'], color='lightgreen')
    axes[2].axhline(y=1.0, color='green', linestyle='--', label='Paper: 1.0')
    axes[2].set_title('Topic Diversity (IRBO)')
    axes[2].set_ylabel('IRBO Score')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('results_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: results_comparison.png")
    
    # Print top topics
    print("\nTop 10 Topics per Group:")
    for name in groups.keys():
        print(f"\n{name.upper()}:")
        topic_info = topic_models[name].get_topic_info()
        topic_info_filtered = topic_info[topic_info['Topic'] != -1].head(10)
        
        for _, row in topic_info_filtered.iterrows():
            topic_id = row['Topic']
            count = row['Count']
            words = topic_models[name].get_topic(topic_id)
            top_words = [word for word, _ in words[:5]]
            print(f"  Topic {topic_id} ({count} posts): {', '.join(top_words)}")
    
    # Interactive visualization for All group
    print("\nGenerating interactive visualization for All group...")
    try:
        fig = topic_models['All'].visualize_topics()
        fig.write_html('topic_visualization_all.html')
        print("  Saved: topic_visualization_all.html")
    except Exception as e:
        print(f"  Warning: Could not generate interactive plot: {e}")
    
    print(f"\n✓ Visualization complete")

if __name__ == '__main__':
    main()

