"""
Step 7: Final Comparison of All Models
Generate comprehensive comparison plots for LDA, LSI, NMF, and BERTopic
"""
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def load_all_results():
    """Load results from all models"""
    print("Loading results from all models...")
    
    # Load BERTopic results
    bertopic_results = pd.read_csv('results/bertopic_results_summary.csv', index_col=0)
    
    # Load traditional models results
    with open('models/traditional_models_all_groups.pkl', 'rb') as f:
        traditional_results = pickle.load(f)
    
    return bertopic_results, traditional_results

def prepare_comparison_data(bertopic_results, traditional_results):
    """Prepare unified comparison data"""
    all_data = {}
    
    for group_name in traditional_results.keys():
        all_data[group_name] = {}
        
        # Add traditional models
        for model_name in ['LDA', 'LSI', 'NMF']:
            if model_name in traditional_results[group_name]:
                all_data[group_name][model_name] = traditional_results[group_name][model_name]
        
        # Add BERTopic
        if group_name in bertopic_results.index:
            all_data[group_name]['BERTopic'] = {
                'n_topics': int(bertopic_results.loc[group_name, 'n_topics']),
                'coherence_cv': float(bertopic_results.loc[group_name, 'coherence_cv']),
                'irbo': float(bertopic_results.loc[group_name, 'irbo'])
            }
    
    return all_data

def plot_main_comparison(all_data):
    """
    Main comparison plot: 4 subplots for 4 groups
    X-axis: Number of Topics
    Y-axis: Coherence Score
    Shows: LDA, LSI, NMF curves + BERTopic point
    """
    sns.set_style("whitegrid")
    
    groups = list(all_data.keys())
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    
    colors = {'LDA': '#1f77b4', 'LSI': '#ff7f0e', 'NMF': '#2ca02c', 'BERTopic': '#d62728'}
    markers = {'LDA': 'o', 'LSI': 's', 'NMF': '^', 'BERTopic': '*'}
    
    for idx, group_name in enumerate(groups):
        ax = axes[idx]
        group_data = all_data[group_name]
        
        # Plot grid search curves for traditional models
        for model_name in ['LDA', 'LSI', 'NMF']:
            if model_name in group_data and 'grid_search' in group_data[model_name]:
                grid_results = group_data[model_name]['grid_search']
                ks = [r['k'] for r in grid_results]
                coherences = [r['coherence'] for r in grid_results]
                
                ax.plot(ks, coherences, 
                       marker=markers[model_name], 
                       color=colors[model_name],
                       linewidth=2, markersize=8,
                       label=f"{model_name}", alpha=0.7)
                
                # Mark best point
                best_k = group_data[model_name]['n_topics']
                best_coh = group_data[model_name]['coherence_cv']
                ax.scatter([best_k], [best_coh], 
                          s=200, marker=markers[model_name],
                          color=colors[model_name], 
                          edgecolors='black', linewidth=2,
                          zorder=5)
        
        # Add BERTopic result as a single point
        if 'BERTopic' in group_data:
            bertopic_topics = group_data['BERTopic']['n_topics']
            bertopic_coh = group_data['BERTopic']['coherence_cv']
            ax.scatter([bertopic_topics], [bertopic_coh],
                      s=400, marker=markers['BERTopic'],
                      color=colors['BERTopic'],
                      edgecolors='black', linewidth=3,
                      label='BERTopic', zorder=10)
            
            # Annotate BERTopic
            ax.annotate(f'BERTopic\n({bertopic_topics}, {bertopic_coh:.3f})',
                       xy=(bertopic_topics, bertopic_coh),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.set_xlabel('Number of Topics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Coherence Score (C_v)', fontsize=12, fontweight='bold')
        ax.set_title(f'{group_name} Group', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/model_comparison_all_groups.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/model_comparison_all_groups.png")
    plt.close()

def plot_summary_bars(all_data):
    """Bar charts comparing best results for each model"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    model_names = ['LDA', 'LSI', 'NMF', 'BERTopic']
    groups = list(all_data.keys())
    x = np.arange(len(groups))
    width = 0.2
    
    colors = {'LDA': '#1f77b4', 'LSI': '#ff7f0e', 'NMF': '#2ca02c', 'BERTopic': '#d62728'}
    
    # Best n_topics for each model and group
    ax = axes[0]
    for i, model_name in enumerate(model_names):
        topics = []
        for group_name in groups:
            if model_name in all_data[group_name]:
                topics.append(all_data[group_name][model_name]['n_topics'])
            else:
                topics.append(0)
        
        ax.bar(x + i*width, topics, width, 
               label=model_name, color=colors[model_name],
               edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Groups', fontsize=12, fontweight='bold')
    ax.set_ylabel('Optimal Number of Topics', fontsize=12, fontweight='bold')
    ax.set_title('Optimal Number of Topics by Model and Group', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(groups)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Best coherence for each model and group
    ax = axes[1]
    for i, model_name in enumerate(model_names):
        coherences = []
        for group_name in groups:
            if model_name in all_data[group_name]:
                coherences.append(all_data[group_name][model_name]['coherence_cv'])
            else:
                coherences.append(0)
        
        ax.bar(x + i*width, coherences, width, 
               label=model_name, color=colors[model_name],
               edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Groups', fontsize=12, fontweight='bold')
    ax.set_ylabel('Best Coherence Score', fontsize=12, fontweight='bold')
    ax.set_title('Best Coherence Score by Model and Group', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(groups)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/model_comparison_summary.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/model_comparison_summary.png")
    plt.close()

def save_comparison_csv(all_data):
    """Save comprehensive comparison table"""
    summary_data = []
    for group_name, group_results in all_data.items():
        for model_name, model_results in group_results.items():
            summary_data.append({
                'Group': group_name,
                'Model': model_name,
                'Topics': model_results['n_topics'],
                'Coherence': model_results['coherence_cv']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('results/all_models_comparison.csv', index=False)
    print("✓ Saved: results/all_models_comparison.csv")
    
    return summary_df

def print_comparison_table(summary_df):
    """Print formatted comparison table"""
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*80)
    
    for group_name in summary_df['Group'].unique():
        print(f"\n{group_name} Group:")
        print("-" * 80)
        group_df = summary_df[summary_df['Group'] == group_name]
        print(group_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("PAPER REFERENCE (All group):")
    print("="*80)
    print("  NMF: k=6, coherence=~0.66")
    print("  BERTopic: k=~57, coherence=~0.616")
    print("\nOur Results (All group):")
    all_group = summary_df[summary_df['Group'] == 'All']
    for _, row in all_group.iterrows():
        print(f"  {row['Model']}: k={int(row['Topics'])}, coherence={row['Coherence']:.4f}")
    print("="*80)

def main():
    print("="*80)
    print("Step 7: Final Comparison of All Models")
    print("="*80)
    
    # Load results
    print("\n[1/5] Loading all model results...")
    bertopic_results, traditional_results = load_all_results()
    
    # Prepare comparison data
    print("\n[2/5] Preparing comparison data...")
    all_data = prepare_comparison_data(bertopic_results, traditional_results)
    
    # Generate main comparison plot
    print("\n[3/5] Generating main comparison plot...")
    plot_main_comparison(all_data)
    
    # Generate summary bar charts
    print("\n[4/5] Generating summary bar charts...")
    plot_summary_bars(all_data)
    
    # Save comparison CSV and print table
    print("\n[5/5] Saving comparison table...")
    summary_df = save_comparison_csv(all_data)
    print_comparison_table(summary_df)
    
    print("\n" + "="*80)
    print("✓ FINAL COMPARISON COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - results/all_models_comparison.csv")
    print("  - results/model_comparison_all_groups.png ⭐ (main plot)")
    print("  - results/model_comparison_summary.png")
    print("\nMain plot shows:")
    print("  - X-axis: Number of Topics")
    print("  - Y-axis: Coherence Score")
    print("  - Models: LDA, LSI, NMF (curves) + BERTopic (star point)")
    print("  - All 4 groups in one figure")
    print("="*80)

if __name__ == '__main__':
    main()

