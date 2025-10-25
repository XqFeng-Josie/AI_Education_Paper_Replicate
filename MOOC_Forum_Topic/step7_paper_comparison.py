"""
Step 7: Generate Paper-Style Visualizations and Comparison Tables
Creates visualizations matching the paper format and compares our results with paper results
"""
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from bertopic import BERTopic

def load_paper_results():
    """Load reference results from paper"""
    print("Loading paper reference results...")
    
    paper_data = {}
    group_files = {
        'All': 'paper/Table6_bertopic_group_ALL.csv',
        'Education': 'paper/Table6_bertopic_group_Education.csv',
        'Humanities': 'paper/Table6_bertopic_group_Humanities.csv',
        'Medicine': 'paper/Table6_bertopic_group_Medicine.csv'
    }
    
    for group_name, file_path in group_files.items():
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, sep='\t')
            df = df.set_index(df.columns[0])
            paper_data[group_name] = df
    
    return paper_data

def load_our_results():
    """Load our experimental results"""
    print("Loading our results...")
    
    # Load BERTopic results
    bertopic_results = pd.read_csv('results/bertopic_results_summary.csv', index_col=0)
    
    # Load traditional models results
    with open('models/traditional_models_all_groups.pkl', 'rb') as f:
        traditional_results = pickle.load(f)
    
    return bertopic_results, traditional_results

def plot_fig6_all_groups(traditional_results, bertopic_results):
    """
    Generate Fig.6 style plots for all 4 groups
    X-axis: Num Topics
    Y-axis: Coherence score
    """
    print("\nGenerating Fig.6 style plots for all groups...")
    
    groups = ['All', 'Education', 'Humanities', 'Medicine']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Colors matching the paper
    colors = {
        'LDA': '#3F51B5',    # Blue
        'LSI': '#4CAF50',    # Green  
        'NMF': '#F44336',    # Red
        'BERTopic': '#212121' # Black
    }
    
    for idx, group_name in enumerate(groups):
        ax = axes[idx]
        
        if group_name not in traditional_results:
            continue
            
        group_data = traditional_results[group_name]
        
        # Plot traditional models
        all_ks = []
        for model_name in ['LDA', 'LSI', 'NMF']:
            if model_name in group_data and 'grid_search' in group_data[model_name]:
                grid_results = group_data[model_name]['grid_search']
                ks = [r['k'] for r in grid_results]
                coherences = [r['coherence'] for r in grid_results]
                all_ks.extend(ks)
                
                marker = 'o' if model_name == 'LDA' else ('s' if model_name == 'LSI' else '^')
                ax.plot(ks, coherences, 
                       color=colors[model_name],
                       linewidth=2,
                       label=f'{model_name}_Model',
                       marker=marker,
                       markersize=5,
                       markevery=max(1, len(ks)//10))
        
        # Add BERTopic
        if group_name in bertopic_results.index:
            bertopic_topics = int(bertopic_results.loc[group_name, 'n_topics'])
            bertopic_coh = float(bertopic_results.loc[group_name, 'coherence_cv'])
            
            if all_ks:
                x_min, x_max = min(all_ks), max(all_ks)
                # BERTopic as horizontal line extending to its topic count
                x_range = range(x_min, min(bertopic_topics + 5, x_max))
                y_values = [bertopic_coh] * len(x_range)
                
                ax.plot(x_range, y_values,
                       color=colors['BERTopic'],
                       linewidth=2.5,
                       label='BERTopic_Model',
                       linestyle='-')
                
                ax.set_xlim(x_min - 2, x_max + 2)
        
        ax.set_xlabel('Num Topics', fontsize=12)
        ax.set_ylabel('Coherence score', fontsize=12)
        ax.set_title(f'Group {chr(65+idx)}: {group_name}', fontsize=13, fontweight='bold')
        ax.legend(frameon=True, loc='best', fontsize=9)
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('results/fig6_optimal_topics_all_groups.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/fig6_optimal_topics_all_groups.png")
    plt.close()

def create_comparison_tables(paper_data, traditional_results, bertopic_results):
    """
    Create comparison tables between paper and our results
    """
    print("\nGenerating comparison tables...")
    
    os.makedirs('results/comparison_tables', exist_ok=True)
    
    all_comparisons = []
    
    for group_name in ['All', 'Education', 'Humanities', 'Medicine']:
        print(f"\n  Processing {group_name} group...")
        
        # Get paper results
        if group_name not in paper_data:
            print(f"    Warning: No paper data for {group_name}")
            continue
        
        paper_df = paper_data[group_name]
        
        # Prepare comparison data
        comparison_data = []
        
        for model_name in ['LDA', 'LSI', 'NMF', 'BERTopic']:
            row = {'Group': group_name, 'Model': model_name}
            
            # Paper values
            if model_name in paper_df.columns:
                row['Paper_Topics'] = paper_df.loc['Number_of_topics', model_name]
                row['Paper_Coherence'] = paper_df.loc['Coherence_score', model_name]
                row['Paper_IRBO'] = paper_df.loc['IRBO', model_name]
            else:
                row['Paper_Topics'] = '-'
                row['Paper_Coherence'] = '-'
                row['Paper_IRBO'] = '-'
            
            # Our values
            if model_name == 'BERTopic':
                if group_name in bertopic_results.index:
                    row['Our_Topics'] = int(bertopic_results.loc[group_name, 'n_topics'])
                    row['Our_Coherence'] = float(bertopic_results.loc[group_name, 'coherence_cv'])
                    row['Our_IRBO'] = float(bertopic_results.loc[group_name, 'irbo'])
                else:
                    row['Our_Topics'] = '-'
                    row['Our_Coherence'] = '-'
                    row['Our_IRBO'] = '-'
            else:
                if group_name in traditional_results and model_name in traditional_results[group_name]:
                    data = traditional_results[group_name][model_name]
                    row['Our_Topics'] = data['n_topics']
                    row['Our_Coherence'] = round(data['coherence_cv'], 3)
                    row['Our_IRBO'] = '-'  # Traditional models don't have IRBO in our implementation
                else:
                    row['Our_Topics'] = '-'
                    row['Our_Coherence'] = '-'
                    row['Our_IRBO'] = '-'
            
            # Calculate differences
            try:
                if row['Paper_Topics'] != '-' and row['Our_Topics'] != '-':
                    row['Diff_Topics'] = int(row['Our_Topics']) - int(row['Paper_Topics'])
                else:
                    row['Diff_Topics'] = '-'
                
                if row['Paper_Coherence'] != '-' and row['Our_Coherence'] != '-':
                    diff = float(row['Our_Coherence']) - float(row['Paper_Coherence'])
                    row['Diff_Coherence'] = f"{diff:+.3f}"
                    row['Diff_Coherence_Pct'] = f"{(diff/float(row['Paper_Coherence'])*100):+.1f}%"
                else:
                    row['Diff_Coherence'] = '-'
                    row['Diff_Coherence_Pct'] = '-'
                
                if row['Paper_IRBO'] != '-' and row['Our_IRBO'] != '-':
                    diff = float(row['Our_IRBO']) - float(row['Paper_IRBO'])
                    row['Diff_IRBO'] = f"{diff:+.3f}"
                else:
                    row['Diff_IRBO'] = '-'
            except:
                row['Diff_Topics'] = '-'
                row['Diff_Coherence'] = '-'
                row['Diff_Coherence_Pct'] = '-'
                row['Diff_IRBO'] = '-'
            
            comparison_data.append(row)
            all_comparisons.append(row)
        
        # Save individual group table
        df_comparison = pd.DataFrame(comparison_data)
        csv_path = f'results/comparison_tables/table_comparison_{group_name.lower()}.csv'
        df_comparison.to_csv(csv_path, index=False)
        print(f"    ✓ Saved: {csv_path}")
    
    # Save combined table
    df_all = pd.DataFrame(all_comparisons)
    df_all.to_csv('results/comparison_tables/table_comparison_all.csv', index=False)
    print("\n  ✓ Saved: results/comparison_tables/table_comparison_all.csv")
    
    return df_all

def print_comparison_summary(df_all):
    """Print a formatted summary of comparisons"""
    print("\n" + "="*100)
    print("COMPARISON SUMMARY: Our Results vs Paper Results")
    print("="*100)
    
    for group_name in ['All', 'Education', 'Humanities', 'Medicine']:
        group_data = df_all[df_all['Group'] == group_name]
        
        if len(group_data) == 0:
            continue
        
        print(f"\n{'─'*100}")
        print(f"Group: {group_name}")
        print(f"{'─'*100}")
        print(f"{'Model':<12} | {'Paper':<20} | {'Our Results':<20} | {'Difference':<30}")
        print(f"{'─'*100}")
        
        for _, row in group_data.iterrows():
            model = row['Model']
            
            # Format paper results
            paper_str = f"T:{row['Paper_Topics']:<3} C:{row['Paper_Coherence']}"
            
            # Format our results
            our_str = f"T:{row['Our_Topics']:<3} C:{row['Our_Coherence']}"
            
            # Format difference
            diff_str = f"ΔT:{row['Diff_Topics']:<3} ΔC:{row['Diff_Coherence']:<8} ({row['Diff_Coherence_Pct']})"
            
            print(f"{model:<12} | {paper_str:<20} | {our_str:<20} | {diff_str:<30}")
    
    print("="*100)
    print("\nLegend: T=Topics, C=Coherence, Δ=Difference")
    print("="*100)

def generate_comparison_heatmap(df_all):
    """Generate a heatmap showing coherence differences"""
    print("\nGenerating comparison heatmap...")
    
    # Prepare data for heatmap
    pivot_data = []
    for group in ['All', 'Education', 'Humanities', 'Medicine']:
        group_data = df_all[df_all['Group'] == group]
        row = {}
        for _, r in group_data.iterrows():
            if r['Diff_Coherence'] != '-':
                try:
                    row[r['Model']] = float(r['Diff_Coherence'])
                except:
                    row[r['Model']] = 0
        pivot_data.append(row)
    
    df_pivot = pd.DataFrame(pivot_data, index=['All', 'Education', 'Humanities', 'Medicine'])
    
    # Create heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Coherence Difference (Our - Paper)'},
                linewidths=1, linecolor='white')
    plt.title('Coherence Score Differences: Our Results vs Paper Results', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Group', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/comparison_tables/coherence_difference_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/comparison_tables/coherence_difference_heatmap.png")
    plt.close()

def main():
    print("="*100)
    print("Step 7: Generate Paper-Style Visualizations and Comparison Tables")
    print("="*100)
    
    # Load data
    print("\n[1/5] Loading data...")
    paper_data = load_paper_results()
    bertopic_results, traditional_results = load_our_results()
    
    print(f"  ✓ Loaded paper results for {len(paper_data)} groups")
    print(f"  ✓ Loaded our results for {len(traditional_results)} groups")
    
    # Generate Fig.6 style plots
    print("\n[2/5] Generating Fig.6 style plots...")
    plot_fig6_all_groups(traditional_results, bertopic_results)
    
    # Create comparison tables
    print("\n[3/5] Creating comparison tables...")
    df_all = create_comparison_tables(paper_data, traditional_results, bertopic_results)
    
    # Generate heatmap
    print("\n[4/5] Generating comparison heatmap...")
    generate_comparison_heatmap(df_all)
    
    # Print summary
    print("\n[5/5] Printing comparison summary...")
    print_comparison_summary(df_all)
    
    print("\n" + "="*100)
    print("✓ All visualizations and comparisons complete!")
    print("="*100)
    print("\nGenerated files:")
    print("  📊 results/fig6_optimal_topics_all_groups.png")
    print("  📋 results/comparison_tables/table_comparison_*.csv")
    print("  🔥 results/comparison_tables/coherence_difference_heatmap.png")
    print("="*100)

if __name__ == '__main__':
    main()

