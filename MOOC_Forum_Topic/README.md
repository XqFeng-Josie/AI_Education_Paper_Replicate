# MOOC Forum Topic Analysis - Paper Replication

Replication of: [Efficient topic identification for urgent MOOC Forum posts using BERTopic and traditional topic modeling techniques](https://link.springer.com/content/pdf/10.1007/s10639-024-13003-4.pdf)  
*Khodeir, Nabila and Elghannam, Fatma (2024)*

## 📊 Dataset
Dataset: [Stanford MOOC Forum Posts](http://infolab.stanford.edu/~paepcke/stanfordMOOCForumPostsSet.tar.gz) (29,590 posts)

## 🎯 Objective
Identify topics in urgent MOOC forum posts (urgency > 4) using BERTopic topic modeling across four groups:
- **All**: All urgent posts (~6,415)
- **Education**: Education courses
- **Humanities**: Humanities courses  
- **Medicine**: Medicine courses

## 🚀 Quick Start

### 1. Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**Note**: The code works with flexible package versions. If you encounter numpy compatibility issues with existing packages in your environment, the installation will automatically upgrade to compatible versions (numpy 2.x with scikit-learn 1.7+, pandas 2.3+).

### 2. Run Pipeline

```bash
# Option A: Run all steps
./run_pipeline.sh

# Option B: Step by step
python step1_preprocess_data.py
python step2_text_preprocessing.py
python step3_generate_embeddings.py
python step4_train_bertopic.py
python step4_visualize_grid_search.py
python step5_evaluate_visualize.py
python step6_traditional_models.py
python step7_final_comparison.py
python step8_paper_style_visualization.py  # ⭐ Paper figures
```

## 📋 Pipeline Steps

| Step | Description | Key Output |
|------|-------------|------------|
| 1 | Data preprocessing | groups_raw.pkl |
| 2 | Text cleaning | groups_preprocessed.pkl |
| 3 | Generate embeddings | embeddings.pkl |
| 4 | BERTopic + Grid search | models/, best_params.json |
| 4b | Grid search visualization | grid_search plots |
| 5 | BERTopic evaluation | bertopic_results_summary.csv |
| 6 | Train LDA/LSI/NMF | traditional_models_all_groups.pkl |
| 7 | Model comparison | model_comparison plots |
| 8 | Paper-style figures ⭐ | Fig.6, Tables 7-9, Fig.2 |

## 📁 Structure
```
├── step1-8: Pipeline scripts
├── utils_metrics.py: Shared metrics
├── dataset/: Raw data
├── models/: Trained models  
├── results/: Outputs
└── README.md
```

## 📊 Key Outputs

**Paper-Style Figures (Step 8):**
- `fig6_optimal_topics_all_group.png` - Topic count vs coherence
- `table7-9_*.csv` - Results tables for each group
- `fig2_topics_per_class.png` - Topics distribution
- `topic_word_scores.png` - Top words visualization

**Models:**
- `models/bertopic_*` - Trained BERTopic models
- `models/traditional_models_all_groups.pkl` - LDA/LSI/NMF results

**Intermediate:**
- `results/best_params.json` - Optimal n_neighbors
- `results/bertopic_results_summary.csv` - BERTopic metrics
- `results/all_models_comparison.csv` - Complete comparison

## 🔑 Key Parameters
- Embedding: `all-MiniLM-L6-v2`
- UMAP: n_components=5, metric=cosine, min_dist=0.05
- **n_neighbors**: Grid search [5,7,10,12,15,18,20,25,30] ⭐
- HDBSCAN: min_cluster_size=10
- Vectorizer: ngram_range=(1,3)

## 📝 Notes

**Key Outputs:** Focus on step8 paper-style figures (Fig.6, Tables 7-9, Fig.2)  
**Intermediate Results:** Managed by .gitignore, not tracked in git  
**Single Documentation:** Only README.md maintained

## 📖 Reference
Khodeir, N., & Elghannam, F. (2024). Efficient topic identification for urgent MOOC Forum posts using BERTopic and traditional topic modeling techniques. *Education and Information Technologies*. Springer.
