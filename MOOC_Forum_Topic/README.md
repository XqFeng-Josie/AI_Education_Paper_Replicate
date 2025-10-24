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

### 2. Run Experiment
```bash
jupyter notebook experiment.ipynb
```

Run cells sequentially from top to bottom.

## 📋 Pipeline Overview

### Step 1-2: Data Loading & Grouping
- Load `stanfordMOOCForumPostsSet.xlsx`
- Filter posts with urgency > 4
- Create 4 analysis groups

### Step 3: Preprocessing
- Expand contractions
- Remove URLs, numbers, special characters
- Lowercase & POS tagging (keep NOUN, ADJ, VERB, ADV)
- Remove stopwords (NLTK 3.8.1)

### Step 4: Embedding
- Model: `all-MiniLM-L6-v2` (Sentence-BERT)
- Generate 384-dim vectors for each post

### Step 5: Dimensionality Reduction
- UMAP: n_components=5, metric=cosine, min_dist=0.05, random_state=100

### Step 6-7: BERTopic Training
- Vectorizer: CountVectorizer with ngram_range=(1,3)
- Clustering: HDBSCAN (min_cluster_size=10)
- Topic representation: c-TF-IDF
- Auto-discover topics (nr_topics='auto')

### Step 8: Evaluation
- **Coherence (C_v)**: Semantic consistency of topics (via Gensim)
- **IRBO**: Topic diversity (1 - mean RBO similarity)
  - Custom RBO implementation (original `rbo` package incompatible with numpy 2.x)
- Top N words: 10

### Step 9: Results Comparison
Compare with paper's benchmark:
- **All group**: ~50-57 topics, C_v ≈ 0.616, IRBO ≈ 1.0

## 📁 Project Structure
```
MOOC_Forum_Topic/
├── dataset/
│   └── stanfordMOOCForumPostsSet.xlsx
├── paper/
│   └── s10639-024-13003-4.pdf
├── models/                          # Generated models (after running)
├── experiment.ipynb                 # Main experiment notebook
├── requirements.txt                 # Dependencies
├── results_summary.csv              # Results table (generated)
├── results_comparison.png           # Visualization (generated)
└── README.md
```

## 📊 Expected Outputs
- `results_summary.csv`: Metrics table for all groups
- `results_comparison.png`: Bar charts comparing metrics
- `topic_visualization_all.html`: Interactive topic map
- `models/`: Saved BERTopic models and processed data

## 🔑 Key Parameters
| Component | Parameter | Value |
|-----------|-----------|-------|
| Embedding | Model | all-MiniLM-L6-v2 |
| UMAP | n_components | 5 |
| UMAP | metric | cosine |
| UMAP | min_dist | 0.05 |
| UMAP | random_state | 100 |
| HDBSCAN | min_cluster_size | 10 |
| Vectorizer | ngram_range | (1, 3) |
| Evaluation | top_n_words | 10 |

## 📖 Reference
```bibtex
@article{khodeir2024efficient,
  title={Efficient topic identification for urgent MOOC Forum posts using BERTopic and traditional topic modeling techniques},
  author={Khodeir, Nabila and Elghannam, Fatma},
  journal={Education and Information Technologies},
  year={2024},
  publisher={Springer}
}
```
