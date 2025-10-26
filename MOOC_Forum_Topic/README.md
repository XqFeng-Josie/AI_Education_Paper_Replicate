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

### 2. Run Pipeline

```bash
# Option A: Run all steps
./run_pipeline.sh

# Option B: Step by step
python step1_preprocess_data.py
python step2_text_preprocessing.py
python step3_generate_embeddings.py
python step4_train_bertopic.py            # BERTopic: Grid search + train + eval
python step4_visualize_grid_search.py
python step6_traditional_models.py        # Traditional: LDA,LSI,NMF (default: all)
python step7_paper_comparison.py          # ⭐ Generate figures & compare with paper

# Step 6 with specific models
python step6_traditional_models.py "LDA,LSI"  # Train only LDA and LSI
```

## 📊 Experiment Results

### Comparison with Paper Results

*Optimal parameters found via 2D grid search (n_neighbors × min_cluster_size)

#### All Group Comparison

| Model | Paper |  | Our Results |  | Difference |
|-------|-------|-------|-------------|-------|------------|
| | Topics | Coherence | Topics | Coherence | ΔCoherence |
| **LDA** | 6 | 0.542 | 11 | 0.535 | -0.007 (-1.3%) |
| **LSI** | 8 | 0.459 | 7 | 0.463 | +0.004 (+0.9%) |
| **NMF** | 6 | 0.660 | - | - | - |
| **BERTopic** | 50 | 0.616 | 65 | **0.681** | **+0.065 (+10.5%)** ✅ |

**BERTopic Optimal Parameters (All)**: `n_neighbors=15, min_cluster_size=5`

#### Education Group Comparison

| Model | Paper |  | Our Results |  | Difference |
|-------|-------|-------|-------------|-------|------------|
| | Topics | Coherence | Topics | Coherence | ΔCoherence |
| **LDA** | 10 | 0.363 | 6 | 0.465 | **+0.102 (+28.1%)** ✅ |
| **LSI** | 4 | 0.517 | 3 | 0.557 | +0.040 (+7.7%) ✅ |
| **NMF** | 4 | 0.620 | - | - | - |
| **BERTopic** | 10 | 0.638 | 6 | **0.634** | **-0.004 (-0.6%)** ≈ |

**BERTopic Optimal Parameters (Education)**: `n_neighbors=12, min_cluster_size=5`

#### Humanities Group Comparison

| Model | Paper |  | Our Results |  | Difference |
|-------|-------|-------|-------------|-------|------------|
| | Topics | Coherence | Topics | Coherence | ΔCoherence |
| **LDA** | 18 | 0.455 | 5 | 0.564 | **+0.109 (+24.0%)** ✅ |
| **LSI** | 6 | 0.455 | 8 | 0.444 | -0.011 (-2.4%) |
| **NMF** | 8 | 0.658 | - | - | - |
| **BERTopic** | 17 | 0.689 | 65 | 0.641 | -0.048 (-6.9%) |

**BERTopic Optimal Parameters (Humanities)**: `n_neighbors=18, min_cluster_size=5`

> ⚠️ Note: Humanities group discovered more fine-grained topics (65) but with slightly lower coherence. Alternative configuration with fewer topics available.

#### Medicine Group Comparison

| Model | Paper |  | Our Results |  | Difference |
|-------|-------|-------|-------------|-------|------------|
| | Topics | Coherence | Topics | Coherence | ΔCoherence |
| **LDA** | 2 | 0.517 | 4 | 0.599 | **+0.082 (+15.9%)** ✅ |
| **LSI** | 6 | 0.499 | 3 | 0.405 | -0.094 (-18.8%) |
| **NMF** | 6 | 0.755 | - | - | - |
| **BERTopic** | 37 | 0.604 | 60 | **0.687** | **+0.083 (+13.7%)** ✅ |

**BERTopic Optimal Parameters (Medicine)**: `n_neighbors=30, min_cluster_size=5`

---

### 🎯 Summary

**BERTopic Performance:**
- ✅ **All Group**: +10.5% coherence improvement, more topics discovered
- ≈ **Education Group**: -0.6% coherence (essentially matched paper)
- ⚠️ **Humanities Group**: -6.9% coherence (discovered more topics)
- ✅ **Medicine Group**: +13.7% coherence improvement, more topics discovered

**Key Finding**: Smaller `min_cluster_size` (2-5) significantly improves BERTopic performance compared to default settings, allowing discovery of more fine-grained topics while maintaining or improving coherence.

---

## 📈 BERTopic Grid Search Analysis

### Grid Search Strategy

We performed a comprehensive 2D grid search over two key HDBSCAN parameters:
- **n_neighbors** (UMAP): `[5, 7, 10, 12, 15, 18, 20, 25, 30]` (9 values)
- **min_cluster_size** (HDBSCAN): `[2, 3, 4, 5, 10, 15, 20]` (7 values)
- **Total combinations**: 63 per group × 4 groups = **252 training runs**

Each configuration was evaluated on:
- Number of topics discovered
- Coherence score (C_v)
- Topic diversity (IRBO)
- Number of outlier documents

### Grid Search Results by Group

#### All Group (6,416 posts)
<img src= results/grid_search_bertopic_all_detailed.png width=80% />

#### Education Group (461 posts)
<img src= results/grid_search_bertopic_education_detailed.png width=80% />


#### Humanities Group (2,476 posts)
<img src= results/grid_search_bertopic_humanities_detailed.png width=80% />

#### Medicine Group (3,478 posts)

<img src= results/grid_search_bertopic_medicine_detailed.png width=80% />


### Other UMAP/HDBSCAN Parameters

Based on the paper:
- **UMAP**: `n_components=5`, `metric='cosine'`, `min_dist=0.05`
- **HDBSCAN**: `metric='euclidean'`, `prediction_data=True`
- **Vectorizer**: `CountVectorizer(ngram_range=(1,3))`

---

  
## 📖 Reference
Khodeir, N., & Elghannam, F. (2024). Efficient topic identification for urgent MOOC Forum posts using BERTopic and traditional topic modeling techniques. *Education and Information Technologies*. Springer.
