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

*Optimal parameters found via grid search

#### All Group Comparison

| Model | Paper |  | Our Results |  | Difference |
|-------|-------|-------|-------------|-------|------------|
| | Topics | Coherence | Topics | Coherence | ΔCoherence |
| **LDA** | 6 | 0.542 | 11 | 0.535 | -0.007 (-1.3%) |
| **LSI** | 8 | 0.459 | 7 | 0.463 | +0.004 (+0.9%) |
| **NMF** | 6 | 0.660 | - | - | - |
| **BERTopic** | 50 | 0.616 | 25 | **0.660** | **+0.044 (+7.2%)** ✅ |

#### Education Group Comparison

| Model | Paper |  | Our Results |  | Difference |
|-------|-------|-------|-------------|-------|------------|
| | Topics | Coherence | Topics | Coherence | ΔCoherence |
| **LDA** | 10 | 0.363 | 6 | 0.465 | **+0.102 (+28.1%)** ✅ |
| **LSI** | 4 | 0.517 | 3 | 0.557 | +0.040 (+7.7%) ✅ |
| **BERTopic** | 10 | 0.638 | 7 | 0.569 | -0.069 (-10.8%) |

#### Humanities Group Comparison

| Model | Paper |  | Our Results |  | Difference |
|-------|-------|-------|-------------|-------|------------|
| | Topics | Coherence | Topics | Coherence | ΔCoherence |
| **LDA** | 18 | 0.455 | 5 | 0.564 | **+0.109 (+24.0%)** ✅ |
| **LSI** | 6 | 0.455 | 8 | 0.444 | -0.011 (-2.4%) |
| **BERTopic** | 17 | 0.689 | 3 | 0.619 | -0.070 (-10.2%) |

#### Medicine Group Comparison

| Model | Paper |  | Our Results |  | Difference |
|-------|-------|-------|-------------|-------|------------|
| | Topics | Coherence | Topics | Coherence | ΔCoherence |
| **LDA** | 2 | 0.517 | 4 | 0.599 | **+0.082 (+15.9%)** ✅ |
| **LSI** | 6 | 0.499 | 3 | 0.405 | -0.094 (-18.8%) |
| **BERTopic** | 37 | 0.604 | 18 | 0.634 | +0.030 (+5.0%) ✅ |

  
## 📖 Reference
Khodeir, N., & Elghannam, F. (2024). Efficient topic identification for urgent MOOC Forum posts using BERTopic and traditional topic modeling techniques. *Education and Information Technologies*. Springer.
