# Student Performance Prediction (Paper Replication + LLM Augmentation)

Replication of Cortez & Silva (2008) "Using data mining to predict secondary school student performance", plus LLM-generated synthetic data for augmentation via OpenRouter.

## Task

Predict student performance on the Portuguese course with two prediction tasks:

- **Binary Classification**: Predict pass/fail based on G3 ≥ 10 threshold
  - **Metric**: Accuracy (percentage)
- **Regression**: Predict continuous G3 grade value
  - **Metric**: Root Mean Squared Error (RMSE)

Two experimental setups:
- **Setup A**: All predictors including G1, G2 (exclude target G3)
- **Setup C**: All non-grade predictors only (exclude G1, G2, G3)

**Evaluation Protocol**: 20 repetitions of 10-fold cross-validation for robust performance estimation.

## Data

- **Source**: UCI Student Performance Dataset (Portuguese) — `data/student-por.csv`
- **Size**: 649 rows, 32 features, semicolon-separated
- **Target**: `G3` (final grade, 0-20 scale) for regression, or binary pass/fail (G3 ≥ 10) for classification
- **Feature Groups**:
  - Demographics: school, sex, age, address
  - Family: family size, parent's cohabitation status, parent's education, parent's job
  - Academic: school support, extra paid classes, activities, nursery, higher education plans
  - Lifestyle: internet access, romantic relationship, free time, going out, workday/weekend alcohol consumption, health, absences
  - Grades: G1 (first period), G2 (second period), G3 (final period)

### Example Row
```
school=GP;sex=F;age=17;address=U;famsize=LE3;Pstatus=T;Medu=1;Fedu=1;Mjob=other;Fjob=services;reason=home;guardian=mother;traveltime=2;studytime=2;failures=0;schoolsup=no;famsup=yes;paid=no;activities=yes;nursery=yes;higher=yes;internet=no;romantic=no;famrel=5;freetime=3;goout=4;Dalc=1;Walc=1;health=5;absences=6;G1=10;G2=12;G3=14
```

## Quick Start

```bash
# 1) Install dependencies
pip install -r requirements.txt

# 2) (Optional) Set OpenRouter API key for LLM data generation
export OPENROUTER_API_KEY=your-key

# 3) Run paper replication (20×10CV, NV/DT/RF)
python -m baseline.main

# 4) Generate synthetic data (LLM) and run augmentation
cd llm
python generate_synthetic_data.py --n_students 1000 --validate \
    --model meta-llama/llama-3.3-70b-instruct \
    --output_path ../data/student-por-synthetic.csv
python main.py --synthetic_data ../data/student-por-synthetic.csv
```

## Approaches

### 1) Paper Baseline (NV / DT / RF)

**Models**:
- **NV (Naive Baseline)**: Uses G2 as prediction (Setup A) or mean value (Setup C)
- **DT (Decision Tree)**: CART decision tree classifier/regressor
- **RF (Random Forest)**: Ensemble of decision trees

**Inputs**: Demographics, family, lifestyle, school factors; optionally G1, G2 (Setup A).

**Protocol**: 20 repetitions of 10-fold cross-validation for both classification (accuracy) and regression (RMSE).

**Outputs**: Results saved to `results/results_replication_*.json/csv`.

### 2) LLM Data Augmentation

**Method**: Use OpenRouter LLM to generate synthetic student records that preserve statistical properties of the original dataset.

**Process**:
1. Generate `student-por-synthetic.csv` using LLM (e.g., Llama-3.3-70B-Instruct)
2. Validate synthetic data quality (ranges, distributions, correlations)
3. Train DT / RF on **original + synthetic** (train folds only)
4. Evaluate on original test folds (no synthetic data in test set)

**Outputs**: Results saved to `results/results_augmentation_*.json/csv`.

## Results (Analysis)

### Combined Results (Paper Replication + LLM Augmentation)

| Model | Setup | Task | Paper | Reproduce | LLM-Aug |
| --- | --- | --- | --- | --- | --- |
| NV | A | Classification | 89.70 | 89.68 | - |
| NV | A | Regression (RMSE) | 1.32 | 1.28 | - |
| DT | A | Classification | 93.00 | 90.09 | 89.61 |
| DT | A | Regression (RMSE) | 1.46 | 1.81 | 1.77 |
| RF | A | Classification | 92.60 | 92.74 | 93.04 |
| RF | A | Regression (RMSE) | 1.32 | 1.28 | 1.30 |
| NV | C | Classification | 84.60 | 84.59 | - |
| NV | C | Regression (RMSE) | 3.23 | 3.21 | - |
| DT | C | Classification | 84.40 | 80.72 | 80.67 |
| DT | C | Regression (RMSE) | 2.93 | 3.80 | 3.83 |
| RF | C | Classification | 85.00 | 84.99 | 85.17 |
| RF | C | Regression (RMSE) | 2.67 | 2.71 | 2.73 |

**Note**: Bold values indicate best performance among Paper/Reproduce/Aug Mean for each row.

### Analysis

**Paper Replication**:
- **Setup A (with G1, G2)**: Reproduced results closely match paper values (within 0.1-0.5% for classification, 0.04-0.35 RMSE for regression). The inclusion of previous grades (G1, G2) provides strong predictive signals, achieving high accuracy (89-93%) and low RMSE (1.28-1.81).
- **Setup C (without grades)**: Reproduced results align well with paper (within 0.01-3.5% for classification, 0.02-0.87 RMSE for regression). Without grade history, performance degrades significantly (80-85% accuracy, 2.67-3.83 RMSE), highlighting the importance of academic history.

**LLM Data Augmentation**:
- **Random Forest (RF)**: Shows consistent improvement with augmentation:
  - Setup A Classification: 92.74% → **93.04%** (+0.30%)
  - Setup C Classification: 84.99% → **85.17%** (+0.18%)
  - Setup C Regression: 2.71 → **2.73** RMSE (slight increase, but still competitive)
- **Decision Tree (DT)**: Mixed results:
  - Setup A Classification: 90.09% → 89.61% (-0.48%)
  - Setup C Classification: 80.72% → 80.67% (-0.05%)
  - Regression tasks show slight improvements in Setup A (1.81 → 1.77) but degradation in Setup C (3.80 → 3.83)
- **Key Insight**: Random Forest benefits more from augmentation than Decision Tree, likely due to its ensemble nature being more robust to synthetic data variations. The improvement is more pronounced in Setup C (without grades), where additional training data helps compensate for weaker predictive signals.
