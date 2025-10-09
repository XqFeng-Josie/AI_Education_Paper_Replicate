# Unifying AI Tutor Evaluation Paper Replication

Replication of: [Unifying AI Tutor Evaluation: An Evaluation Taxonomy for Pedagogical Ability Assessment of LLM-Powered AI Tutors](https://arxiv.org/pdf/2412.09416)  
Kaushal Kumar Maurya, KV Aditya Srivatsa, Kseniia Petukhova, and Ekaterina Kochmar

## 📊 Dataset Description

**MRBench Dataset**: A comprehensive benchmark for evaluating AI tutors' pedagogical abilities across multiple dimensions.

| File | Description |
|------|-------------|
| `MRBench_V1.json` | Original dataset containing 192 dialogues as detailed in the paper |
| `MRBench_V2.json` | Updated version with additional 8 dialogues, bringing the total to 200 examples |

*Dataset Describtion*

1. `conversation_id`: Serves as a unique identifier to track each dialogue in the dataset.
2. `conversation_history`: Captures the dialogue context relevant to the ongoing interaction.
3. `Data`: Specifies the dataset used for the interaction, such as MathDial or Bridge.
4. `Split`: Indicates whether the data point belongs to the test, train, or validation set.
5. `Topic`: Categorizes the dialogue into broad sub topics in Mathematics for easier filtering and analysis.
6. `Ground_Truth_Solution`: Provides a step-by-step solution to the problem discussed in the conversation, serving as a gold standard for evaluation.
7. `anno_llm_responses`: Stores LLM-specific responses with detailed annotations for evaluation based on multiple dimensions like mistake identification, guidance, etc.


## 🚀 Quick Start

#### Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Run Model Inference
```bash
# Generate responses using Llama models
python code/inference_llama.py

# Generate responses using Mistral models  
python code/inference_mistralai.py

# Evaluate model responses using llama
python code/inference_llama_eval.py
```


### 3. Data Analysis
Open and run the Jupyter notebooks for detailed analysis(Metric):
```bash
jupyter notebook code/data_analysis.ipynb
```

## 🔬 Experimental Results

### Table 3: Pedagogical Ability Assessment Results(<u>Origin Data Analysis, W/o inference</u>)

Performance comparison between original paper and our replication across 8 evaluation dimensions:

**Difference = Our Results - Paper Results**

| Tutor | Mistake_Identification | Mistake_Location | Revealing_of_the_Answer | Providing_Guidance | Actionability | Coherence | Tutor_Tone | Human-likeness |
|-------|:---------------------:|:----------------:|:----------------------:|:------------------:|:-------------:|:---------:|:----------:|:--------------:|
| **Expert** | | | | | | | | |
| Paper | 76.04 | 63.02 | 90.62 | 67.19 | 76.04 | 79.17 | 92.19 | 87.50 |
| Our | 81.25 | 68.75 | 97.92 | 72.92 | 81.77 | 84.90 | 17.19 | 94.79 |
| Diff | +5.21 | +5.73 | +7.30 | +5.73 | +5.73 | +5.73 | **-75.00** | +7.29 |
| **GPT-4** | | | | | | | | |
| Paper | 94.27 | 84.38 | 53.12 | 76.04 | 46.35 | 90.17 | 37.50 | 89.62 |
| Our | 94.27 | 85.42 | 54.69 | 77.08 | 46.88 | 92.71 | 36.98 | 93.23 |
| Diff | **0.00** | **-1.04** | **-1.57** | **-1.04** | **-0.53** | **-2.54** | +0.52 | **-3.61** |
| **Gemini** | | | | | | | | |
| Paper | 63.02 | 39.58 | 67.71 | 37.50 | 42.71 | 56.77 | 21.88 | 68.23 |
| Our | 87.50 | 62.50 | 92.71 | 58.85 | 61.98 | 82.29 | 39.58 | 95.31 |
| Diff | +24.48 | +22.92 | +25.00 | +21.35 | +19.27 | +25.52 | +17.70 | +27.08 |
| **Llama3.1-405B** | | | | | | | | |
| Paper | 94.27 | 84.38 | 80.73 | 77.08 | 74.48 | 91.67 | 16.15 | 90.62 |
| Our | 95.31 | 84.90 | 81.77 | 77.60 | 75.52 | 94.27 | 17.71 | 93.23 |
| Diff | +1.04 | +0.52 | +1.04 | +0.52 | +1.04 | +2.60 | +1.56 | +2.61 |
| **Llama3.1-8B** | | | | | | | | |
| Paper | 80.21 | 54.69 | 73.96 | 45.31 | 42.71 | 80.73 | 19.79 | 93.75 |
| Our | 81.25 | 56.25 | 76.56 | 46.88 | 42.71 | 82.81 | 19.79 | 96.35 |
| Diff | +1.04 | +1.56 | +2.60 | +1.57 | **0.00** | +2.08 | **0.00** | +2.60 |
| **Mistral** | | | | | | | | |
| Paper | 93.23 | 73.44 | 86.46 | 63.54 | 70.31 | 86.98 | 15.10 | 95.31 |
| Our | 93.23 | 74.48 | 89.06 | 66.15 | 71.35 | 88.02 | 16.67 | 97.40 |
| Diff | **0.00** | +1.04 | +2.60 | +2.61 | +1.04 | +1.04 | +1.57 | +2.09 |
| **Sonnet** | | | | | | | | |
| Paper | 85.42 | 69.79 | 94.79 | 59.38 | 60.94 | 88.54 | 54.69 | 96.30 |
| Our | 86.98 | 71.35 | 96.88 | 63.02 | 62.50 | 90.62 | 57.81 | 98.96 |
| Diff | +1.56 | +1.56 | +2.09 | +3.64 | +1.56 | +2.08 | +3.12 | +2.66 |

#### Key Findings

Check Gemini/Expert results!!!(done, no mistake)

### Table 3: Pedagogical Ability Assessment Results(<u>W inference</u>)

Performance comparison between original paper and our replication across 8 evaluation dimensions:
Note: Paper result is manual annotation of llama inference

**Difference = Our Results - Paper Results**

| Tutor | Mistake_Identification | Mistake_Location | Revealing_of_the_Answer | Providing_Guidance | Actionability | Coherence | Tutor_Tone | Human-likeness |
|-------|:---------------------:|:----------------:|:----------------------:|:------------------:|:-------------:|:---------:|:----------:|:--------------:|
| **Llama3.1-8B** | | | | | | | | |
| Paper | 80.21 | 54.69 | 73.96 | 45.31 | 42.71 | 80.73 | 19.79 | 93.75 |
| Our | 89.32 | 76.82 | 62.5 | 50.52 | 41.93 | 88.02 | 59.64 | 77.86 |
| Diff | +9.11 | +22.13 | **-11.46** | +5.21 | **-0.78** | +7.29 | +39.85 | **-15.89** |
<!-- | **Mistral** | | | | | | | | |
| Paper | 93.23 | 73.44 | 86.46 | 63.54 | 70.31 | 86.98 | 15.10 | 95.31 |
| Our | 41.67 | 38.54 | 43.75 | 22.4 | 43.23 | 62.5 | 46.35 | 69.79 |
| Diff | **-51.56** | **-34.90** | **-42.71** | **-41.14** | **-27.08** | **-24.48** | +31.25 | **-25.52** | -->

### Table 6: Pedagogical Ability Assessment Results (AC Analysis)

Performance comparison using AC (Accuracy-Consistency) analysis across 8 evaluation dimensions:

**Difference = Our Results - Paper Results**

| Tutor | Mistake_Identification | Mistake_Location | Revealing_of_the_Answer | Providing_Guidance | Actionability | Coherence | Tutor_Tone | Human-likeness |
|-------|:---------------------:|:----------------:|:----------------------:|:------------------:|:-------------:|:---------:|:----------:|:--------------:|
| **Expert** | | | | | | | | |
| Paper | -0.01 | -0.25 | -0.13 | -0.19 | -0.08 | -0.11 | -0.40 | +0.01 |
| Our | +0.053 | +0.096 | +0.235 | +0.064 | +0.082 | +0.106 | +0.354 | +0.012 |
| Diff | **+0.063** | **+0.346** | **+0.365** | **+0.254** | **+0.162** | **+0.216** | **+0.754** | **+0.002** |
| **GPT-4** | | | | | | | | |
| Paper | -0.07 | +0.01 | -0.20 | -0.21 | +0.02 | -0.02 | -0.11 | +0.08 |
| Our | +0.393 | +0.144 | +0.552 | +0.202 | +0.174 | +0.106 | +0.395 | +0.136 |
| Diff | **+0.463** | **+0.134** | **+0.752** | **+0.412** | **+0.154** | **+0.126** | **+0.505** | **+0.056** |
| **Gemini** | | | | | | | | |
| Paper | +0.02 | +0.09 | -0.06 | -0.16 | -0.12 | -0.07 | -0.24 | +0.07 |
| Our | +0.077 | +0.042 | +0.368 | +0.061 | +0.026 | +0.009 | +0.493 | +0.012 |
| Diff | **+0.057** | **-0.048** | **+0.428** | **+0.221** | **+0.146** | **+0.079** | **+0.733** | **-0.058** |
| **Llama-3.1-405B** | | | | | | | | |
| Paper | -0.03 | -0.08 | -0.05 | -0.05 | +0.00 | +0.06 | -0.13 | +0.11 |
| Our | +0.009 | -0.113 | +0.450 | -0.046 | +0.129 | -0.038 | +0.316 | +0.056 |
| Diff | **+0.039** | **-0.033** | **+0.500** | **+0.004** | **+0.129** | **-0.098** | **+0.446** | **-0.054** |
| **Llama-3.1-8B** | | | | | | | | |
| Paper | -0.12 | -0.37 | -0.17 | +0.04 | -0.07 | -0.16 | -0.29 | +0.11 |
| Our | +0.143 | +0.081 | +0.243 | +0.138 | +0.062 | -0.017 | +0.389 | +0.083 |
| Diff | **+0.263** | **+0.451** | **+0.413** | **+0.098** | **+0.132** | **+0.143** | **+0.679** | **-0.027** |
| **Mistral** | | | | | | | | |
| Paper | -0.06 | -0.11 | -0.10 | -0.23 | -0.15 | -0.20 | -0.19 | +0.06 |
| Our | +0.238 | -0.047 | +0.437 | +0.042 | +0.113 | +0.076 | +0.337 | +0.052 |
| Diff | **+0.298** | **+0.063** | **+0.537** | **+0.272** | **+0.263** | **+0.276** | **+0.527** | **-0.008** |
| **Novice** | | | | | | | | |
| Paper | -0.37 | +0.09 | -0.56 | -0.72 | +0.15 | -0.15 | -0.71 | +0.18 |
| Our | +0.637 | +0.563 | +0.646 | +0.780 | -0.065 | +0.237 | +0.609 | +0.009 |
| Diff | **+1.007** | **+0.473** | **+1.206** | **+1.500** | **-0.215** | **+0.387** | **+1.319** | **-0.171** |
| **Phi3** | | | | | | | | |
| Paper | -0.67 | -0.58 | -0.51 | -0.51 | -0.46 | -0.33 | -0.62 | +0.03 |
| Our | +0.579 | +0.492 | +0.301 | +0.570 | +0.260 | +0.483 | +0.566 | +0.215 |
| Diff | **+1.249** | **+1.072** | **+0.811** | **+1.080** | **+0.720** | **+0.813** | **+1.186** | **+0.185** |
| **Sonnet** | | | | | | | | |
| Paper | -0.11 | -0.12 | -0.21 | -0.11 | -0.22 | -0.08 | -0.20 | +0.07 |
| Our | +0.080 | +0.087 | +0.509 | +0.102 | +0.213 | +0.236 | -0.022 | -0.077 |
| Diff | **+0.190** | **+0.207** | **+0.719** | **+0.212** | **+0.433** | **+0.316** | **+0.178** | **-0.147** |

## Summary
 - (LLama318)  - The DAMR metric is consistent with the manual results reported in the original paper(EP1).
 - (LLama318) -  (EP2)The AC correlation metric is generally higher, aligning with EP1(Table 3).



<!-- 1. as we get high alignment with llama inference result, so that ac is high. it reason -->
