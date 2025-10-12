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

Note: Only the evaluation metrics are calculated — no inference is included.

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

Note: 
- The paper results are based on manual annotations of the official inferences.
- Our results are generated using the Llama evaluator on inferences from different models.


**Difference = Our Results - Paper Results**

| Tutor | Mistake_Identification | Mistake_Location | Revealing_of_the_Answer | Providing_Guidance | Actionability | Coherence | Tutor_Tone | Human-likeness |
|-------|:---------------------:|:----------------:|:----------------------:|:------------------:|:-------------:|:---------:|:----------:|:--------------:|
| **Llama3.1-8B** | | | | | | | | |
| Paper | 80.21 | 54.69 | 73.96 | 45.31 | 42.71 | 80.73 | 19.79 | 93.75 |
| Our | 85.42 | 61.98 | 50.52 | 5.21 | 11.98 | 80.73 | 19.27 | 56.25 |
| Diff | +5.21 | +7.29 | **-23.44** | **-40.1** | **-30.73** | 0 | **-0.52** | **-37.5** |
| **Mistral** | | | | | | | | |
| Paper | 93.23 | 73.44 | 86.46 | 63.54 | 70.31 | 86.98 | 15.10 | 95.31 |
| Our | 80.21 | 44.79 | 36.98 | 3.12 | 14.58 | 80.21 | 81.77 | 47.92 |
| Diff | **-13.02** | **-28.65** | **-49.48** | **-60.42** | **-55.73** | **-6.77** | +66.67 | **-47.39** |

### Table 3: Pedagogical Ability Assessment Results (<u>Llama Evaluator on Original Data</u>)

Note: 
- The paper results are based on manual annotations.
- Our results are generated using Llama as an automatic evaluator on the same original responses from the dataset.

**Difference = Our Results (Llama Evaluator) - Paper Results (Manual)**

| Tutor | Mistake_Identification | Mistake_Location | Revealing_of_the_Answer | Providing_Guidance | Actionability | Coherence | Tutor_Tone | Human-likeness |
|-------|:---------------------:|:----------------:|:----------------------:|:------------------:|:-------------:|:---------:|:----------:|:--------------:|
| **Expert** | | | | | | | | |
| Paper | 76.04 | 63.02 | 90.62 | 67.19 | 76.04 | 79.17 | 92.19 | 87.50 |
| Llama Eval | 58.33 | 35.42 | 57.81 | 3.65 | 1.56 | 54.69 | 16.15 | 45.31 |
| Diff | **-17.71** | **-27.60** | **-32.81** | **-63.54** | **-74.48** | **-24.48** | **-76.04** | **-42.19** |
| **GPT-4** | | | | | | | | |
| Paper | 94.27 | 84.38 | 53.12 | 76.04 | 46.35 | 90.17 | 37.50 | 89.62 |
| Llama Eval | 91.15 | 60.94 | 33.33 | 4.69 | 22.92 | 83.33 | 45.83 | 46.35 |
| Diff | **-3.12** | **-23.44** | **-19.79** | **-71.35** | **-23.43** | **-6.84** | +8.33 | **-43.27** |
| **Gemini** | | | | | | | | |
| Paper | 63.02 | 39.58 | 67.71 | 37.50 | 42.71 | 56.77 | 21.88 | 68.23 |
| Llama Eval | 77.60 | 40.62 | 88.54 | 3.12 | 9.90 | 75.00 | 46.35 | 47.40 |
| Diff | +14.58 | +1.04 | +20.83 | **-34.38** | **-32.81** | +18.23 | +24.47 | **-20.83** |
| **Llama3.1-405B** | | | | | | | | |
| Paper | 94.27 | 84.38 | 80.73 | 77.08 | 74.48 | 91.67 | 16.15 | 90.62 |
| Llama Eval | 89.58 | 60.94 | 67.71 | 8.33 | 15.10 | 82.29 | 45.83 | 57.29 |
| Diff | **-4.69** | **-23.44** | **-13.02** | **-68.75** | **-59.38** | **-9.38** | +29.68 | **-33.33** |
| **Llama3.1-8B** | | | | | | | | |
| Paper | 80.21 | 54.69 | 73.96 | 45.31 | 42.71 | 80.73 | 19.79 | 93.75 |
| Llama Eval | 84.90 | 55.21 | 46.35 | 6.77 | 9.90 | 72.40 | 26.04 | 47.92 |
| Diff | +4.69 | +0.52 | **-27.61** | **-38.54** | **-32.81** | **-8.33** | +6.25 | **-45.83** |
| **Mistral** | | | | | | | | |
| Paper | 93.23 | 73.44 | 86.46 | 63.54 | 70.31 | 86.98 | 15.10 | 95.31 |
| Llama Eval | 90.62 | 57.81 | 69.79 | 6.77 | 18.75 | 81.77 | 21.88 | 57.81 |
| Diff | **-2.61** | **-15.63** | **-16.67** | **-56.77** | **-51.56** | **-5.21** | +6.78 | **-37.50** |
| **Novice** | | | | | | | | |
| Paper | 43.33 | 16.67 | 80.00 | 11.67 | 1.67 | 50.00 | 90.00 | 35.00 |
| Llama Eval | 26.42 | 5.66 | 75.47 | 0.00 | 0.00 | 33.96 | 58.49 | 5.66 |
| Diff | **-16.91** | **-11.01** | **-4.53** | **-11.67** | **-1.67** | **-16.04** | **-31.51** | **-29.34** |
| **Phi3** | | | | | | | | |
| Paper | 28.65 | 26.04 | 73.96 | 17.71 | 11.98 | 39.58 | 45.31 | 52.08 |
| Llama Eval | 40.10 | 26.04 | 44.27 | 2.60 | 3.65 | 49.48 | 42.19 | 25.52 |
| Diff | +11.45 | **0.00** | **-29.69** | **-15.11** | **-8.33** | +9.90 | **-3.12** | **-26.56** |
| **Sonnet** | | | | | | | | |
| Paper | 85.42 | 69.79 | 94.79 | 59.38 | 60.94 | 88.54 | 54.69 | 96.35 |
| Llama Eval | 73.44 | 30.21 | 86.46 | 3.65 | 3.65 | 88.54 | 39.06 | 63.54 |
| Diff | **-11.98** | **-39.58** | **-8.33** | **-55.73** | **-57.29** | **0.00** | **-15.63** | **-32.81** |

#### Key Findings
- Llama evaluator shows significant discrepancies compared to manual annotations, particularly in Providing_Guidance and Actionability dimensions.
- The evaluator tends to be more conservative (lower scores) in most pedagogical dimensions.
- Coherence dimension shows relatively more consistency between Llama evaluator and manual annotations.

<!-- ### Table 6: Pedagogical Ability Assessment Results (AC Analysis)

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
| Diff | **+0.190** | **+0.207** | **+0.719** | **+0.212** | **+0.433** | **+0.316** | **+0.178** | **-0.147** | -->

<!-- ## Summary
 - (LLama318)  - The DAMR metric is consistent with the manual results reported in the original paper(EP1).
 - (LLama318) -  (EP2)The AC correlation metric is generally higher, aligning with EP1(Table 3). -->



<!-- 1. as we get high alignment with llama inference result, so that ac is high. it reason -->
