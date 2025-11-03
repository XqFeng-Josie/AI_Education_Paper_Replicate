# LLM Multi-Agent System for At-Risk Student Prediction

This module implements a multi-agent system using Large Language Models (LLMs) for predicting at-risk students in the OULAD dataset, inspired by the SimClass paper (NAACL 2025) and MAIC framework.

## 🎯 Overview

Traditional machine learning approaches for dropout prediction rely solely on structured features (VLE clicks, login counts, etc.) and lack interpretability. This LLM-based multi-agent system aims to:

1. **Understand Learning Behaviors**: Use LLMs to interpret student behavioral patterns semantically
2. **Multi-Perspective Analysis**: Deploy specialized agents for different aspects (academic, behavioral, temporal, peer comparison)
3. **Explainable Predictions**: Provide interpretable risk assessments with reasoning
4. **Actionable Interventions**: Generate personalized intervention recommendations

## 🏗️ Architecture

### Multi-Agent System Components

```
┌─────────────────────────────────────────────────────────────┐
│                   Student Behavior Data                      │
│              (VLE logs, demographics, assessments)           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │ Behavior-to-Text     │
          │ Converter            │
          └──────────┬───────────┘
                     │
                     ▼
    ┌────────────────────────────────────────┐
    │      Student Narrative                  │
    │  (Natural language description)         │
    └────┬───────────────────────────────┬───┘
         │                               │
         ▼                               ▼
┌─────────────────┐           ┌──────────────────┐
│ Academic        │           │ Behavioral       │
│ Advisor Agent   │           │ Analyst Agent    │
└────────┬────────┘           └────────┬─────────┘
         │                             │
         ▼                             ▼
┌─────────────────┐           ┌──────────────────┐
│ Peer Comparison │           │ Time Series      │
│ Agent           │           │ Analyst Agent    │
└────────┬────────┘           └────────┬─────────┘
         │                             │
         └──────────┬──────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  Decision Maker      │
         │  Agent               │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  Final Risk          │
         │  Assessment +        │
         │  Interventions       │
         └──────────────────────┘
```

### Agent Roles

1. **Academic Advisor**: Analyzes VLE engagement, study consistency, academic red flags
2. **Behavioral Analyst**: Examines login patterns, behavioral trends, disengagement signals  
3. **Peer Comparator**: Compares student with cohort, identifies outliers
4. **Time Series Analyst**: Detects temporal trends, momentum, early warnings
5. **Decision Maker**: Synthesizes all analyses, makes final risk determination

## 📦 Installation

```bash
# Install dependencies
pip install -r llm_experiments/requirements.txt

# Set up API keys (choose one)
export OPENAI_API_KEY="your-openai-key"
# or
export ANTHROPIC_API_KEY="your-anthropic-key"
# or use local models (no API key needed)
```

### 🚄 Multi-GPU Setup (Recommended for Large-Scale Experiments)

```bash
# 1. 配置GPU数量（编辑 llm/server/start_multi_gpu_servers.sh）
GPUS=(0 1 2 3)  # 根据你的GPU数量调整

# 2. start multi-gpu llama server
cd llm/server
bash start_multi_gpu_servers.sh
# 3. using multi-gpu llama 
python -m ll,.experiments.run_paper_replication \
    --pilot --n_students 20 \
    --llm_config config/llm_config_multi_gpu.yaml

# 4. stop multi-gpu llama server
cd llm/server
bash stop_multi_gpu_servers.sh
```

## 📁 Project Structure

```
llm/
├── config/
│   ├── llm_config.yaml          # LLM provider settings
│   └── agent_config.yaml        # Agent roles and weights
├── behavior_to_text.py          # Convert structured data to narratives
├── agents/
│   ├── base_agent.py            # Base agent class
│   ├── academic_advisor.py      # Academic advisor agent
│   ├── behavioral_analyst.py    # Behavioral analyst agent
│   ├── peer_comparator.py       # Peer comparison agent
│   ├── time_series_analyst.py   # Time series analyst agent
│   └── decision_maker.py        # Final decision maker
├── models/
│   ├── llm_wrapper.py           # LLM API wrappers
│   ├── llama_wrapper.py         # Llama model support
│   └── multi_agent_system.py   # Multi-agent coordinator
├── prompt.py                   # Prompt templates for agents
├── metrics.py                  # Evaluation metrics
├── run_paper_replication.py    # Paper replication experiment script(using llm)
├── server/
│   ├── llama_server.py              # Llama inference server
│   ├── multi_gpu_client.py          # Multi-GPU load balancer
│   ├── start_llama_server.sh        # Single GPU server startup
│   ├── start_multi_gpu_servers.sh   # Multi-GPU servers startup
│   └── stop_multi_gpu_servers.sh    # Stop all GPU servers
└── README.md
```


## 📊 Example Output

```json
{
  "final_decision": {
    "final_risk_level": "Risk",
    "risk_factors": [
      "Extremely low engagement (only 5 active days in 45 days)",
      "No activity in the last 15 days",
      "Activity level in bottom 5% of cohort"
    ],
    "confidence": "High",
    "recommended_interventions": [
      {"priority": "high", "intervention": "Immediate outreach by academic advisor"},
      {"priority": "high", "intervention": "Assessment of personal circumstances"}
    ]
  },
  "agent_analyses": {
    "academic_advisor": {"risk_score": 8.5, ...},
    "behavioral_analyst": {"risk_score": 9.0, ...},
    ...
  },
  "ground_truth": 0
}
```


## 📚 References

1. **Ouroboros**: [Early identification of at-risk students without models based on legacy data](https://oro.open.ac.uk/49731/1/paper.pdf)
2. **SimClass**: Zhang et al. "Simulating Classroom Education with LLM-Empowered Agents" (NAACL 2025) - [Paper](https://aclanthology.org/2025.naacl-long.520.pdf)
3. **MAIC Framework**: [GitHub](https://github.com/THU-MAIC/MAIC-Core)
4. **OULAD Dataset**: [Open University Learning Analytics Dataset](https://analyse.kmi.open.ac.uk/open-dataset)

## 📝 Experimental Settings

This implementation exactly replicates the traditional ML experimental setup:
- **Modules**: BBB, DDD, EEE, FFF (4 modules)
- **Presentation**: 2014J (single presentation)
- **Assessment**: TMA 1 (single assessment)
- **Time points**: Day 0-11 (12 consecutive days)
- **Label**: "submitted" (same as traditional ML)
- **Total configurations**: 4 modules × 1 presentation × 12 days = **48 configurations**

Key difference from traditional ML:
- **Features**: Focus on interpretable features (demographics, VLE statistics, registration) for better narrative generation, excluding detailed daily activity features

## 📊 Experimental Results

### Performance Summary (100 students per module per day)

| Day | PR-AUC | ROC-AUC | F1 | Precision | Recall | N_Students |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.3118 | 0.2597 | 0.6132 | 0.4559 | 0.9521 | 400 |
| 1 | 0.2448 | 0.3502 | 0.4765 | 0.3185 | 0.9592 | 400 |
| 2 | 0.1828 | 0.3358 | 0.3908 | 0.2477 | 0.9490 | 400 |
| 3 | 0.1571 | 0.2793 | 0.3852 | 0.2403 | 1.0000 | 400 |
| 4 | 0.1627 | 0.3305 | 0.3690 | 0.2299 | 0.9711 | 400 |
| 5 | 0.1349 | 0.2875 | 0.3438 | 0.2116 | 0.9737 | 400 |
| 6 | 0.1310 | 0.2743 | 0.3453 | 0.2132 | 0.9616 | 400 |
| 7 | 0.2525† | 0.3806 | 0.2663 | 0.1628 | 0.7500 | 400 |
| 8 | 0.2599† | 0.3661 | 0.2709 | 0.1702 | 0.6705 | 400 |
| 9 | 0.3655† | 0.3783 | 0.1908 | 0.1186 | 0.4881 | 400 |
| 10 | 0.3668† | 0.3744 | 0.1798 | 0.1109 | 0.4773 | 400 |
| 11 | 0.1450 | 0.2838 | 0.3156 | 0.2011 | 0.8166 | 400 |


## 🔧 Utility Scripts

### Calculate Metrics
```bash
# Calculate metrics from saved results
python -m llm.calculate_metrics --results_dir ./llm_experiments/results/paper_replication

# Calculate for specific module
python -m llm.calculate_metrics --results_dir ./results --module BBB
```


