# LLM Multi-Agent Simulation

> **Status**: ✅ Implemented  
> **Method**: LLM-driven student behavior generation using Llama 3.1 8B  
> **Comparison**: Statistical simulation (data-driven) vs LLM simulation (semantic-driven)

---

## 🚀 Quick Start

### 1. Start Llama Server

```bash
# From project root
cd /projects/bdns/xfeng4/AI_Education_Paper_Replicate/ouroboros_OULAD

# Start server (in a separate terminal or background)
python llm/server/llama_server.py --port 8000
```

### 2. Test Connection

```bash
cd llm_simulation
python test_llama_connection.py
```

### 3. Run Pilot Test (20 students)

```bash
python run_llm_agent_experiment.py --mode pilot --n_students 20
```

### 4. Run Full Experiment (1000 students)

```bash
# This will take 4-8 hours depending on GPU
python run_llm_agent_experiment.py --mode full --n_students 1000 --seed 42
```

---

## 📂 Directory Structure

```
generators/llm/
├── agents/
│   ├── instructor_agent.py      # Teacher Agent
│   └── student_agent.py         # Student Agents (4 personality types)
│
├── simulation/
│   ├── action_to_vle_mapper.py  # Action → VLE Event conversion
│   └── course_simulator.py      # 8-week simulation orchestrator
│
├── prompts/
│   ├── instructor_prompts.yaml  # Instructor prompts
│   └── student_prompts.yaml     # Student prompts (by personality)
│
├── llm_client.py                # Llama server client
└── README.md                    # This file
```

---

## 🎯 How It Works

### Architecture

```
Llama Server (port 8000)
    ↓
LlamaClient
    ↓
┌─────────────────────────────────┐
│   Instructor Agent              │
│   - Post weekly content         │
│   - Respond to forum questions  │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│   Student Agents (N students)   │
│   - high_performing (8.7%)      │
│   - average (8.4%)              │
│   - struggling (17.1%)          │
│   - at_risk (65.9%)             │
│                                 │
│   Each day, LLM decides:        │
│   - view_lecture                │
│   - read_resource               │
│   - post_forum                  │
│   - work_on_assignment          │
│   - ... or do_nothing           │
└─────────────────────────────────┘
    ↓
ActionToVLEMapper
    ↓
VLE Events (OULAD format)
    ↓
Features (23-dim) + Labels
    ↓
Augment Training Data
    ↓
Retrain & Evaluate
```

### Example Flow

**Week 1, Day 1**:
- Instructor posts: "Week 1 materials on Introduction to Computing"
- Student #0001 (high_performing):
  - LLM decides: `["check_homepage", "view_lecture", "read_resource"]`
  - Mapped to VLE: 3 events with clicks
- Student #0500 (at_risk):
  - LLM decides: `[]` (do nothing)
  - No VLE events

**Week 4, Day 28 (Deadline)**:
- All students decide whether to submit TMA 1
- LLM considers: personality, recent activities, deadline pressure
- high_performing → 100% submit
- at_risk → 0% submit

---

## 🔬 Personality Types

Based on real OULAD data distribution:

| Type | Proportion | Motivation | Time Mgmt | Behavior |
|------|-----------|------------|-----------|----------|
| **high_performing** | 8.7% | High | Organized | Regular access, early submission |
| **average** | 8.4% | Medium | Moderate | Decent engagement |
| **struggling** | 17.1% | Medium | Procrastinator | Late start, help-seeking |
| **at_risk** | 65.9% | Low | Poor | Minimal engagement |

---

## 📊 Output Files

After running experiment, you'll get:

```
results/llm_agent/full_1000_YYYYMMDD_HHMMSS/
├── vle_logs_1000.csv              # VLE events (OULAD format)
├── interaction_log_1000.json      # Full agent interaction log
└── synthetic_features_1000.csv    # Extracted features
```

---

## 🔧 Configuration

### Personality Distribution

Edit in `run_llm_agent_experiment.py`:
```python
personality_distribution = {
    'high_performing': 0.087,
    'average': 0.084,
    'struggling': 0.171,
    'at_risk': 0.659
}
```

### Prompts

Customize behavior by editing:
- `prompts/instructor_prompts.yaml`
- `prompts/student_prompts.yaml`

### LLM Parameters

In agent files, adjust:
```python
response = self.llm.generate(
    prompt=prompt,
    temperature=0.9,  # Higher = more creative
    max_tokens=128
)
```

---

## 🆚 Comparison with Statistical Method

| Aspect | Statistical (Method 1) | LLM Agent (Method 2) |
|--------|----------------------|---------------------|
| **Data Source** | Real OULAD statistics | LLM reasoning |
| **Speed** | ~3 min for 1000 | ~4-8 hours for 1000 |
| **Behavior** | Fixed 4 types | Dynamic per student |
| **Diversity** | Medium | High |
| **Interpretability** | High | Medium |
| **Realism** | Statistical | Semantic |

---

## ⚠️ Troubleshooting

### Server not starting
```bash
# Check if model exists
ls /u/xfeng4/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct

# Check GPU availability
nvidia-smi
```

### Generation too slow
- Use smaller model (8B instead of 70B)
- Reduce `max_tokens`
- Use batch processing

### Unrealistic behavior
- Adjust prompts in `prompts/` directory
- Tune temperature (lower = more consistent)
- Add validation rules in agents

---

## 📝 Next Steps

After generating synthetic data:

1. **Convert to full features** (if needed):
```bash
python features/mapper.py --vle_logs results/.../vle_logs_1000.csv
```

2. **Assign labels**:
```bash
python augmentation/label_assignment.py \
    --synthetic_features results/.../synthetic_features_1000.csv
```

3. **Merge and retrain**:
```bash
python augmentation/dataset_merger.py \
    --synthetic_labeled results/.../synthetic_features_1000_labeled.csv
```

4. **Evaluate**:
```bash
python evaluation/model_trainer.py --compare_methods
```

---

**🎊 LLM Agent simulation provides a complementary approach to statistical methods, potentially capturing more realistic learning behaviors!**

