# LLM引导的学生模拟与数据增强

> **项目目标**: 使用大语言模型驱动的多智能体系统模拟真实教学场景，生成高质量合成学生数据，增强训练集以提升预测模型性能  
> **方法**: LLM Multi-Agent模拟（基于Llama 3.1/3.3）  
> **参考**: instruction.txt Step-by-Step Instructions (LLM-Guided Simulation)

---

## 📚 文档导航

| 文档 | 用途 | 状态 |
|------|------|------|
| **[LLM_AGENT_EXPERIMENT_DESIGN.md](LLM_AGENT_EXPERIMENT_DESIGN.md)** | 🤖 LLM Agent实验设计 | ✅ 已完成 |
| **[generators/llm/README.md](generators/llm/README.md)** | 🔧 LLM Agent使用指南 | ✅ 已完成 |

---

## ⚡ 快速开始

### 环境准备

```bash
# 激活环境
conda activate oulad_env

# 确保依赖
pip install pandas numpy scikit-learn matplotlib seaborn pyyaml tables
```

### 1. 启动Llama服务器

```bash
# 在项目根目录启动Llama服务器
cd /projects/bdns/xfeng4/AI_Education_Paper_Replicate/ouroboros_OULAD
python llm/server/llama_server.py --port 8000 &
```

### 2. 测试连接

```bash
cd llm_simulation
python test_llama_connection.py
```

### 3. 运行端到端实验

**Pilot测试（20学生，每个模块5个，约30-60分钟）**:
```bash
python run_llm_end_to_end.py --mode pilot --n_students 20 --seed 42
```

**完整实验（200学生，每个模块50个，约2-4小时）**:
```bash
python run_llm_end_to_end.py --mode full --n_students 200 --seed 42
```

**启用流式写入与断点续跑（推荐大规模时使用）**:
```bash
python run_llm_end_to_end.py --mode full --n_students 200 \
  --stream_dir results/llm_agent_stream --resume
```

---

## 🎯 实验流程

完整流程严格遵循 instruction.txt 的6个步骤：

### Step 1: 模拟学生队列（8周）
- 使用LLM Multi-Agent系统模拟学生学习行为
- 生成N=200/500/1000个合成学生
- 输出：VLE事件日志（OULAD格式）

### Step 2: 映射OULAD特征
- 将VLE日志转换为OULAD风格的周度特征
- 特征包括：weekly_vle_clicks, active_days_per_week, recency_gaps等
- 输出：synthetic_features_N.csv
- 脚本：`step2_baseline_labeling.py`（或批处理 `step2_batch_labeling.py`）

### Step 3: 分配标签
- 使用多模块baseline模型（LR）预测合成数据的submitted标签
- 使用BBB, DDD, EEE, FFF四个模块的联合数据训练baseline
- 输出：synthetic_features_N_labeled.csv

### Step 4: 增强训练集
- 创建 train_plus_200, train_plus_500, train_plus_1000
- 仅增强TRAIN集，不修改TEST集
- 输出：增强数据集CSV文件
- 脚本：`step4_augment_datasets.py`

### Step 5: 重训练和评估
- 使用相同的超参数和pipeline作为baseline
- 模型：Logistic Regression, Random Forest, Naive Bayes, XGBoost
- 指标：PR-AUC, F1, Precision, Recall
- 输出：评估结果
- 脚本：`step5_train_and_evaluate.py`

### Step 6: 生成报告
- 输出 metrics_sim.csv（格式：condition, seed, pr_auc, f1, precision, recall）
- 生成可视化图表（PR-AUC对比图、提升热力图）
- 输出：完整实验报告
- 脚本：`step6_generate_report.py`

---

## 📂 项目结构

```
llm_simulation/
├── generators/
│   └── llm/                                # LLM Agent系统
│       ├── agents/
│       │   ├── instructor_agent.py         # 教师Agent
│       │   └── student_agent.py            # 学生Agent
│       ├── simulation/
│       │   ├── course_simulator.py         # 课程模拟器
│       │   └── action_to_vle_mapper.py     # Action→VLE转换
│       ├── prompts/
│       │   ├── instructor_prompts.yaml     # 教师提示词
│       │   └── student_prompts.yaml        # 学生提示词
│       ├── llm_client.py                   # Llama客户端
│       └── README.md                       # LLM Agent使用指南
│
├── features/                               # 特征映射
│   ├── mapper.py                           # VLE→特征转换
│   └── validator.py                        # 特征验证器
│
├── augmentation/                           # 数据增强
│   ├── multi_module_label_assignment.py    # 多模块标签分配
│   └── dataset_merger.py                   # 数据集合并
│
├── evaluation/                             # 模型评估
│   ├── model_trainer.py                    # 模型训练器
│   └── result_reporter.py                  # 结果报告
│
├── utils/                                  # 工具函数
│   ├── oulad_loader.py                     # OULAD数据加载
│   └── metrics.py                          # 评估指标
│
├── run_llm_end_to_end.py                   # 主入口脚本 ⭐
├── test_llama_connection.py                # 连接测试
├── statistics_llm_data.py                  # 数据统计工具
│
├── results/                                # 实验结果
│   └── llm_agent/                          # LLM方法结果
│
├── LLM_AGENT_EXPERIMENT_DESIGN.md         # LLM实验设计
└── README.md                               # 本文件
```

---

## 🚀 核心特性

### LLM Multi-Agent系统
- 🤖 基于Llama 3.1/3.3（本地部署）
- 👨‍🏫 Instructor Agent（发布内容、评分、答疑）
- 👨‍🎓 Student Agents（多种personality，动态行为）
- 🔄 Agent推理 → Actions → VLE Events → Features
- 📊 端到端流程（生成→特征→标签→合并→训练→评估）
- 🎯 多模块支持（BBB/DDD/EEE/FFF联合训练）

### 实验设计
- ✅ 严格遵循instruction.txt的6步骤设计
- ✅ 多模块baseline模型（4个模块联合训练）
- ✅ 源头平衡策略（保持因果关系）
- ✅ 完整评估指标（PR-AUC, F1, Precision, Recall）
- ✅ 可视化结果（对比图、热力图）

---

## 📊 输出文件

每次运行生成独立的结果目录：

```
results/llm_agent/end_to_end_full_200_YYYYMMDD_HHMMSS/
├── vle_logs_200.csv                        # VLE事件日志
├── synthetic_features_200.csv              # 合成特征
├── synthetic_features_200_labeled.csv      # 带标签的合成特征
├── metrics_llm_200.csv                     # 评估结果 ⭐
├── augmented_datasets/                     # 增强数据集
│   ├── llm_200_train_baseline.csv
│   ├── llm_200_train_plus_200.csv
│   ├── llm_200_train_plus_500.csv
│   └── llm_200_train_plus_1000.csv
└── llm_end_to_end.log                      # 运行日志
```

---

## 🔧 使用说明

### 命令行参数

```bash
python run_llm_end_to_end.py [OPTIONS]

选项:
  --mode {pilot,full}        实验模式 (默认: pilot)
  --n_students N             总学生数 (默认: 20 for pilot, 200 for full)
  --modules MODULES          模块列表 (默认: BBB DDD EEE FFF)
  --llama_url URL            Llama服务器URL (默认: http://localhost:8000)
  --output_dir DIR           输出目录 (默认: results/llm_agent)
  --stream_dir DIR           流式写入目录 (启用增量写入)
  --resume                   从检查点恢复 (需要--stream_dir)
  --seed SEED                随机种子 (默认: 42)
  --skip_simulation          跳过模拟步骤 (使用现有VLE logs)
```

### 示例

```bash
# 完整实验（200学生）
python run_llm_end_to_end.py --mode full --n_students 200 --seed 42

# 使用流式写入和断点续跑
python run_llm_end_to_end.py --mode full --n_students 500 \
  --stream_dir results/llm_agent_stream --resume

# 仅评估（跳过模拟）
python run_llm_end_to_end.py --mode full --n_students 200 \
  --skip_simulation --output_dir results/existing_run
```

---

## 📞 获取帮助

### LLM Agent相关
- **实验设计**: [LLM_AGENT_EXPERIMENT_DESIGN.md](LLM_AGENT_EXPERIMENT_DESIGN.md)
- **使用指南**: [generators/llm/README.md](generators/llm/README.md)

### 常见问题

**Q: Llama服务器不启动**
```bash
# 检查模型路径
ls /u/xfeng4/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct

# 检查GPU
nvidia-smi

# 查看日志
tail -f llm_server.log
```

**Q: 生成速度太慢**
- 减小`max_tokens`
- 使用更小的模型
- 先用pilot模式测试
- 使用流式写入和断点续跑

**Q: 学生数据不完整（少于56天）**
- 使用`--resume`参数从检查点恢复
- 系统会自动检测并重新生成不完整的学生

---

## 🎊 项目状态

- ✅ **LLM Agent系统**: 代码实现完成，支持端到端实验
- ✅ **端到端流程**: 完整实现6个步骤
- ✅ **多模块支持**: 支持BBB/DDD/EEE/FFF联合训练
- 📋 **下一步**: 运行完整实验，评估LLM生成数据的有效性

---

## 📖 相关资源

### 参考论文
1. Wu et al. (2024) - AutoGen: Enabling Next-Gen LLM Applications
2. Zhang et al. (2024) - Simulating Classroom Education with LLM-Empowered Agents
3. Chen et al. (2023) - AgentVerse: Facilitating Multi-Agent Collaboration

### 数据集
- OULAD: https://analyse.kmi.open.ac.uk/open-dataset
- 论文: https://dl.acm.org/doi/pdf/10.1145/3027385.3027449

---

*最后更新: 2025-01-XX*  
*基于Llama 3.1/3.3的LLM Multi-Agent学生模拟系统*
