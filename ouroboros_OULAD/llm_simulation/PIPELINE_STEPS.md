# LLM模拟Pipeline步骤说明

本目录包含三个独立的步骤脚本，用于将LLM模拟pipeline拆分为可独立执行的步骤。

## 步骤概览

1. **步骤1：模拟生成VLE数据** (`step1_generate_vle_data.py`)
   - 使用LLM Agent生成模拟的VLE数据
   - 输出格式：`studentVle.csv`（与真实OULAD格式一致）

2. **步骤2：Baseline训练并打标** (`step2_baseline_labeling.py`)
   - 从VLE数据提取特征
   - 使用真实OULAD数据训练baseline模型
   - 为合成数据分配标签

3. **步骤3：数据提取** (`step3_extract_students.py`)
   - 从已生成的VLE数据中按需提取200/500/1000个学生的数据
   - 确保格式与真实OULAD数据一致

## 使用流程

### 步骤1：生成VLE数据

```bash
# 生成200个学生的VLE数据
python step1_generate_vle_data.py \
    --n_students 200 \
    --modules BBB DDD EEE FFF \
    --llama_url http://localhost:8000 \
    --output_dir results/vle_data

# 使用流式输出（支持增量写入和恢复）
python step1_generate_vle_data.py \
    --n_students 200 \
    --modules BBB DDD EEE FFF \
    --stream_dir results/vle_data_stream \
    --resume

# 生成更多学生（用于后续提取）
python step1_generate_vle_data.py \
    --n_students 1000 \
    --modules BBB DDD EEE FFF \
    --output_dir results/vle_data
```

**输出文件：**
- `results/vle_data/vle_data_200_*/studentVle_200.csv` - VLE数据（studentVle.csv格式）
- `results/vle_data/vle_data_200_*/metadata.json` - 元数据

### 步骤2：Baseline训练并打标

```bash
# 为200个学生的VLE数据打标
python step2_baseline_labeling.py \
    --vle_data results/vle_data/vle_data_200_*/studentVle_200.csv \
    --modules BBB DDD EEE FFF \
    --output_dir results/labeled_data

# 为500个学生的VLE数据打标
python step2_baseline_labeling.py \
    --vle_data results/vle_data/vle_data_500_*/studentVle_500.csv \
    --output_dir results/labeled_data
```

```bash
# 批处理：多个VLE文件 × 多个days_to_cutoff
python step2_batch_labeling.py \
    --vle_files results/vle_data/studentVle_200_ex.csv results/vle_data/studentVle_500_ex.csv \
    --days_to_cutoff 0 4 7 \
    --output_dir results/labeled_data_batch \
    --timestamp_prefix day_sweep
```

**输出文件：**
- `results/labeled_data/labeled_200_*/synthetic_features_200.csv` - 特征数据
- `results/labeled_data/labeled_200_*/synthetic_features_200_labeled.csv` - 带标签的特征数据
- `results/labeled_data/labeled_200_*/baseline_info.json` - Baseline模型信息

### 步骤3：数据提取

```bash
# 从1000个学生的数据中提取200个
python step3_extract_students.py \
    --input results/vle_data/vle_data_1000_*/studentVle_1000.csv \
    --n_students 200 \
    --output results/extracted/studentVle_200.csv

# 从1000个学生的数据中提取500个
python step3_extract_students.py \
    --input results/vle_data/vle_data_1000_*/studentVle_1000.csv \
    --n_students 500 \
    --output results/extracted/studentVle_500.csv

# 提取1000个（完整数据）
python step3_extract_students.py \
    --input results/vle_data/vle_data_1000_*/studentVle_1000.csv \
    --n_students 1000 \
    --output results/extracted/studentVle_1000.csv
```

**输出文件：**
- `results/extracted/studentVle_200.csv` - 提取的VLE数据
- `results/extracted/studentVle_200.csv.metadata.json` - 提取元数据

### 步骤4：增强训练集（Train-Only）

```bash
python step4_augment_datasets.py \
    --synthetic_csv results/labeled_data/labeled_200_*/synthetic_features_200_labeled.csv \
    --synthetic_csv results/labeled_data/labeled_500_*/synthetic_features_500_labeled.csv \
    --output_dir results/augmented_data \
    --dataset_prefix llm
```

**输出文件：**
- `baseline_test_set.csv`：固定真实测试集（之后步骤5复用）
- `*/augmented_datasets/llm_xxx_train_{baseline,plus_200,...}.csv`
- `augmentation_summary.json`：记录输入/输出概览

### 步骤5：训练 + 评估

```bash
python step5_train_and_evaluate.py \
    --augmented_root results/augmented_data/step4_augmented_20251117_120000 \
    --test_csv results/augmented_data/step4_augmented_20251117_120000/baseline_test_set.csv \
    --dataset_names llm_1000 \
    --models LR RF NB \
    --output_dir results/model_training
```

**输出文件：**
- `training_results.json/.csv`：各模型×条件指标
- `config.json`：运行配置记录

### 步骤6：生成报告

```bash
python step6_generate_report.py \
    --results_json results/model_training/step5_results_*/training_results.json \
    --dataset_names llm_1000 \
    --output_dir results/model_reports
```

**输出文件：**
- `metrics_sim.csv`, `summary_table.csv`
- `prauc_comparison.png`, `improvement_heatmap.png`
- `report_config.json`

## 完整工作流示例

### 方案1：逐步生成（推荐用于小规模测试）

```bash
# 1. 生成200个学生的VLE数据
python step1_generate_vle_data.py --n_students 200 --modules BBB DDD EEE FFF

# 2. 为200个学生打标
python step2_baseline_labeling.py \
    --vle_data results/vle_data/vle_data_200_*/studentVle_200.csv

# 3. 增强 + 训练 + 报告
python step4_augment_datasets.py \
    --synthetic_csv results/labeled_data/labeled_200_*/synthetic_features_200_labeled.csv \
    --output_dir results/augmented_data

python step5_train_and_evaluate.py \
    --augmented_root results/augmented_data/step4_augmented_* \
    --test_csv results/augmented_data/step4_augmented_*/baseline_test_set.csv

python step6_generate_report.py \
    --results_json results/model_training/step5_results_*/training_results.json
```

### 方案2：批量生成后提取（推荐用于大规模实验）

```bash
# 1. 生成1000个学生的VLE数据（一次性生成，支持流式输出和恢复）
python step1_generate_vle_data.py \
    --n_students 1000 \
    --modules BBB DDD EEE FFF \
    --stream_dir results/vle_data_stream \
    --resume

# 2. 为1000个学生打标
python step2_baseline_labeling.py \
    --vle_data results/vle_data/vle_data_1000_*/studentVle_1000.csv

# 3. 按需提取200/500/1000个学生
python step3_extract_students.py \
    --input results/vle_data/vle_data_1000_*/studentVle_1000.csv \
    --n_students 200 \
    --output results/extracted/studentVle_200.csv

python step3_extract_students.py \
    --input results/vle_data/vle_data_1000_*/studentVle_1000.csv \
    --n_students 500 \
    --output results/extracted/studentVle_500.csv

# 4. 为提取的数据打标（如果需要）
python step2_baseline_labeling.py \
    --vle_data results/extracted/studentVle_200.csv

# 5. 自动化后续步骤
python step4_augment_datasets.py \
    --synthetic_csv results/labeled_data/labeled_1000_*/synthetic_features_1000_labeled.csv \
    --output_dir results/augmented_data

python step5_train_and_evaluate.py \
    --augmented_root results/augmented_data/step4_augmented_* \
    --test_csv results/augmented_data/step4_augmented_*/baseline_test_set.csv

python step6_generate_report.py \
    --results_json results/model_training/step5_results_*/training_results.json
```

## 数据格式说明

### studentVle.csv格式

所有步骤输出的VLE数据都遵循OULAD `studentVle.csv`格式：

| 列名 | 说明 | 示例 |
|------|------|------|
| `code_module` | 模块代码 | BBB, DDD, EEE, FFF |
| `code_presentation` | 课程呈现 | 2014J |
| `id_student` | 学生ID | BBB_llm_student_0001 |
| `id_site` | VLE资源ID | 1001, 2001, ... |
| `date` | 课程天数（1-56） | 1, 2, ..., 56 |
| `sum_click` | 点击数 | 0, 1, 2, ... |

### 特征数据格式

步骤2输出的特征数据包含：

- **基本信息**：`code_module`, `id_student`
- **周度特征**：`weekly_vle_clicks` (8周), `active_days_per_week` (8周), `recency_gaps` (8周)
- **累计统计**：`sum_click_fromvleopen`, `count_days_fromvleopen`, `last_login_rel`, 等
- **Demographics**：`gender`, `age_band`, `highest_education`, `region`, 等
- **标签**：`submitted` (0/1), `submitted_proba` (概率)

## 注意事项

1. **数据完整性验证**：步骤1和步骤3会自动验证每个学生是否有完整的56天数据（日期1-56）

2. **流式输出**：步骤1支持流式输出（`--stream_dir`），可以：
   - 增量写入，避免内存问题
   - 支持恢复（`--resume`），中断后可继续
   - 自动清理不完整的学生数据

3. **随机种子**：所有步骤都支持`--seed`参数，确保可重复性

4. **模块分配**：步骤3默认按模块均匀分配学生（`--strategy balanced`），也可以随机采样（`--strategy random`）

5. **数据格式一致性**：所有输出的VLE数据格式与真实OULAD数据完全一致，可以直接用于后续的特征提取和模型训练

## 故障排除

### 问题1：Llama服务器未就绪

```
错误: Llama服务器未就绪: http://localhost:8000
```

**解决**：确保Llama服务器正在运行
```bash
# 检查服务器状态
curl http://localhost:8000/health
```

### 问题2：学生数据不完整

```
警告: 发现 X 个不完整学生（不是所有56天）
```

**解决**：
- 步骤1会自动清理不完整学生
- 步骤3默认会跳过不完整学生（使用`--skip_validation`可禁用）

### 问题3：内存不足

**解决**：使用流式输出
```bash
python step1_generate_vle_data.py \
    --n_students 1000 \
    --stream_dir results/vle_data_stream
```

## 下一步

完成六个步骤后，您将自动获得：

1. `baseline_test_set.csv`（真实测试集快照）
2. `augmented_datasets/*.csv`（Baseline/Train+200/500/1000）
3. `training_results.json/.csv`（模型评估结果）
4. `metrics_sim.csv` 与可视化图表（步骤6输出）

