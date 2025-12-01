# LLM Multi-Agent模拟实验设计

> **目标**: 使用大语言模型驱动的多智能体系统模拟真实教学场景，生成高质量合成学生数据  
> **核心创新**: 从**统计驱动**转向**语义驱动**的数据生成  
> **参考**: instruction.txt Step-by-Step Instructions (LLM-Guided Simulation)

---

## 📋 目录

1. [实验对比](#1-实验对比统计驱动-vs-llm驱动)
2. [框架选择](#2-框架选择)
3. [系统架构](#3-系统架构)
4. [实验设计](#4-实验设计)
5. [实施步骤](#5-实施步骤)
6. [评估方案](#6-评估方案)

---

## 1. 实验对比：统计驱动 vs LLM驱动

### 1.1 两种方法的本质区别

| 维度 | 方法1: 统计驱动 (已完成) | 方法2: LLM驱动 (新实验) |
|------|----------------------|---------------------|
| **数据来源** | 真实OULAD统计分布 | LLM模拟的教学场景 |
| **生成逻辑** | 采样+规则 | Agent交互+推理 |
| **行为模式** | 4类学生（固定行为参数） | N个Agent（动态行为） |
| **VLE生成** | 直接生成点击数 | 对话/行为→VLE事件 |
| **时间复杂度** | O(N×W) 快速 | O(N×W×T) 较慢（LLM推理） |
| **数据多样性** | 中等（受统计约束） | 高（LLM创造性） |
| **可解释性** | 高（参数明确） | 中（LLM黑盒） |
| **计算成本** | 低（秒级） | 高（分钟级，需GPU） |

### 1.2 为什么需要LLM方法？

**统计驱动的局限**:
```
统计模拟器：
  ├─ 学生类型固定（4类）
  ├─ 行为参数固定（VLE clicks范围、提交概率等）
  ├─ 缺乏真实的"为什么"
  └─ 无法捕捉复杂的学习动机和策略
```

**LLM驱动的优势**:
```
LLM Agent模拟器：
  ├─ 每个学生有unique personality（基于prompt）
  ├─ 行为基于reasoning（"我今天感觉焦虑，多看资料"）
  ├─ 自然的时序依赖（Week 1经历影响Week 2行为）
  └─ 捕捉真实学习模式（拖延、临时抱佛脚、peer影响等）
```

### 1.3 研究问题

**RQ1**: LLM生成的合成数据是否比统计方法更有效提升模型性能？  
**RQ2**: LLM能否生成更真实的学习行为模式（时序依赖、社交互动等）？  
**RQ3**: 两种方法生成的数据有何质量差异？

---

## 2. 框架选择

### 2.1 instruction.txt推荐的3个框架

| 框架 | 优点 | 缺点 | 推荐度 |
|------|------|------|--------|
| **AutoGen** | ✅ 灵活通用<br>✅ 支持Llama 3.3<br>✅ 活跃社区 | ⚠️ 需自己设计教学场景 | ⭐⭐⭐⭐⭐ **推荐** |
| **SimClass** | ✅ 专为教学设计<br>✅ 内置session控制 | ❌ 需改造支持Llama | ⭐⭐⭐⭐ |
| **AgentVerse** | ✅ 多轮自我改进 | ❌ 复杂度高 | ⭐⭐⭐ |

**选择**: **AutoGen** - 最灵活，最适合本实验

### 2.2 AutoGen框架概述

```
AutoGen Multi-Agent架构:
  ├─ Instructor Agent (教师)
  │   ├─ 发布课程内容
  │   ├─ 布置作业
  │   ├─ 回答学生提问
  │   └─ 评分和反馈
  │
  └─ Student Agents (N个学生)
      ├─ 学习课程材料
      ├─ 访问VLE资源
      ├─ 提问和讨论
      └─ 提交作业
```

---

## 3. 系统架构

### 3.1 整体流程

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: LLM Multi-Agent Simulation (8 weeks)               │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Instructor Agent  ←→  Student Agent 1                      │
│       ↓                     ↓                                 │
│  发布内容/作业          访问VLE/提问/提交                      │
│                                                               │
│  Interaction Logs:                                            │
│  - Timestamp                                                  │
│  - Agent ID                                                   │
│  - Action Type (view_resource, post_forum, submit_quiz...)  │
│  - Content (dialogue, resource accessed...)                  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Interaction → VLE Event Conversion                 │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Agent Action           →  VLE Activity Type                │
│  ────────────────────────────────────────                   │
│  "view lecture"         →  oucontent (1 click)              │
│  "read PDF"             →  resource (1 click)               │
│  "post question"        →  forumng (2 clicks)               │
│  "submit assignment"    →  oucollaborate (5 clicks)         │
│  "check homepage"       →  homepage (1 click)               │
│  "take quiz"            →  quiz (3-10 clicks)               │
│                                                               │
│  Output: VLE Event Log (OULAD格式)                          │
│  - code_module, code_presentation, id_student                │
│  - id_site (VLE resource ID)                                 │
│  - date (相对于课程开始)                                      │
│  - sum_click (该资源该天的点击次数)                           │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 3: VLE Events → Weekly Features (复用现有mapper.py)  │
├─────────────────────────────────────────────────────────────┤
│  使用 features/mapper.py 转换为23维特征                      │
│  - weekly_vle_clicks (len=8)                                │
│  - active_days_per_week (len=8)                             │
│  - recency_gaps (len=8)                                     │
│  - demographics (复用统计方法)                               │
│  - assessment features (a1_submitted, a1_score等)           │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 4-6: Label Assignment & Augmentation & Evaluation     │
│  (复用现有pipeline: label_assignment.py, dataset_merger.py) │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 关键模块

#### 3.2.1 **Instructor Agent**

```python
class InstructorAgent:
    """
    教师Agent - 负责课程组织和内容发布
    
    Prompt Template:
    ---
    You are an experienced university instructor teaching a computer science course.
    Your responsibilities:
    - Post weekly learning materials (lectures, readings, assignments)
    - Respond to student questions in the forum
    - Grade assignments and provide feedback
    - Monitor student progress
    
    Current Week: {week_num}
    Course Content: {content_description}
    Available VLE Resources:
    - Lecture videos (oucontent)
    - PDF readings (resource)
    - Discussion forum (forumng)
    - Quiz system (quiz)
    - Assignment portal (oucollaborate)
    ---
    """
    
    def post_weekly_content(self, week_num):
        """发布本周学习材料"""
        pass
    
    def respond_to_question(self, student_question):
        """回答学生提问"""
        pass
    
    def grade_assignment(self, student_id, submission):
        """评分作业"""
        pass
```

#### 3.2.2 **Student Agent**

```python
class StudentAgent:
    """
    学生Agent - 每个学生有unique personality和学习策略
    
    Prompt Template:
    ---
    You are a university student enrolled in a computer science course.
    
    Personality Profile:
    - Learning Style: {learning_style}  # visual/reading/hands-on
    - Motivation Level: {motivation}    # high/medium/low
    - Time Management: {time_mgmt}      # organized/procrastinator
    - Prior Knowledge: {prior_knowledge}  # strong/moderate/weak
    - Social Tendency: {social}         # active/passive in forums
    
    Current Situation:
    - Week: {week_num}
    - Upcoming Deadline: {deadline}
    - Your Progress: {progress_summary}
    
    Available Actions:
    - view_lecture(topic)
    - read_resource(file_name)
    - post_forum(question)
    - take_quiz(quiz_id)
    - submit_assignment(assignment_id)
    - check_homepage()
    
    Decide your next action based on your personality and current situation.
    ---
    """
    
    def __init__(self, student_id, personality_profile):
        self.id = student_id
        self.profile = personality_profile
        self.history = []  # 历史行为
        self.knowledge_state = {}  # 知识掌握情况
        
    def decide_next_action(self, week_info):
        """
        基于personality和当前状态决定下一步行动
        LLM推理：考虑deadline、历史行为、知识gaps等
        """
        pass
    
    def view_resource(self, resource_type, resource_name):
        """访问VLE资源"""
        pass
    
    def interact_with_forum(self, action):
        """论坛交互（post/reply）"""
        pass
```

#### 3.2.3 **Action → VLE Event Mapper**

```python
class ActionToVLEMapper:
    """
    将Agent的高层action转换为OULAD格式的VLE events
    """
    
    ACTION_TO_VLE_MAPPING = {
        'view_lecture': {
            'activity_type': 'oucontent',
            'click_count': lambda: random.randint(1, 3),  # 看lecture可能点击1-3次
            'duration': lambda: random.randint(5, 30)  # 5-30分钟
        },
        'read_pdf': {
            'activity_type': 'resource',
            'click_count': lambda: 1,
            'duration': lambda: random.randint(10, 60)
        },
        'post_forum_question': {
            'activity_type': 'forumng',
            'click_count': lambda: random.randint(2, 5),  # 写帖子需多次点击
            'duration': lambda: random.randint(5, 20)
        },
        'reply_forum': {
            'activity_type': 'forumng',
            'click_count': lambda: random.randint(1, 3),
            'duration': lambda: random.randint(2, 10)
        },
        'take_quiz': {
            'activity_type': 'quiz',
            'click_count': lambda: random.randint(5, 15),  # quiz多题多点击
            'duration': lambda: random.randint(10, 45)
        },
        'submit_assignment': {
            'activity_type': 'oucollaborate',
            'click_count': lambda: random.randint(3, 10),  # 提交过程多步骤
            'duration': lambda: random.randint(5, 30)
        },
        'check_homepage': {
            'activity_type': 'homepage',
            'click_count': lambda: 1,
            'duration': lambda: random.randint(1, 5)
        },
        'access_glossary': {
            'activity_type': 'glossary',
            'click_count': lambda: 1,
            'duration': lambda: random.randint(2, 10)
        },
        'view_subpage': {
            'activity_type': 'subpage',
            'click_count': lambda: 1,
            'duration': lambda: random.randint(2, 15)
        },
        'external_link': {
            'activity_type': 'url',
            'click_count': lambda: 1,
            'duration': lambda: random.randint(5, 30)
        }
    }
    
    def convert_action_to_vle_event(self, agent_action, timestamp):
        """
        转换单个action为VLE event
        
        Input: 
            agent_action = {
                'student_id': 'student_001',
                'action_type': 'view_lecture',
                'resource_name': 'Week1_IntroToCS',
                'timestamp': datetime
            }
        
        Output:
            vle_event = {
                'id_student': 'student_001',
                'date': 1,  # 相对于课程开始的天数
                'id_site': '12345',  # VLE resource ID
                'activity_type': 'oucontent',
                'sum_click': 2
            }
        """
        mapping = self.ACTION_TO_VLE_MAPPING.get(agent_action['action_type'])
        
        return {
            'id_student': agent_action['student_id'],
            'date': self._calculate_relative_day(timestamp),
            'activity_type': mapping['activity_type'],
            'sum_click': mapping['click_count'](),
            # ... 其他字段
        }
```

---

## 4. 实验设计

### 4.1 实验配置

**课程设置** (严格遵循instruction.txt):
```yaml
course:
  name: "BBB - Computer Science Basics"
  duration: 8 weeks
  presentation: "2014J"
  
assessments:
  - name: "TMA 1"
    type: "Tutor Marked Assignment"
    due_week: 4
    due_day: 28
    weight: 30%
    
vle_resources:
  - homepage (course_overview)
  - lectures_week_1_to_8 (oucontent)
  - readings (resource, PDF files)
  - discussion_forum (forumng)
  - quiz_bank (quiz)
  - assignment_portal (oucollaborate)
  - glossary (glossary)
  - external_references (url)
```

**Agent配置**:
```yaml
instructor:
  model: "llama-3.3-70b-instruct"  # 本地部署
  temperature: 0.7
  max_tokens: 512
  
students:
  model: "llama-3.3-70b-instruct"
  temperature: 0.9  # 更高的创造性
  max_tokens: 256
  
  # 学生personality分布（基于真实数据的4类）
  personality_distribution:
    high_performing:
      proportion: 0.087
      traits:
        motivation: "high"
        time_management: "organized"
        prior_knowledge: "strong"
        learning_style: ["visual", "reading", "hands-on"]
        social_tendency: "active"
        
    average:
      proportion: 0.084
      traits:
        motivation: "medium"
        time_management: "moderate"
        prior_knowledge: "moderate"
        learning_style: ["visual", "reading"]
        social_tendency: "moderate"
        
    struggling:
      proportion: 0.171
      traits:
        motivation: "medium"
        time_management: "procrastinator"
        prior_knowledge: "weak"
        learning_style: ["hands-on"]
        social_tendency: "active"  # 寻求帮助
        
    at_risk:
      proportion: 0.659
      traits:
        motivation: "low"
        time_management: "poor"
        prior_knowledge: "weak"
        learning_style: "passive"
        social_tendency: "passive"
```

### 4.2 模拟流程

```python
# Pseudo-code for 8-week simulation

for week in range(1, 9):
    # Instructor posts weekly content
    instructor.post_weekly_materials(week)
    
    # Each day of the week
    for day in range(7):
        # Each student decides actions for today
        for student in students:
            # LLM reasoning: 基于personality、deadline、history
            actions = student.decide_daily_actions(
                week=week,
                day=day,
                upcoming_deadline=get_next_deadline(),
                history=student.history,
                knowledge_gaps=student.knowledge_state
            )
            
            # Execute actions → VLE events
            for action in actions:
                vle_event = action_mapper.convert(action, timestamp)
                vle_logs.append(vle_event)
                student.history.append(action)
        
        # Instructor responds to forum posts (if any)
        if forum_has_new_posts():
            instructor.respond_to_forum()
    
    # End of week: Assessment submission (Week 4)
    if week == 4:
        for student in students:
            # Decide whether to submit based on personality
            if student.decide_submit_assignment():
                student.submit_tma1()
                vle_logs.append({
                    'id_student': student.id,
                    'date': week * 7,
                    'activity_type': 'oucollaborate',
                    'sum_click': 10,
                    'submission': True
                })

# Output: vle_logs (OULAD format)
```

### 4.3 实验变体

我们设计**3组实验**对比：

| 实验组 | 数据来源 | 样本量 | 说明 |
|--------|---------|--------|------|
| **Baseline** | 真实OULAD | 1137 | 基线 |
| **Method 1** | 统计驱动模拟 | +200/500/1000 | 已完成（数据驱动配置） |
| **Method 2** | LLM Agent模拟 | +200/500/1000 | **新实验** |
| **Method 3** | 混合方法 | +500 (50% M1 + 50% M2) | 探索性 |

---

## 5. 实施步骤

### Phase 1: 环境搭建（1-2天）

```bash
# Step 1: 安装AutoGen
pip install pyautogen

# Step 2: 配置Llama 3.3本地模型
# 使用Ollama部署 (https://ollama.com/)
ollama pull llama3.3:70b-instruct

# Step 3: 测试AutoGen + Llama连接
python test_autogen_llama.py
```

**示例配置** (`autogen_config.yaml`):
```yaml
llm_config:
  model: "llama3.3:70b-instruct"
  base_url: "http://localhost:11434/v1"  # Ollama API endpoint
  api_key: "ollama"  # dummy key
  temperature: 0.7
  max_tokens: 512
```

### Phase 2: 实现Agent系统（3-5天）

**目录结构**:
```
llm_simulation/
├── agents/
│   ├── __init__.py
│   ├── instructor_agent.py      # 教师Agent
│   ├── student_agent.py         # 学生Agent
│   └── agent_factory.py         # Agent工厂（生成N个学生）
│
├── simulation/
│   ├── __init__.py
│   ├── course_simulator.py      # 8周课程模拟器
│   ├── action_to_vle_mapper.py  # Action→VLE转换器
│   └── interaction_logger.py    # 交互日志记录
│
├── prompts/
│   ├── instructor_prompts.yaml  # 教师prompt模板
│   └── student_prompts.yaml     # 学生prompt模板
│
└── run_llm_agent_experiment.py  # 主实验脚本
```

**关键实现**:

1. **`agents/instructor_agent.py`**:
```python
from autogen import AssistantAgent

class InstructorAgent:
    def __init__(self, llm_config):
        self.agent = AssistantAgent(
            name="Instructor",
            system_message=self._load_instructor_prompt(),
            llm_config=llm_config
        )
    
    def _load_instructor_prompt(self):
        return """
        You are an experienced university instructor...
        [详细的教师角色prompt]
        """
    
    def post_weekly_content(self, week_num):
        message = f"Post learning materials for Week {week_num}"
        response = self.agent.generate_reply(messages=[{"role": "user", "content": message}])
        return self._parse_content_post(response)
```

2. **`agents/student_agent.py`**:
```python
class StudentAgent:
    def __init__(self, student_id, personality_profile, llm_config):
        self.id = student_id
        self.profile = personality_profile
        
        # 个性化system message
        system_msg = self._build_student_prompt(personality_profile)
        
        self.agent = AssistantAgent(
            name=f"Student_{student_id}",
            system_message=system_msg,
            llm_config=llm_config
        )
    
    def _build_student_prompt(self, profile):
        return f"""
        You are a university student with the following characteristics:
        - Motivation: {profile['motivation']}
        - Time Management: {profile['time_management']}
        - Prior Knowledge: {profile['prior_knowledge']}
        ...
        [根据personality动态生成prompt]
        """
    
    def decide_daily_actions(self, context):
        prompt = f"""
        Current Situation:
        - Week: {context['week']}
        - Day: {context['day']}
        - Upcoming Deadline: {context['deadline']}
        - Your Recent Activities: {context['history'][-5:]}
        
        What will you do today? Choose 0-5 actions from:
        - view_lecture(topic)
        - read_resource(name)
        - post_forum(question)
        - take_quiz(id)
        - check_homepage()
        - [do nothing]
        
        Output format: JSON list of actions
        """
        
        response = self.agent.generate_reply(messages=[{"role": "user", "content": prompt}])
        return self._parse_actions(response)
```

3. **`simulation/course_simulator.py`**:
```python
class CourseSimulator:
    """
    8周课程模拟器
    """
    
    def __init__(self, n_students, llm_config, random_seed=42):
        self.instructor = InstructorAgent(llm_config)
        
        # 生成N个学生（按personality分布）
        self.students = AgentFactory.create_students(
            n=n_students,
            personality_dist=load_personality_distribution(),
            llm_config=llm_config,
            seed=random_seed
        )
        
        self.action_mapper = ActionToVLEMapper()
        self.vle_logs = []
    
    def simulate_8_weeks(self):
        """运行8周模拟"""
        
        for week in range(1, 9):
            print(f"=== Simulating Week {week} ===")
            
            # Instructor posts content
            weekly_content = self.instructor.post_weekly_content(week)
            
            # Simulate each day
            for day in range(1, 8):
                daily_date = (week - 1) * 7 + day
                
                # Each student acts
                for student in self.students:
                    actions = student.decide_daily_actions({
                        'week': week,
                        'day': day,
                        'date': daily_date,
                        'deadline': self._get_next_deadline(daily_date),
                        'history': student.history
                    })
                    
                    # Convert actions to VLE events
                    for action in actions:
                        vle_event = self.action_mapper.convert(
                            action, 
                            student_id=student.id,
                            date=daily_date
                        )
                        self.vle_logs.append(vle_event)
                
                # Instructor responds (optional)
                if self._forum_has_questions():
                    self.instructor.respond_to_forum()
            
            # Week 4: Assignment submission
            if week == 4:
                self._handle_assignment_submission(week)
        
        return self.vle_logs
    
    def export_to_oulad_format(self, output_path):
        """导出为OULAD格式的VLE日志"""
        df = pd.DataFrame(self.vle_logs)
        df.to_csv(output_path, index=False)
```

### Phase 3: 测试与调试（2-3天）

```bash
# 小规模测试（20个学生，2周）
python run_llm_agent_experiment.py --mode pilot --n_students 20 --weeks 2

# 检查生成的VLE logs
python -c "
import pandas as pd
df = pd.read_csv('results/llm_pilot/vle_logs.csv')
print(df.head())
print(df['activity_type'].value_counts())
print(df.groupby('id_student')['sum_click'].sum().describe())
"
```

**预期调试问题**:
1. LLM输出格式不一致 → 需要robust parsing
2. 某些学生生成行为过于极端 → 调整prompt
3. VLE event分布不合理 → 调整action mapping
4. 计算速度慢 → 考虑batch processing或并行化

### Phase 4: 全量实验（5-7天）

```bash
# 生成200/500/1000个学生（每组约2-8小时，取决于GPU）
python run_llm_agent_experiment.py \
    --mode full \
    --n_students 1000 \
    --weeks 8 \
    --seed 42 \
    --output_dir results/llm_agent_full

# 将VLE logs转为特征（复用现有mapper）
python convert_vle_to_features.py \
    --vle_logs results/llm_agent_full/vle_logs_1000.csv \
    --output results/llm_agent_full/synthetic_features_1000.csv

# 标签分配（复用现有pipeline）
python augmentation/label_assignment.py \
    --synthetic_features results/llm_agent_full/synthetic_features_1000.csv \
    --output results/llm_agent_full/synthetic_features_1000_labeled.csv

# 数据增强和重训练（复用）
python run_full_experiment.py \
    --synthetic_data results/llm_agent_full/ \
    --mode evaluate \
    --output_dir results/llm_vs_statistical
```

---

## 6. 评估方案

### 6.1 评估维度

#### 维度1: 模型性能（主要指标）

| 指标 | Baseline | 统计驱动+1000 | LLM Agent+1000 | 期望 |
|------|----------|--------------|---------------|------|
| **PR-AUC (LR)** | 0.5227 | 0.5251 (+0.46%) | ? | > 0.53 |
| **PR-AUC (RF)** | 0.5077 | 0.5447 (+7.27%) | ? | > 0.55 |
| **PR-AUC (NB)** | 0.5777 | 0.6182 (+7.00%) | ? | > 0.62 |

**关键问题**: LLM生成的数据能否进一步提升性能？

#### 维度2: 数据质量

**统计检验**:
```python
# 分布相似度测试
from scipy.stats import ks_2samp, wasserstein_distance

# 比较合成数据vs真实数据的分布
for feature in ['sum_click', 'active_days', 'recency_gaps']:
    # Kolmogorov-Smirnov test
    stat, pvalue = ks_2samp(real_data[feature], synthetic_data[feature])
    
    # Wasserstein distance (Earth Mover's Distance)
    dist = wasserstein_distance(real_data[feature], synthetic_data[feature])
    
    print(f"{feature}: KS_stat={stat:.3f}, p={pvalue:.3f}, WD={dist:.3f}")
```

**对比表**:

| 特征 | 真实数据 | 统计驱动 | LLM Agent | 最优 |
|------|---------|---------|----------|------|
| VLE Clicks分布 | μ=45.3, σ=32.1 | ? | ? | - |
| KS统计量 (越小越好) | - | 0.12 | ? | < 0.15 |
| Wasserstein距离 | - | 8.3 | ? | < 10.0 |
| 标签平衡性 | 34.1% | 34.2% | ? | ~34% |

#### 维度3: 行为真实性（定性分析）

**人工评估** (抽样50个学生):

| 评估项 | 统计驱动 | LLM Agent | 评分标准 |
|--------|---------|----------|---------|
| **时序依赖性** | 低（独立采样） | ? | Week 1-4行为是否合理递进？ |
| **行为多样性** | 中（4类固定） | ? | 是否有unique patterns？ |
| **社交互动** | 无 | ? | 论坛互动是否自然？ |
| **deadline效应** | 弱（规则） | ? | Week 4前是否有冲刺？ |

#### 维度4: 计算成本

| 方法 | 生成1000学生时间 | GPU需求 | 内存 |
|------|----------------|---------|------|
| 统计驱动 | ~3分钟 | 无 | <2GB |
| LLM Agent | ~4-8小时 | A100 40GB | ~30GB |

---

### 6.2 实验报告结构

生成文件：`LLM_AGENT_EXPERIMENT_RESULTS.md`

```markdown
# LLM Agent实验结果报告

## 1. 执行概况
- 生成样本: 200, 500, 1000
- 运行时间: X小时
- LLM模型: Llama 3.3 70B
- 总token消耗: X million

## 2. 性能对比

### Table 1: PR-AUC对比
| Model | Baseline | +Statistical | +LLM Agent | Improvement |
|-------|----------|--------------|------------|-------------|
| LR    | 0.5227   | 0.5251       | X.XXXX     | +X.XX%      |
| RF    | 0.5077   | 0.5447       | X.XXXX     | +X.XX%      |
| NB    | 0.5777   | 0.6182       | X.XXXX     | +X.XX%      |

### Table 2: 三种方法性能排名
| 排名 | N=200 | N=500 | N=1000 |
|------|-------|-------|--------|
| 1st  | ?     | ?     | ?      |
| 2nd  | ?     | ?     | ?      |
| 3rd  | ?     | ?     | ?      |

## 3. 数据质量分析

### 3.1 分布相似度
[KS-test, Wasserstein distance结果]

### 3.2 行为模式可视化
[t-SNE降维可视化：真实vs统计vs LLM]

## 4. 关键发现

### 4.1 LLM Agent的优势
- ...

### 4.2 LLM Agent的局限
- ...

## 5. 结论与建议
```

---

## 7. 技术挑战与解决方案

### 挑战1: LLM输出格式不稳定

**问题**: LLM可能不按JSON格式输出action list

**解决方案**:
```python
def parse_llm_actions(response, max_retries=3):
    """Robust parsing with retries"""
    
    for attempt in range(max_retries):
        try:
            # Try JSON parsing
            actions = json.loads(response)
            return actions
        except json.JSONDecodeError:
            # Try regex extraction
            pattern = r'\{.*?\}'
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                return [json.loads(m) for m in matches]
            
            # Retry with stricter prompt
            if attempt < max_retries - 1:
                response = retry_with_stricter_prompt()
    
    # Fallback: return empty actions
    return []
```

### 挑战2: 计算成本高

**优化策略**:
1. **Batch Processing**: 多个学生并行生成
2. **Caching**: 缓存常见action的LLM响应
3. **Model Distillation**: 考虑使用更小的模型（Llama 3.3 8B）
4. **Selective Simulation**: 只对关键时刻（Week 4前）使用LLM，其他时间用规则

### 挑战3: 行为过于极端

**问题**: 某些学生可能生成0点击或1000+点击

**解决方案**: 添加soft constraints
```python
def validate_daily_actions(actions, student_profile):
    """Validate and clip extreme behaviors"""
    
    # 根据personality设定合理上下限
    if student_profile['motivation'] == 'low':
        max_actions = 3
    elif student_profile['motivation'] == 'high':
        max_actions = 10
    else:
        max_actions = 5
    
    # Clip
    if len(actions) > max_actions:
        actions = actions[:max_actions]
    
    return actions
```

---

## 8. 预期成果

### 8.1 研究贡献

1. **方法论创新**: 首次对比统计驱动vs LLM驱动的教育数据合成方法
2. **实证证据**: 提供LLM在教育数据增强中的有效性证据
3. **开源工具**: 可复现的LLM Agent模拟框架

### 8.2 潜在论文

**标题**: "Statistical vs Semantic: A Comparative Study of Data Augmentation Methods for Learning Analytics"

**Abstract**:
```
We compare two approaches to synthetic student data generation:
(1) Statistical simulation based on real data distributions
(2) LLM-powered multi-agent simulation with semantic reasoning

Experiments on OULAD show that:
- Statistical methods achieve X% improvement with low cost
- LLM methods achieve Y% improvement with higher cost but better behavior realism
- Hybrid approaches may offer the best trade-off

Our findings suggest that...
```

---

## 9. 下一步行动计划

### Week 1: 环境搭建
- [ ] 安装AutoGen + Llama 3.3
- [ ] 测试连接和基础对话
- [ ] 准备prompt templates

### Week 2: Agent实现
- [ ] 实现InstructorAgent
- [ ] 实现StudentAgent (多personality)
- [ ] 实现ActionToVLEMapper

### Week 3: 小规模测试
- [ ] Pilot run (20学生, 2周)
- [ ] 调试prompt和parsing
- [ ] 验证VLE logs格式

### Week 4: 全量实验
- [ ] 生成200/500/1000学生
- [ ] 特征转换和标签分配
- [ ] 模型重训练和评估

### Week 5: 分析与报告
- [ ] 性能对比分析
- [ ] 数据质量评估
- [ ] 撰写实验报告

---

## 10. 参考资源

### AutoGen文档
- Official Docs: https://microsoft.github.io/autogen/
- Multi-Agent Patterns: https://microsoft.github.io/autogen/docs/tutorial/introduction

### Llama 3.3配置
- Ollama Setup: https://ollama.com/
- Model Card: https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct

### 相关论文
1. Wu et al. (2024) - AutoGen: Enabling Next-Gen LLM Applications
2. Zhang et al. (2024) - Simulating Classroom Education with LLM-Empowered Agents
3. Chen et al. (2023) - AgentVerse: Facilitating Multi-Agent Collaboration

---

**🎯 核心目标**: 验证LLM驱动的数据生成方法是否能超越统计方法，为教育数据稀缺场景提供更有效的解决方案！

*设计完成时间: 2025-11-09*  
*预计实施周期: 4-5周*

