"""
Unified Single-Agent System - Cost-Optimized Multi-Perspective Analysis
Integrates multiple agent perspectives but uses only one LLM call to complete all analysis
Supports both zero-shot and few-shot modes
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import json
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class UnifiedAgent:
    """
    Unified Single-Agent System
    Integrates perspectives from academic advisor, behavioral analyst, peer comparison expert, and temporal analyst
    Completes multi-dimensional analysis and final decision in one LLM call
    """
    
    def __init__(self, 
                 llm_wrapper,
                 behavior_converter=None,
                 mode: str = "zero_shot",
                 num_few_shot_examples: int = 3):
        """
        Initialize Unified Agent
        
        Args:
            llm_wrapper: LLM wrapper instance
            behavior_converter: BehaviorToTextConverter instance (optional)
            mode: Operation mode
                - "zero_shot": Zero-shot mode, no examples used
                - "few_shot": Few-shot mode, uses examples
            num_few_shot_examples: Number of examples to use in few-shot mode
        """
        self.llm = llm_wrapper
        self.converter = behavior_converter
        self.mode = mode
        self.num_few_shot_examples = num_few_shot_examples
        
        # Store few-shot examples (for few-shot mode)
        self.few_shot_examples = []
        
        logger.info(f"Initialized UnifiedAgent in {mode} mode")
    
    def set_few_shot_examples(self, examples: List[Dict[str, Any]]):
        """
        Set few-shot examples
        
        Args:
            examples: List of examples, each containing:
                - 'student_narrative': Student behavior description
                - 'peer_context': Peer comparison context (optional)
                - 'risk_level': Risk level ('High Risk', 'Medium Risk', 'Low Risk', 'No Risk')
                - 'risk_score': Risk score (0-10)
                - 'reasoning': Analysis reasoning process
        """
        self.few_shot_examples = examples
        logger.info(f"Set {len(examples)} few-shot examples")
    
    def _build_zero_shot_prompt(self, 
                               student_narrative: str,
                               peer_context: str = "") -> str:
        """
        Build prompt for zero-shot mode
        
        Args:
            student_narrative: Student behavior description
            peer_context: Peer comparison context
            
        Returns:
            Complete prompt string
        """
        prompt = f"""You are an experienced educational data analysis expert who needs to comprehensively analyze from multiple dimensions whether a student will submit their assignment.

## Task Objective
**Core Task**: Predict whether a student will submit their assignment before the deadline.
- **High Risk**: Student is very likely NOT to submit the assignment (submitted=0)
- **Low Risk**: Student is very likely to submit the assignment (submitted=1)

## Multi-Agent Collaborative Analysis Framework
You need to play the roles of four professional analysts, independently analyze from different perspectives, and then provide a comprehensive assessment:

### Agent 1: Academic Performance Analyst
**Role**: Focus on student's academic performance and assignment-related behaviors
**Analysis Focus**:
- Evaluate student's historical assignment submission patterns (whether submitted on time, submission quality)
- Analyze VLE (Virtual Learning Environment) activity levels (click counts, material access, interaction frequency)
- Assess academic progress trajectory (grade trends, learning engagement)
- Identify academic risk signals (low activity levels, lack of participation)

### Agent 2: Behavioral Pattern Analyst
**Role**: Focus on student's learning behavior patterns and engagement
**Analysis Focus**:
- Analyze login frequency and regularity (daily/weekly login patterns)
- Evaluate learning session patterns (session duration, session intervals, learning intensity)
- Identify engagement consistency (whether behavior is stable, whether there are sudden changes)
- Detect disengagement signals (decreasing login frequency, shortened session duration, prolonged inactivity)

### Agent 3: Peer Comparison Expert
**Role**: Identify anomalies and relative risks through peer comparison
**Analysis Focus**:
- Conduct relative performance analysis with peers/same-course students (ranking, percentile)
- Identify outliers (behaviors significantly deviating from peer group)
- Assess position in the cohort (whether in low-engagement group)
- Analyze relative risk (even if absolute level is normal, relatively low may still indicate risk)

### Agent 4: Temporal Analyst
**Role**: Identify early warning signals through temporal trends
**Analysis Focus**:
- Detect behavior trends (whether engagement is increasing, decreasing, or stable)
- Assess engagement trajectory (change patterns in recent weeks/days)
- Identify early warning signals (sustained engagement decline, sudden activity interruption)
- Detect temporal anomalies (sudden behavior pattern changes, periodic interruptions)

## Student Information

### Student Behavior Description:
{student_narrative}

"""
        
        if peer_context:
            prompt += f"""### Peer Comparison Context:
{peer_context}

"""
        
        prompt += """## Analysis Requirements

Please follow the structure below, independently analyze from three professional perspectives, and then provide a comprehensive assessment:

### 1. Academic & Behavioral Analysis (Agent 1 Perspective)
**As an Academic & Behavioral Analyst, please analyze:**
- Key Findings: [Specific findings based on VLE activity, engagement patterns, learning behaviors]
- Risk Indicators: [List specific risk indicators, e.g., "Very low VLE clicks (below 10)", "zero active days", "no recent activity", etc.]
- Submission Probability: [0.0-1.0, where 0=very unlikely to submit, 1=very likely to submit]

### 2. Temporal & Trend Analysis (Agent 2 Perspective)
**As a Temporal & Trend Analyst, please analyze:**
- Key Findings: [Specific findings based on temporal trends, engagement trajectory, recent patterns]
- Risk Indicators: [List specific temporal risk indicators, e.g., "decreasing engagement trend", "no activity in last 7 days", etc.]
- Submission Probability: [0.0-1.0, where 0=very unlikely to submit, 1=very likely to submit]

### 3. Peer Comparison Analysis (Agent 3 Perspective)
**As a Peer Comparison Analyst, please analyze:**
- Key Findings: [Specific findings based on peer comparison, e.g., "engagement below cohort median", "in bottom 25% percentile", etc.]
- Risk Indicators: [List specific relative risk indicators based on comparison with peers]
- Submission Probability: [0.0-1.0, where 0=very unlikely to submit, 1=very likely to submit]

### 4. Comprehensive Assessment (Multi-Agent Collaborative Decision)
**Based on the analysis from all three agents, provide the final assessment:**
- Final Submission Probability: [0.0-1.0, weighted average considering all three perspectives]
  - 0.0-0.2: Very unlikely to submit (High Risk, submitted=0)
  - 0.2-0.4: Unlikely to submit (Medium-High Risk)
  - 0.4-0.6: Uncertain (Medium Risk)
  - 0.6-0.8: Likely to submit (Low-Medium Risk)
  - 0.8-1.0: Very likely to submit (Low Risk, submitted=1)
- Risk Level: [High Risk / Medium Risk / Low Risk]
- Main Risk Factors: [List the 2-3 most important factors influencing the prediction]
- Confidence: [High / Medium / Low, based on data quality and consistency across perspectives]
- Key Reasoning: [Brief explanation of why this probability was assigned]

## Output Format

Please output in JSON format as follows:
```json
{{
    "academic_behavioral_analysis": {{
        "key_findings": "...",
        "risk_indicators": ["...", "..."],
        "submission_probability": 0.5
    }},
    "temporal_trend_analysis": {{
        "key_findings": "...",
        "risk_indicators": ["...", "..."],
        "submission_probability": 0.5
    }},
    "peer_comparison_analysis": {{
        "key_findings": "...",
        "risk_indicators": ["...", "..."],
        "submission_probability": 0.5
    }},
    "final_assessment": {{
        "submission_probability": 0.5,
        "risk_level": "Medium Risk",
        "main_risk_factors": ["...", "..."],
        "confidence": "Medium",
        "key_reasoning": "..."
    }}
}}
```

Please begin the analysis:"""
        
        return prompt
    
    def _build_few_shot_prompt(self,
                               student_narrative: str,
                               peer_context: str = "") -> str:
        """
        Build prompt for few-shot mode
        
        Args:
            student_narrative: Student behavior description
            peer_context: Peer comparison context
            
        Returns:
            Complete prompt string
        """
        # Build examples section
        examples_text = "## Examples\n\n"
        for i, example in enumerate(self.few_shot_examples[:self.num_few_shot_examples], 1):
            examples_text += f"""### Example {i}

**Student Behavior Description:**
{example['student_narrative']}

"""
            if example.get('peer_context'):
                examples_text += f"""**Peer Comparison Context:**
{example['peer_context']}

"""
            
            examples_text += f"""**Analysis Result:**
```json
{{
    "academic_behavioral_analysis": {{
        "key_findings": "{example.get('academic_findings', 'N/A')}",
        "submission_probability": {example.get('submission_probability', 0.5)}
    }},
    "temporal_trend_analysis": {{
        "key_findings": "{example.get('temporal_findings', 'N/A')}",
        "submission_probability": {example.get('submission_probability', 0.5)}
    }},
    "peer_comparison_analysis": {{
        "key_findings": "{example.get('peer_findings', 'N/A')}",
        "submission_probability": {example.get('submission_probability', 0.5)}
    }},
    "final_assessment": {{
        "submission_probability": {example.get('submission_probability', 0.5)},
        "risk_level": "{example.get('risk_level', 'Medium Risk')}",
        "main_risk_factors": {json.dumps(example.get('risk_factors', []))},
        "confidence": "{example.get('confidence', 'Medium')}",
        "key_reasoning": "{example.get('reasoning', 'N/A')}"
    }}
}}
```

"""
        
        # Build task description (same as zero-shot)
        prompt = f"""You are an experienced educational data analysis expert who needs to comprehensively analyze from multiple dimensions whether a student will submit their assignment.

## Task Objective
**Core Task**: Predict whether a student will submit their assignment before the deadline.
- **submitted=1**: Student WILL submit the assignment
- **submitted=0**: Student will NOT submit the assignment

**Prediction Output**: Provide a submission probability between 0.0 and 1.0:
- **0.0-0.2**: Very unlikely to submit (High Risk)
- **0.2-0.4**: Unlikely to submit (Medium-High Risk)
- **0.4-0.6**: Uncertain (Medium Risk)
- **0.6-0.8**: Likely to submit (Low-Medium Risk)
- **0.8-1.0**: Very likely to submit (Low Risk)

## Multi-Agent Collaborative Analysis Framework
You need to play the roles of three professional analysts, independently analyze from different perspectives, and then provide a comprehensive assessment:

### Agent 1: Academic & Behavioral Analyst
**Role**: Focus on student's academic performance and learning behaviors
**Analysis Focus**:
- Analyze VLE (Virtual Learning Environment) activity levels (total clicks, active days, engagement intensity)
- Evaluate learning session patterns (activity distribution, consistency, peak activity)
- Assess engagement quality (consecutive learning days, recent activity levels)
- Identify risk signals (low activity, inactivity periods, zero engagement)

### Agent 2: Temporal & Trend Analyst
**Role**: Identify patterns and warnings through temporal trends
**Analysis Focus**:
- Detect behavior trends (whether engagement is increasing, decreasing, or stable)
- Assess engagement trajectory (early period vs late period comparison)
- Identify early warning signals (sustained engagement decline, activity interruption)
- Analyze temporal patterns (recent activity vs overall pattern)

### Agent 3: Peer Comparison Analyst
**Role**: Identify anomalies through peer comparison (if peer data available)
**Analysis Focus**:
- Conduct relative performance analysis with cohort (percentile ranking)
- Identify outliers (behaviors significantly deviating from peer median)
- Assess position in the cohort (above/below median performance)
- Analyze relative risk (even if absolute level seems normal, being significantly below peers indicates risk)

{examples_text}

## Current Student Information

### Student Behavior Description:
{student_narrative}

"""
        
        if peer_context:
            prompt += f"""### Peer Comparison Context:
{peer_context}

"""
        
        prompt += """## Analysis Requirements

Please follow the structure below, independently analyze from three professional perspectives, and then provide a comprehensive assessment:

### 1. Academic & Behavioral Analysis (Agent 1 Perspective)
**As an Academic & Behavioral Analyst, please analyze:**
- Key Findings: [Specific findings based on VLE activity, engagement patterns, learning behaviors]
- Risk Indicators: [List specific risk indicators, e.g., "Very low VLE clicks (below 10)", "zero active days", "no recent activity", etc.]
- Submission Probability: [0.0-1.0, where 0=very unlikely to submit, 1=very likely to submit]

### 2. Temporal & Trend Analysis (Agent 2 Perspective)
**As a Temporal & Trend Analyst, please analyze:**
- Key Findings: [Specific findings based on temporal trends, engagement trajectory, recent patterns]
- Risk Indicators: [List specific temporal risk indicators, e.g., "decreasing engagement trend", "no activity in last 7 days", etc.]
- Submission Probability: [0.0-1.0, where 0=very unlikely to submit, 1=very likely to submit]

### 3. Peer Comparison Analysis (Agent 3 Perspective)
**As a Peer Comparison Analyst, please analyze:**
- Key Findings: [Specific findings based on peer comparison, e.g., "engagement below cohort median", "in bottom 25% percentile", etc.]
- Risk Indicators: [List specific relative risk indicators based on comparison with peers]
- Submission Probability: [0.0-1.0, where 0=very unlikely to submit, 1=very likely to submit]

### 4. Comprehensive Assessment (Multi-Agent Collaborative Decision)
**Based on the analysis from all three agents, provide the final assessment:**
- Final Submission Probability: [0.0-1.0, weighted average considering all three perspectives]
  - 0.0-0.2: Very unlikely to submit (High Risk, submitted=0)
  - 0.2-0.4: Unlikely to submit (Medium-High Risk)
  - 0.4-0.6: Uncertain (Medium Risk)
  - 0.6-0.8: Likely to submit (Low-Medium Risk)
  - 0.8-1.0: Very likely to submit (Low Risk, submitted=1)
- Risk Level: [High Risk / Medium Risk / Low Risk]
- Main Risk Factors: [List the 2-3 most important factors influencing the prediction]
- Confidence: [High / Medium / Low, based on data quality and consistency across perspectives]
- Key Reasoning: [Brief explanation of why this probability was assigned]

## Output Format

Please output in JSON format as follows:
```json
{{
    "academic_behavioral_analysis": {{
        "key_findings": "...",
        "risk_indicators": ["...", "..."],
        "submission_probability": 0.5
    }},
    "temporal_trend_analysis": {{
        "key_findings": "...",
        "risk_indicators": ["...", "..."],
        "submission_probability": 0.5
    }},
    "peer_comparison_analysis": {{
        "key_findings": "...",
        "risk_indicators": ["...", "..."],
        "submission_probability": 0.5
    }},
    "final_assessment": {{
        "submission_probability": 0.5,
        "risk_level": "Medium Risk",
        "main_risk_factors": ["...", "..."],
        "confidence": "Medium",
        "key_reasoning": "..."
    }}
}}
```

Please refer to the examples above and analyze the current student:"""
        
        return prompt
    
    def predict(self,
                student_narrative: str,
                peer_context: str = "",
                return_analysis: bool = True) -> Dict[str, Any]:
        """
        Predict student risk level
        
        Args:
            student_narrative: Student behavior description text
            peer_context: Peer comparison context (optional)
            return_analysis: Whether to return detailed analysis
            
        Returns:
            Dictionary containing prediction results
        """
        logger.info(f"UnifiedAgent prediction (mode={self.mode})")
        
        # Select prompt based on mode
        if self.mode == "few_shot" and len(self.few_shot_examples) > 0:
            prompt = self._build_few_shot_prompt(student_narrative, peer_context)
        else:
            prompt = self._build_zero_shot_prompt(student_narrative, peer_context)
        
        # Call LLM
        try:
            response = self.llm.generate(prompt)
            
            # Parse JSON response
            result = self._parse_llm_response(response)
            
            # Add metadata
            result['mode'] = self.mode
            result['student_narrative'] = student_narrative
            # Save prompt and response for debugging/analysis
            result['prompt'] = prompt
            result['raw_response'] = response
            
            logger.info(f"Prediction complete: {result.get('final_assessment', {}).get('risk_level', 'Unknown')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in UnifiedAgent prediction: {e}")
            return {
                'error': str(e),
                'final_assessment': {
                    'risk_level': 'No Risk',
                    'risk_score': 0,
                    'confidence': 'Low'
                },
                'mode': self.mode,
                'prompt': prompt,  # Save prompt even on error
                'raw_response': None  # No response due to error
            }
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM's JSON response
        
        Args:
            response: LLM's raw response
            
        Returns:
            Parsed dictionary
        """
        try:
            # Try to extract JSON section
            if '```json' in response:
                json_start = response.find('```json') + 7
                json_end = response.find('```', json_start)
                json_str = response[json_start:json_end].strip()
            elif '```' in response:
                json_start = response.find('```') + 3
                json_end = response.find('```', json_start)
                json_str = response[json_start:json_end].strip()
            else:
                # Try to find first { and last }
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                json_str = response[json_start:json_end]
            
            result = json.loads(json_str)
            return result
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON, attempting to extract key information: {e}")
            # If JSON parsing fails, try to extract key information
            return self._extract_key_info_from_text(response)
    
    def _extract_key_info_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract key information from text (fallback method)
        
        Args:
            text: LLM's text response
            
        Returns:
            Extracted information dictionary
        """
        # Simple text extraction logic
        result = {
            'academic_analysis': {'risk_score': 5},
            'behavioral_analysis': {'risk_score': 5},
            'peer_analysis': {'risk_score': 5},
            'temporal_analysis': {'risk_score': 5},
            'final_assessment': {
                'risk_level': 'Medium Risk',
                'risk_score': 5,
                'confidence': 'Low'
            }
        }
        
        # Try to extract risk level
        if 'High Risk' in text or 'high risk' in text.lower():
            result['final_assessment']['risk_level'] = 'High Risk'
            result['final_assessment']['risk_score'] = 8
        elif 'Low Risk' in text or 'low risk' in text.lower():
            result['final_assessment']['risk_level'] = 'Low Risk'
            result['final_assessment']['risk_score'] = 3
        elif 'No Risk' in text or 'no risk' in text.lower():
            result['final_assessment']['risk_level'] = 'No Risk'
            result['final_assessment']['risk_score'] = 1
        
        return result
    
    def predict_batch(self,
                     student_narratives: List[str],
                     peer_contexts: Optional[List[str]] = None,
                     return_analysis: bool = False) -> List[Dict[str, Any]]:
        """
        Batch prediction
        
        Args:
            student_narratives: List of student behavior descriptions
            peer_contexts: List of peer comparison contexts (optional)
            return_analysis: Whether to return detailed analysis
            
        Returns:
            List of prediction results
        """
        if peer_contexts is None:
            peer_contexts = [""] * len(student_narratives)
        
        results = []
        for i, (narrative, peer_context) in enumerate(zip(student_narratives, peer_contexts)):
            logger.info(f"Processing student {i+1}/{len(student_narratives)}")
            result = self.predict(
                student_narrative=narrative,
                peer_context=peer_context,
                return_analysis=return_analysis
            )
            results.append(result)
        
        return results
    
    def get_risk_probability(self, result: Dict[str, Any]) -> float:
        """
        Extract submission probability from result (between 0-1)
        
        Args:
            result: Result returned by predict method
            
        Returns:
            Submission probability (0-1), where 1=likely to submit, 0=likely NOT to submit
        """
        # Try to get submission_probability from final_assessment
        submission_prob = result.get('final_assessment', {}).get('submission_probability', None)
        
        if submission_prob is not None:
            return float(submission_prob)
        
        # Fallback: if still using old format with risk_score (1-10)
        # Convert risk_score to submission_probability (invert the scale)
        risk_score = result.get('final_assessment', {}).get('risk_score', None)
        if risk_score is not None:
            # risk_score: 1=likely submit, 10=likely NOT submit
            # submission_probability: 1=likely submit, 0=likely NOT submit
            # So: submission_prob = (11 - risk_score) / 10
            return (11 - float(risk_score)) / 10.0
        
        # Default fallback
        return 0.5


class FewShotExampleSelector:
    """
    Few-shot example selector
    Used to select representative examples from training set
    """
    
    def __init__(self, 
                 strategy: str = "diverse",
                 num_examples: int = 3):
        """
        Initialize example selector
        
        Args:
            strategy: Selection strategy
                - "diverse": Diverse sampling, covering different risk levels
                - "similar": Similarity sampling, select most similar to target student
                - "balanced": Balanced sampling, select equal number from each risk level
            num_examples: Number of examples to select
        """
        self.strategy = strategy
        self.num_examples = num_examples
    
    def select_examples(self,
                       target_student: Dict[str, Any],
                       candidate_examples: List[Dict[str, Any]],
                       target_narrative: str = None) -> List[Dict[str, Any]]:
        """
        Select few-shot examples
        
        Args:
            target_student: Target student data
            candidate_examples: List of candidate examples
            target_narrative: Target student's behavior description (for similarity calculation)
            
        Returns:
            List of selected examples
        """
        if self.strategy == "diverse":
            return self._diverse_selection(candidate_examples)
        elif self.strategy == "similar":
            return self._similar_selection(target_student, candidate_examples, target_narrative)
        elif self.strategy == "balanced":
            return self._balanced_selection(candidate_examples)
        else:
            # Default random selection
            import random
            return random.sample(candidate_examples, min(self.num_examples, len(candidate_examples)))
    
    def _diverse_selection(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Diverse selection: Ensure coverage of different risk levels
        """
        # Group by risk level
        risk_groups = {
            'High Risk': [],
            'Medium Risk': [],
            'Low Risk': [],
            'No Risk': []
        }
        
        for example in candidates:
            risk_level = example.get('risk_level', 'Medium Risk')
            if risk_level in risk_groups:
                risk_groups[risk_level].append(example)
        
        # Select examples from each group
        selected = []
        examples_per_group = max(1, self.num_examples // len(risk_groups))
        
        for risk_level, examples in risk_groups.items():
            if examples:
                selected.extend(examples[:examples_per_group])
        
        # If not enough, randomly supplement
        if len(selected) < self.num_examples:
            remaining = [e for e in candidates if e not in selected]
            import random
            selected.extend(random.sample(remaining, min(self.num_examples - len(selected), len(remaining))))
        
        return selected[:self.num_examples]
    
    def _similar_selection(self,
                          target_student: Dict[str, Any],
                          candidates: List[Dict[str, Any]],
                          target_narrative: str = None) -> List[Dict[str, Any]]:
        """
        Similarity selection: Select examples most similar to target student
        """
        # Simple similarity calculation (can be improved)
        if target_narrative:
            # Based on text similarity
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            narratives = [target_narrative] + [e.get('student_narrative', '') for e in candidates]
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform(narratives)
            similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
            
            # Select most similar
            top_indices = similarities.argsort()[-self.num_examples:][::-1]
            return [candidates[i] for i in top_indices]
        else:
            # If no narrative, random selection
            import random
            return random.sample(candidates, min(self.num_examples, len(candidates)))
    
    def _balanced_selection(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Balanced selection: Select equal number from each risk level
        """
        risk_groups = {
            'High Risk': [],
            'Medium Risk': [],
            'Low Risk': [],
            'No Risk': []
        }
        
        for example in candidates:
            risk_level = example.get('risk_level', 'Medium Risk')
            if risk_level in risk_groups:
                risk_groups[risk_level].append(example)
        
        selected = []
        examples_per_group = max(1, self.num_examples // len(risk_groups))
        
        for risk_level, examples in risk_groups.items():
            if examples:
                import random
                selected.extend(random.sample(examples, min(examples_per_group, len(examples))))
        
        return selected[:self.num_examples]
    
    def select_from_train_dataframe(self,
                                     train_df: pd.DataFrame,
                                     label_column: str = 'submitted',
                                     behavior_converter=None,
                                     random_state: int = 42) -> List[Dict[str, Any]]:
        """
        Balanced extraction of few-shot examples from training DataFrame
        
        Args:
            train_df: Training DataFrame, must contain label_column
            label_column: Label column name (usually 'submitted', 0=not submitted/high risk, 1=submitted/low risk)
            behavior_converter: BehaviorToTextConverter instance, used to generate student behavior descriptions
            random_state: Random seed for reproducibility
            
        Returns:
            List of selected examples, each containing:
                - 'student_narrative': Student behavior description
                - 'risk_level': Risk level (based on label)
                - 'risk_score': Risk score (0-10)
                - 'label': Original label
        """
        import random
        random.seed(random_state)
        np.random.seed(random_state)
        
        # Group by label
        label_counts = train_df[label_column].value_counts()
        unique_labels = sorted(train_df[label_column].unique())
        
        # Calculate number of samples to select per class
        examples_per_label = max(1, self.num_examples // len(unique_labels))
        
        selected_examples = []
        
        for label in unique_labels:
            # Get all samples with this label
            label_data = train_df[train_df[label_column] == label].copy()
            
            # If insufficient samples in this class, select all; otherwise random selection
            n_samples = min(examples_per_label, len(label_data))
            if n_samples > 0:
                selected_indices = np.random.choice(
                    label_data.index, 
                    size=n_samples, 
                    replace=False
                )
                selected_data = label_data.loc[selected_indices]
                
                # Create example for each selected sample
                for idx, row in selected_data.iterrows():
                    # Generate student behavior description
                    if behavior_converter is not None:
                        try:
                            narrative = behavior_converter.convert_to_text(row)
                        except Exception as e:
                            logger.warning(f"Failed to convert row {idx} to text: {e}")
                            narrative = f"Student ID: {row.get('id_student', 'unknown')}"
                    else:
                        narrative = f"Student ID: {row.get('id_student', 'unknown')}"
                    
                    # Determine submission probability and analysis based on label and actual data
                    # 0 = not submitted, 1 = submitted
                    if label == 0:
                        risk_level = 'High Risk'
                        submission_probability = 0.15  # Very unlikely to submit
                        academic_findings = self._generate_academic_findings(row, submitted=False)
                        temporal_findings = self._generate_temporal_findings(row, submitted=False)
                        peer_findings = self._generate_peer_findings(row, submitted=False)
                        reasoning = "Multiple risk indicators suggest low submission likelihood"
                    else:
                        risk_level = 'Low Risk'
                        submission_probability = 0.85  # Very likely to submit
                        academic_findings = self._generate_academic_findings(row, submitted=True)
                        temporal_findings = self._generate_temporal_findings(row, submitted=True)
                        peer_findings = self._generate_peer_findings(row, submitted=True)
                        reasoning = "Strong engagement indicators suggest high submission likelihood"
                    
                    example = {
                        'student_narrative': narrative,
                        'risk_level': risk_level,
                        'submission_probability': submission_probability,
                        'academic_findings': academic_findings,
                        'temporal_findings': temporal_findings,
                        'peer_findings': peer_findings,
                        'reasoning': reasoning,
                        'risk_factors': self._extract_risk_factors(row, label),
                        'confidence': 'High',
                        'label': int(label),
                        'student_data': row.to_dict()  # Save original data for later use
                    }
                    selected_examples.append(example)
        
        # If insufficient selected samples, randomly supplement
        if len(selected_examples) < self.num_examples:
            # Get selected indices
            selected_idx_set = set()
            for e in selected_examples:
                if 'student_data' in e and 'id_student' in e['student_data']:
                    # Find corresponding DataFrame index
                    student_id = e['student_data']['id_student']
                    matching_idx = train_df[train_df.get('id_student', pd.Series()) == student_id].index
                    if len(matching_idx) > 0:
                        selected_idx_set.update(matching_idx)
            
            remaining_indices = train_df[~train_df.index.isin(selected_idx_set)].index
            if len(remaining_indices) > 0:
                n_additional = min(self.num_examples - len(selected_examples), len(remaining_indices))
                additional_indices = np.random.choice(remaining_indices, size=n_additional, replace=False)
                additional_data = train_df.loc[additional_indices]
                
                for idx, row in additional_data.iterrows():
                    if behavior_converter is not None:
                        try:
                            narrative = behavior_converter.convert_to_text(row)
                        except Exception as e:
                            narrative = f"Student ID: {row.get('id_student', 'unknown')}"
                    else:
                        narrative = f"Student ID: {row.get('id_student', 'unknown')}"
                    
                    label = int(row[label_column])
                    if label == 0:
                        risk_level = 'High Risk'
                        submission_probability = 0.15
                        academic_findings = self._generate_academic_findings(row, submitted=False)
                        temporal_findings = self._generate_temporal_findings(row, submitted=False)
                        peer_findings = self._generate_peer_findings(row, submitted=False)
                        reasoning = "Multiple risk indicators suggest low submission likelihood"
                    else:
                        risk_level = 'Low Risk'
                        submission_probability = 0.85
                        academic_findings = self._generate_academic_findings(row, submitted=True)
                        temporal_findings = self._generate_temporal_findings(row, submitted=True)
                        peer_findings = self._generate_peer_findings(row, submitted=True)
                        reasoning = "Strong engagement indicators suggest high submission likelihood"
                    
                    example = {
                        'student_narrative': narrative,
                        'risk_level': risk_level,
                        'submission_probability': submission_probability,
                        'academic_findings': academic_findings,
                        'temporal_findings': temporal_findings,
                        'peer_findings': peer_findings,
                        'reasoning': reasoning,
                        'risk_factors': self._extract_risk_factors(row, label),
                        'confidence': 'High',
                        'label': label,
                        'student_data': row.to_dict()
                    }
                    selected_examples.append(example)
        
        logger.info(f"Selected {len(selected_examples)} balanced examples from train set")
        logger.info(f"Label distribution: {pd.Series([e['label'] for e in selected_examples]).value_counts().to_dict()}")
        
        return selected_examples[:self.num_examples]
    
    def _generate_academic_findings(self, row: pd.Series, submitted: bool) -> str:
        """Generate academic/behavioral findings based on student data"""
        sum_click = row.get('sum_click', 0)
        num_days = row.get('num_days', 0)
        
        if submitted:
            if sum_click > 100:
                return f"High VLE engagement with {int(sum_click)} total clicks, {int(num_days)} active days showing consistent learning behavior"
            elif sum_click > 50:
                return f"Moderate VLE engagement with {int(sum_click)} total clicks, {int(num_days)} active days indicating regular participation"
            else:
                return f"Some VLE engagement with {int(sum_click)} total clicks, {int(num_days)} active days (lower but sufficient)"
        else:
            if sum_click < 10:
                return f"Very low VLE engagement with only {int(sum_click)} total clicks, {int(num_days)} active days indicating minimal participation"
            elif sum_click < 30:
                return f"Low VLE engagement with {int(sum_click)} total clicks, {int(num_days)} active days showing insufficient learning activity"
            else:
                return f"Moderate VLE clicks ({int(sum_click)}) but only {int(num_days)} active days, indicating inconsistent engagement"
    
    def _generate_temporal_findings(self, row: pd.Series, submitted: bool) -> str:
        """Generate temporal trend findings based on student data"""
        # Find day columns
        day_columns = sorted([col for col in row.index if col.startswith('day_') and col.endswith('_sum_click')])
        
        if not day_columns or len(day_columns) < 7:
            return "Insufficient temporal data for trend analysis"
        
        # Calculate recent vs early activity
        all_activities = [row.get(col, 0) for col in day_columns]
        recent_avg = sum(all_activities[-7:]) / 7 if len(all_activities) >= 7 else sum(all_activities) / len(all_activities)
        early_avg = sum(all_activities[:7]) / 7 if len(all_activities) >= 7 else recent_avg
        
        if submitted:
            if recent_avg > early_avg * 1.2:
                return f"Increasing engagement trend (recent: {recent_avg:.1f} vs early: {early_avg:.1f} clicks/day), positive trajectory"
            elif recent_avg > 5:
                return f"Stable engagement pattern (recent: {recent_avg:.1f} clicks/day), consistent learning behavior"
            else:
                return f"Moderate activity (recent: {recent_avg:.1f} clicks/day) with eventual submission"
        else:
            if recent_avg < early_avg * 0.7:
                return f"Declining engagement trend (recent: {recent_avg:.1f} vs early: {early_avg:.1f} clicks/day), warning sign"
            elif recent_avg < 2:
                return f"Very low recent activity ({recent_avg:.1f} clicks/day), insufficient engagement"
            else:
                return f"Inconsistent engagement pattern (recent: {recent_avg:.1f} clicks/day) leading to non-submission"
    
    def _generate_peer_findings(self, row: pd.Series, submitted: bool) -> str:
        """Generate peer comparison findings (simplified, since we don't have cohort stats here)"""
        sum_click = row.get('sum_click', 0)
        
        if submitted:
            if sum_click > 100:
                return "Above average engagement compared to typical successful students"
            elif sum_click > 50:
                return "Similar to typical successful students in the cohort"
            else:
                return "Below average but still sufficient for submission"
        else:
            if sum_click < 30:
                return "Significantly below average engagement, in bottom quartile of cohort"
            else:
                return "Below median engagement level, indicating relative risk"
    
    def _extract_risk_factors(self, row: pd.Series, label: int) -> List[str]:
        """Extract main risk factors from student data"""
        factors = []
        sum_click = row.get('sum_click', 0)
        num_days = row.get('num_days', 0)
        
        if label == 0:  # Not submitted
            if sum_click < 30:
                factors.append(f"Very low VLE activity ({int(sum_click)} clicks)")
            if num_days < 5:
                factors.append(f"Minimal active days ({int(num_days)} days)")
            
            # Check recent activity
            day_columns = sorted([col for col in row.index if col.startswith('day_') and col.endswith('_sum_click')])
            if day_columns and len(day_columns) >= 7:
                recent_clicks = sum([row.get(col, 0) for col in day_columns[-7:]])
                if recent_clicks < 10:
                    factors.append("No significant recent activity")
            
            if not factors:
                factors.append("Insufficient overall engagement")
        else:  # Submitted
            if sum_click > 100:
                factors.append(f"Strong VLE engagement ({int(sum_click)} clicks)")
            if num_days > 10:
                factors.append(f"High number of active days ({int(num_days)} days)")
            if not factors:
                factors.append("Consistent learning behavior")
        
        return factors[:3]  # Return top 3 factors

