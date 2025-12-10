"""
Improved Prompt templates for few-shot learning experiments
Key improvements:
1. Structured feature grouping (Academic, Family, Lifestyle)
2. Chain-of-Thought reasoning guidance
3. Clear output format requirements
4. Emphasis on important features
5. Better example formatting
"""
from typing import Dict, List, Optional, Tuple
import pandas as pd


class PromptTemplate:
    """Improved template for building few-shot prompts with better structure"""
    
    # Value mappings for categorical features
    VALUE_MAPPINGS = {
        'school': {'GP': 'Gabriel Pereira', 'MS': 'Mousinho da Silveira'},
        'sex': {'F': 'Female', 'M': 'Male'},
        'address': {'R': 'Rural', 'U': 'Urban'},
        'famsize': {'GT3': 'Greater than 3', 'LE3': 'Less or equal to 3'},
        'Pstatus': {'A': 'Apart', 'T': 'Together'},
        'Mjob': {
            'at_home': 'at home',
            'health': 'health care related',
            'other': 'other',
            'services': 'services',
            'teacher': 'teacher'
        },
        'Fjob': {
            'at_home': 'at home',
            'health': 'health care related',
            'other': 'other',
            'services': 'services',
            'teacher': 'teacher'
        },
        'reason': {
            'course': 'course preference',
            'home': 'close to home',
            'other': 'other',
            'reputation': 'school reputation'
        },
        'guardian': {'father': 'father', 'mother': 'mother', 'other': 'other'},
        'schoolsup': {'no': 'no', 'yes': 'yes'},
        'famsup': {'no': 'no', 'yes': 'yes'},
        'paid': {'no': 'no', 'yes': 'yes'},
        'activities': {'no': 'no', 'yes': 'yes'},
        'nursery': {'no': 'no', 'yes': 'yes'},
        'higher': {'no': 'no', 'yes': 'yes'},
        'internet': {'no': 'no', 'yes': 'yes'},
        'romantic': {'no': 'no', 'yes': 'yes'},
    }
    
    # Education level mapping
    EDU_LEVELS = {
        0: 'none',
        1: 'primary education (4th grade)',
        2: '5th to 9th grade',
        3: 'secondary education',
        4: 'higher education'
    }
    
    # Scale descriptions
    SCALE_1_4 = {1: 'very low', 2: 'low', 3: 'medium', 4: 'high'}
    SCALE_1_5 = {1: 'very low', 2: 'low', 3: 'medium', 4: 'high', 5: 'very high'}
    
    def __init__(self, use_cot: bool = True, output_cot: bool = False, selected_features: Optional[List[str]] = None):
        """
        Initialize improved prompt template
        
        Args:
            use_cot: Whether to use Chain-of-Thought reasoning
            output_cot: Whether to output Chain-of-Thought reasoning in JSON format
            selected_features: List of selected feature names to include in prompt (None = all features)
        """
        self.use_cot = use_cot
        self.output_cot = output_cot
        self.selected_features = selected_features
    
    def student_to_structured_text(self, student: Dict, setup: str = 'A') -> str:
        """
        Convert student data to structured natural language description
        Groups features by category for better understanding
        
        Args:
            student: Student data dictionary
            setup: 'A' (include G1, G2) or 'C' (exclude G1, G2, G3)
            
        Returns:
            Structured natural language description of the student
        """
        # Filter student dict if feature selection is enabled
        if self.selected_features is not None:
            student = {k: v for k, v in student.items() if k in self.selected_features}
        
        lines = []
        
        # === ACADEMIC PERFORMANCE & HISTORY ===
        academic_features = []
        if setup == 'A':
            if 'G1' in student:
                academic_features.append(f"  • First period grade (G1): {student.get('G1', 'N/A')}/20")
            if 'G2' in student:
                academic_features.append(f"  • Second period grade (G2): {student.get('G2', 'N/A')}/20")
        if 'failures' in student:
            academic_features.append(f"  • Past class failures: {student.get('failures', 0)}")
        if 'studytime' in student:
            academic_features.append(f"  • Weekly study time: {self._get_scale_1_4(student.get('studytime', 2))}")
        if 'absences' in student:
            academic_features.append(f"  • School absences: {student.get('absences', 0)}")
        if 'higher' in student:
            academic_features.append(f"  • Wants higher education: {self._get_value('higher', student.get('higher', 'no'))}")
        if 'paid' in student:
            academic_features.append(f"  • Extra paid classes: {self._get_value('paid', student.get('paid', 'no'))}")
        if 'schoolsup' in student:
            academic_features.append(f"  • School support: {self._get_value('schoolsup', student.get('schoolsup', 'no'))}")
        
        if academic_features:
            lines.append("## Academic Performance & History")
            lines.extend(academic_features)
            lines.append("")
        
        # === FAMILY BACKGROUND ===
        family_features = []
        if 'Medu' in student:
            family_features.append(f"  • Mother's education: {self._get_edu_level(student.get('Medu', 0))}")
        if 'Fedu' in student:
            family_features.append(f"  • Father's education: {self._get_edu_level(student.get('Fedu', 0))}")
        if 'Mjob' in student:
            family_features.append(f"  • Mother's job: {self._get_value('Mjob', student.get('Mjob', ''))}")
        if 'Fjob' in student:
            family_features.append(f"  • Father's job: {self._get_value('Fjob', student.get('Fjob', ''))}")
        if 'famsize' in student:
            family_features.append(f"  • Family size: {self._get_value('famsize', student.get('famsize', ''))}")
        if 'Pstatus' in student:
            family_features.append(f"  • Parent cohabitation: {self._get_value('Pstatus', student.get('Pstatus', ''))}")
        if 'guardian' in student:
            family_features.append(f"  • Guardian: {self._get_value('guardian', student.get('guardian', ''))}")
        if 'famrel' in student:
            family_features.append(f"  • Family relationship quality: {self._get_scale_1_5(student.get('famrel', 3))}")
        if 'famsup' in student:
            family_features.append(f"  • Family educational support: {self._get_value('famsup', student.get('famsup', 'no'))}")
        
        if family_features:
            lines.append("## Family Background")
            lines.extend(family_features)
            lines.append("")
        
        # === LIFESTYLE & SOCIAL FACTORS ===
        lifestyle_features = []
        if 'age' in student:
            lifestyle_features.append(f"  • Age: {student.get('age', 'N/A')} years old")
        if 'sex' in student:
            lifestyle_features.append(f"  • Gender: {self._get_value('sex', student.get('sex', ''))}")
        if 'address' in student:
            lifestyle_features.append(f"  • Address: {self._get_value('address', student.get('address', ''))}")
        if 'traveltime' in student:
            lifestyle_features.append(f"  • Travel time to school: {self._get_scale_1_4(student.get('traveltime', 2))}")
        if 'internet' in student:
            lifestyle_features.append(f"  • Internet access: {self._get_value('internet', student.get('internet', 'no'))}")
        if 'romantic' in student:
            lifestyle_features.append(f"  • In romantic relationship: {self._get_value('romantic', student.get('romantic', 'no'))}")
        if 'freetime' in student:
            lifestyle_features.append(f"  • Free time after school: {self._get_scale_1_5(student.get('freetime', 3))}")
        if 'goout' in student:
            lifestyle_features.append(f"  • Going out with friends: {self._get_scale_1_5(student.get('goout', 3))}")
        if 'Dalc' in student:
            lifestyle_features.append(f"  • Workday alcohol consumption: {self._get_scale_1_5(student.get('Dalc', 1))}")
        if 'Walc' in student:
            lifestyle_features.append(f"  • Weekend alcohol consumption: {self._get_scale_1_5(student.get('Walc', 1))}")
        if 'health' in student:
            lifestyle_features.append(f"  • Health status: {self._get_scale_1_5(student.get('health', 3))}")
        if 'activities' in student:
            lifestyle_features.append(f"  • Extra-curricular activities: {self._get_value('activities', student.get('activities', 'no'))}")
        
        if lifestyle_features:
            lines.append("## Lifestyle & Social Factors")
            lines.extend(lifestyle_features)
        
        return '\n'.join(lines)
    
    def _get_value(self, feature: str, value) -> str:
        """Get human-readable value for categorical feature"""
        value_str = str(value).strip()
        if feature in self.VALUE_MAPPINGS and value_str in self.VALUE_MAPPINGS[feature]:
            return self.VALUE_MAPPINGS[feature][value_str]
        return value_str
    
    def _get_edu_level(self, level) -> str:
        """Get education level description"""
        try:
            level_int = int(float(level))
            return self.EDU_LEVELS.get(level_int, f'level {level_int}')
        except (ValueError, TypeError):
            return str(level)
    
    def _get_scale_1_4(self, value) -> str:
        """Get scale description for 1-4 scale"""
        try:
            value_int = int(float(value))
            return self.SCALE_1_4.get(value_int, str(value_int))
        except (ValueError, TypeError):
            return str(value)
    
    def _get_scale_1_5(self, value) -> str:
        """Get scale description for 1-5 scale"""
        try:
            value_int = int(float(value))
            return self.SCALE_1_5.get(value_int, str(value_int))
        except (ValueError, TypeError):
            return str(value)
    
    def _extract_json_from_response(self, response: str) -> Optional[dict]:
        """Extract JSON object from response, handling markdown code blocks"""
        import json
        try:
            json_str = response.strip()
            if '```json' in response:
                start = response.find('```json') + 7
                end = response.find('```', start)
                if end != -1:
                    json_str = response[start:end].strip()
            elif '```' in response:
                start = response.find('```') + 3
                end = response.find('```', start)
                if end != -1:
                    json_str = response[start:end].strip()
            
            data = json.loads(json_str)
            return data if isinstance(data, dict) else None
        except (json.JSONDecodeError, ValueError, AttributeError):
            return None
    
    def build_classification_prompt(
        self,
        examples: List[Dict],
        test_student: Dict,
        setup: str = 'A'
    ) -> str:
        """
        Build improved few-shot classification prompt with Chain-of-Thought
        
        Args:
            examples: List of example students with their labels (pass/fail)
            test_student: Test student to predict
            setup: 'A' or 'C'
            
        Returns:
            Complete prompt string
        """
        prompt_parts = []
        
        # Task description with context
        prompt_parts.append("=" * 80)
        prompt_parts.append("TASK: Predict Student Pass/Fail Status")
        prompt_parts.append("=" * 80)
        prompt_parts.append("")
        prompt_parts.append("You are an expert educational analyst. Your task is to predict whether a Portuguese")
        prompt_parts.append("secondary school student will PASS or FAIL based on their characteristics.")
        prompt_parts.append("")
        prompt_parts.append("CRITERIA:")
        prompt_parts.append("  • PASS: Final grade (G3) ≥ 10 out of 20")
        prompt_parts.append("  • FAIL: Final grade (G3) < 10 out of 20")
        prompt_parts.append("")
        prompt_parts.append("IMPORTANT FACTORS (in order of importance):")
        if setup == 'A':
            prompt_parts.append("  1. Prior grades (G1, G2) - strongest predictor")
            prompt_parts.append("  2. Past failures - indicates academic struggles")
            prompt_parts.append("  3. Study time - reflects effort and commitment")
            prompt_parts.append("  4. Family education background - correlates with support")
            prompt_parts.append("  5. Absences - indicates engagement")
        else:
            prompt_parts.append("  1. Past failures - indicates academic struggles")
            prompt_parts.append("  2. Study time - reflects effort and commitment")
            prompt_parts.append("  3. Family education background - correlates with support")
            prompt_parts.append("  4. Absences - indicates engagement")
            prompt_parts.append("  5. Family support and resources")
        prompt_parts.append("")
        
        # Few-shot examples with reasoning
        prompt_parts.append("=" * 80)
        prompt_parts.append("FEW-SHOT EXAMPLES")
        prompt_parts.append("=" * 80)
        prompt_parts.append("")
        
        for i, example in enumerate(examples, 1):
            student_data = example['student']
            label = example['label']
            label_text = "PASS" if label == 1 else "FAIL"
            
            prompt_parts.append(f"--- Example {i} ---")
            prompt_parts.append(self.student_to_structured_text(student_data, setup=setup))
            
            if self.use_cot:
                # Add reasoning for the example
                g1 = student_data.get('G1', None)
                g2 = student_data.get('G2', None)
                failures = student_data.get('failures', 0)
                studytime = student_data.get('studytime', 2)
                
                reasoning = []
                if setup == 'A' and g2 is not None:
                    reasoning.append(f"Recent grade (G2={g2}) is {'strong' if g2 >= 10 else 'weak'}")
                if failures > 0:
                    reasoning.append(f"Has {failures} past failure(s) - concerning")
                if studytime >= 3:
                    reasoning.append("High study time - positive indicator")
                elif studytime <= 1:
                    reasoning.append("Low study time - negative indicator")
                
                if reasoning:
                    prompt_parts.append(f"\nReasoning: {'; '.join(reasoning)}")
            
            prompt_parts.append(f"\nPrediction: {label_text}")
            prompt_parts.append("")
        
        # Test case
        prompt_parts.append("=" * 80)
        prompt_parts.append("NOW PREDICT THIS STUDENT")
        prompt_parts.append("=" * 80)
        prompt_parts.append("")
        prompt_parts.append(self.student_to_structured_text(test_student, setup=setup))
        prompt_parts.append("")
        
        if self.use_cot:
            prompt_parts.append("Think step by step:")
            prompt_parts.append("1. Analyze academic indicators (grades, failures, study time)")
            prompt_parts.append("2. Consider family background and support")
            prompt_parts.append("3. Evaluate lifestyle factors")
            prompt_parts.append("4. Compare with examples above")
            prompt_parts.append("5. Make your prediction")
            prompt_parts.append("")
        
        if self.output_cot:
            # JSON format output with COT reasoning
            prompt_parts.append("OUTPUT FORMAT:")
            prompt_parts.append("  You must output a valid JSON object with the following structure:")
            prompt_parts.append("  {")
            prompt_parts.append("    \"reasoning\": \"Your step-by-step reasoning process\",")
            prompt_parts.append("    \"prediction\": \"PASS\" or \"FAIL\"")
            prompt_parts.append("  }")
            prompt_parts.append("")
            prompt_parts.append("  Example:")
            prompt_parts.append("  {")
            prompt_parts.append("    \"reasoning\": \"The student has G2=15 which is strong. They have 0 failures and high study time. These factors suggest they will PASS.\",")
            prompt_parts.append("    \"prediction\": \"PASS\"")
            prompt_parts.append("  }")
            prompt_parts.append("")
            prompt_parts.append("  Output your response as JSON:")
        else:
            prompt_parts.append("OUTPUT FORMAT:")
            prompt_parts.append("  Your response must be EXACTLY one of: PASS or FAIL")
            prompt_parts.append("  Do not include any explanation, just the word PASS or FAIL")
            prompt_parts.append("")
            prompt_parts.append("Prediction: ")
        
        return '\n'.join(prompt_parts)
    
    def build_regression_prompt(
        self,
        examples: List[Dict],
        test_student: Dict,
        setup: str = 'A'
    ) -> str:
        """
        Build improved few-shot regression prompt with Chain-of-Thought
        
        Args:
            examples: List of example students with their G3 grades
            test_student: Test student to predict
            setup: 'A' or 'C'
            
        Returns:
            Complete prompt string
        """
        prompt_parts = []
        
        # Task description with context
        prompt_parts.append("=" * 80)
        prompt_parts.append("TASK: Predict Final Grade (G3)")
        prompt_parts.append("=" * 80)
        prompt_parts.append("")
        prompt_parts.append("You are an expert educational analyst. Your task is to predict the final grade (G3)")
        prompt_parts.append("of a Portuguese secondary school student based on their characteristics.")
        prompt_parts.append("")
        prompt_parts.append("GRADE SCALE:")
        prompt_parts.append("  • Range: 0 to 20 (0 = lowest, 20 = highest)")
        prompt_parts.append("  • Passing threshold: 10/20")
        prompt_parts.append("")
        prompt_parts.append("IMPORTANT FACTORS (in order of importance):")
        if setup == 'A':
            prompt_parts.append("  1. Prior grades (G1, G2) - strongest predictor")
            prompt_parts.append("  2. Past failures - indicates academic struggles")
            prompt_parts.append("  3. Study time - reflects effort and commitment")
            prompt_parts.append("  4. Family education background - correlates with support")
            prompt_parts.append("  5. Absences - indicates engagement")
        else:
            prompt_parts.append("  1. Past failures - indicates academic struggles")
            prompt_parts.append("  2. Study time - reflects effort and commitment")
            prompt_parts.append("  3. Family education background - correlates with support")
            prompt_parts.append("  4. Absences - indicates engagement")
            prompt_parts.append("  5. Family support and resources")
        prompt_parts.append("")
        
        # Few-shot examples with reasoning
        prompt_parts.append("=" * 80)
        prompt_parts.append("FEW-SHOT EXAMPLES")
        prompt_parts.append("=" * 80)
        prompt_parts.append("")
        
        for i, example in enumerate(examples, 1):
            student_data = example['student']
            g3 = example['g3']
            
            prompt_parts.append(f"--- Example {i} ---")
            prompt_parts.append(self.student_to_structured_text(student_data, setup=setup))
            
            if self.use_cot:
                # Add reasoning for the example
                g1 = student_data.get('G1', None)
                g2 = student_data.get('G2', None)
                failures = student_data.get('failures', 0)
                studytime = student_data.get('studytime', 2)
                
                reasoning = []
                if setup == 'A' and g2 is not None:
                    reasoning.append(f"G2={g2}, predicted G3={g3} (difference: {g3-g2:.1f})")
                if failures > 0:
                    reasoning.append(f"{failures} failure(s) likely reduced grade")
                if studytime >= 3:
                    reasoning.append("High study time supports good grade")
                
                if reasoning:
                    prompt_parts.append(f"\nReasoning: {'; '.join(reasoning)}")
            
            prompt_parts.append(f"\nFinal grade (G3): {g3}/20")
            prompt_parts.append("")
        
        # Test case
        prompt_parts.append("=" * 80)
        prompt_parts.append("NOW PREDICT THIS STUDENT")
        prompt_parts.append("=" * 80)
        prompt_parts.append("")
        prompt_parts.append(self.student_to_structured_text(test_student, setup=setup))
        prompt_parts.append("")
        
        if self.use_cot:
            prompt_parts.append("Think step by step:")
            prompt_parts.append("1. Analyze academic indicators (grades, failures, study time)")
            prompt_parts.append("2. Consider family background and support")
            prompt_parts.append("3. Evaluate lifestyle factors")
            prompt_parts.append("4. Compare with examples above")
            prompt_parts.append("5. Estimate the grade (0-20)")
            prompt_parts.append("")
        
        if self.output_cot:
            # JSON format output with COT reasoning
            prompt_parts.append("OUTPUT FORMAT:")
            prompt_parts.append("  You must output a valid JSON object with the following structure:")
            prompt_parts.append("  {")
            prompt_parts.append("    \"reasoning\": \"Your step-by-step reasoning process\",")
            prompt_parts.append("    \"prediction\": <number between 0 and 20>")
            prompt_parts.append("  }")
            prompt_parts.append("")
            prompt_parts.append("  Example:")
            prompt_parts.append("  {")
            prompt_parts.append("    \"reasoning\": \"The student has G2=15 which is strong. They have 0 failures and high study time. Based on similar examples, I estimate G3 around 15.\",")
            prompt_parts.append("    \"prediction\": 15.0")
            prompt_parts.append("  }")
            prompt_parts.append("")
            prompt_parts.append("  Output your response as JSON:")
        else:
            prompt_parts.append("OUTPUT FORMAT:")
            prompt_parts.append("  Your response must be a NUMBER between 0 and 20")
            prompt_parts.append("  Format: Just the number, e.g., '15' or '12.5'")
            prompt_parts.append("  Do not include '/20' or any other text")
            prompt_parts.append("")
            prompt_parts.append("Final grade (G3): ")
        
        return '\n'.join(prompt_parts)
    
    def parse_classification_response(self, response: str) -> Optional[int]:
        """
        Parse LLM response for classification (improved)
        Supports both plain text and JSON format
        
        Args:
            response: LLM response text
            
        Returns:
            1 for pass, 0 for fail, None if cannot parse
        """
        response = response.strip()
        
        # Try to parse as JSON first (if output_cot is enabled)
        if self.output_cot:
            data = self._extract_json_from_response(response)
            if data:
                prediction = data.get('prediction', '')
                prediction_upper = str(prediction).strip().upper()
                if prediction_upper == 'PASS' or prediction_upper == '1':
                    return 1
                elif prediction_upper == 'FAIL' or prediction_upper == '0':
                    return 0
        
        # Fall back to text parsing
        response_upper = response.upper()
        
        # Direct match
        if response_upper == 'PASS':
            return 1
        if response_upper == 'FAIL':
            return 0
        
        # Check for pass indicators
        if 'PASS' in response_upper:
            return 1
        
        # Check for fail indicators
        if 'FAIL' in response_upper:
            return 0
        
        # Check for pass/fail in text
        if any(word in response_upper for word in ['PASS', '1', 'YES', 'TRUE']):
            if 'FAIL' not in response_upper and 'NO PASS' not in response_upper:
                return 1
        
        if any(word in response_upper for word in ['FAIL', '0', 'NO', 'FALSE']):
            return 0
        
        return None
    
    def parse_classification_response_with_cot(self, response: str) -> Tuple[Optional[int], Optional[str]]:
        """
        Parse LLM response for classification and extract COT reasoning
        
        Args:
            response: LLM response text
            
        Returns:
            Tuple of (prediction, reasoning)
            prediction: 1 for pass, 0 for fail, None if cannot parse
            reasoning: COT reasoning text, None if not available
        """
        reasoning = None
        
        # Try to parse as JSON first
        if self.output_cot:
            data = self._extract_json_from_response(response)
            if data:
                prediction = data.get('prediction', '')
                reasoning = data.get('reasoning', None)
                prediction_upper = str(prediction).strip().upper()
                if prediction_upper == 'PASS' or prediction_upper == '1':
                    return 1, reasoning
                elif prediction_upper == 'FAIL' or prediction_upper == '0':
                    return 0, reasoning
        
        # Fall back to text parsing
        pred = self.parse_classification_response(response)
        return pred, reasoning
    
    def parse_regression_response(self, response: str) -> Optional[float]:
        """
        Parse LLM response for regression (improved)
        Supports both plain text and JSON format
        
        Args:
            response: LLM response text
            
        Returns:
            G3 grade value (0-20), None if cannot parse
        """
        response = response.strip()
        
        # Try to parse as JSON first (if output_cot is enabled)
        if self.output_cot:
            data = self._extract_json_from_response(response)
            if data:
                prediction = data.get('prediction', None)
                if prediction is not None:
                    try:
                        value = float(prediction)
                        return max(0, min(20, value))  # Clamp to [0, 20]
                    except (ValueError, TypeError):
                        pass
        
        # Fall back to text parsing
        import re
        patterns = [
            r'\b(\d+\.?\d*)\s*/?\s*20\b',  # "15/20" or "15 / 20"
            r'\b(\d+\.?\d*)\b',  # Any number
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                try:
                    value = float(matches[0])
                    return max(0, min(20, value))  # Clamp to [0, 20]
                except ValueError:
                    continue
        
        return None
    
    def parse_regression_response_with_cot(self, response: str) -> Tuple[Optional[float], Optional[str]]:
        """
        Parse LLM response for regression and extract COT reasoning
        
        Args:
            response: LLM response text
            
        Returns:
            Tuple of (prediction, reasoning)
            prediction: G3 grade value (0-20), None if cannot parse
            reasoning: COT reasoning text, None if not available
        """
        reasoning = None
        
        # Try to parse as JSON first
        if self.output_cot:
            data = self._extract_json_from_response(response)
            if data:
                prediction = data.get('prediction', None)
                reasoning = data.get('reasoning', None)
                if prediction is not None:
                    try:
                        value = float(prediction)
                        return max(0, min(20, value)), reasoning
                    except (ValueError, TypeError):
                        pass
        
        # Fall back to text parsing
        pred = self.parse_regression_response(response)
        return pred, reasoning

