"""
Few-shot prompt-based predictor using LLM
Combines all features from improved and enhanced versions:
1. Better example selection (similarity-based, diverse, hybrid)
2. Improved prompt templates with Chain-of-Thought
3. Self-Consistency: Multiple sampling and voting
4. Dynamic example selection: Adaptive number of examples
5. Temperature scheduling: Adaptive temperature based on difficulty
6. Error correction: Second-chance prediction for inconsistent results
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

import sys
from pathlib import Path

# Add llm_data_generation to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'llm_data_generation'))
from llm_client import OpenRouterClient

# Import prompt template
try:
    from .prompt_templates import PromptTemplate
except ImportError:
    from prompt_templates import PromptTemplate

# Import feature selector
try:
    from .feature_selector import FeatureSelector
except ImportError:
    from feature_selector import FeatureSelector

logger = logging.getLogger(__name__)


class PromptPredictor:
    """Few-shot prompt-based predictor with all enhanced features"""
    
    def __init__(
        self,
        llm_client: OpenRouterClient,
        n_examples: int = 5,
        temperature: float = 0.3,
        use_cot: bool = True,
        output_cot: bool = False,
        example_strategy: str = 'hybrid',
        use_feature_selection: bool = False,
        feature_selection_model: str = 'rf',
        n_top_features: int = 10,
        # Enhanced features
        use_self_consistency: bool = False,
        n_consistency_samples: int = 5,
        consistency_temperature: float = 0.7,
        use_dynamic_examples: bool = False,
        min_examples: int = 5,
        max_examples: int = 10,
        use_adaptive_temperature: bool = False,
        min_temperature: float = 0.1,
        max_temperature: float = 0.7,
        use_error_correction: bool = False
    ):
        """
        Initialize prompt predictor
        
        Args:
            llm_client: LLM client for API calls
            n_examples: Number of few-shot examples to use
            temperature: Sampling temperature for LLM (lower = more deterministic)
            use_cot: Whether to use Chain-of-Thought reasoning
            output_cot: Whether to output Chain-of-Thought reasoning in JSON format
            example_strategy: 'random', 'balanced', 'similarity', 'diverse', or 'hybrid'
            use_feature_selection: Whether to use feature selection based on importance
            feature_selection_model: 'rf' for Random Forest, 'xgb' for XGBoost
            n_top_features: Number of top features to select (top N)
            use_self_consistency: Whether to use self-consistency (multiple sampling)
            n_consistency_samples: Number of samples for self-consistency (3-7 recommended)
            consistency_temperature: Temperature for consistency sampling (0.5-0.8 recommended)
            use_dynamic_examples: Whether to dynamically adjust example count
            min_examples: Minimum number of examples
            max_examples: Maximum number of examples
            use_adaptive_temperature: Whether to adaptively adjust temperature
            min_temperature: Minimum temperature
            max_temperature: Maximum temperature
            use_error_correction: Whether to use error correction
        """
        self.llm_client = llm_client
        self.n_examples = n_examples
        self.template = PromptTemplate(use_cot=use_cot, output_cot=output_cot)
        self.temperature = temperature
        self.use_cot = use_cot
        self.output_cot = output_cot
        self.example_strategy = example_strategy
        self.use_feature_selection = use_feature_selection
        self.feature_selection_model = feature_selection_model
        self.n_top_features = n_top_features
        self.feature_selector = None
        
        # Enhanced features
        self.use_self_consistency = use_self_consistency
        self.n_consistency_samples = n_consistency_samples
        self.consistency_temperature = consistency_temperature
        self.use_dynamic_examples = use_dynamic_examples
        self.min_examples = min_examples
        self.max_examples = max_examples
        self.use_adaptive_temperature = use_adaptive_temperature
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.use_error_correction = use_error_correction
    
    def _get_max_tokens(self) -> int:
        """Get appropriate max_tokens based on output_cot setting"""
        if self.output_cot:
            return 1200
        else:
            return 100
    
    def select_examples(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        test_student: Optional[Dict] = None,
        task: str = 'classification',
        strategy: Optional[str] = None
    ) -> List[Dict]:
        """
        Select few-shot examples from training data with improved strategies
        
        Args:
            X_train: Training feature DataFrame
            y_train: Training target Series
            test_student: Test student dict (for similarity-based selection)
            task: 'classification' or 'regression'
            strategy: Override self.example_strategy if provided
            
        Returns:
            List of example dictionaries with 'student' and 'label'/'g3' keys
        """
        strategy = strategy or self.example_strategy
        n_examples = min(self.n_examples, len(X_train))
        
        # Helper function to get numeric columns only (for similarity calculations)
        def get_numeric_data(df):
            """Extract only numeric columns from DataFrame"""
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return df
            return df[numeric_cols]
        
        if strategy == 'random':
            indices = np.random.choice(len(X_train), size=n_examples, replace=False)
            
        elif strategy == 'balanced' and task == 'classification':
            pass_indices = np.where(y_train == 1)[0]
            fail_indices = np.where(y_train == 0)[0]
            n_per_class = n_examples // 2
            indices = np.concatenate([
                np.random.choice(pass_indices, size=min(n_per_class, len(pass_indices)), replace=False),
                np.random.choice(fail_indices, size=min(n_per_class, len(fail_indices)), replace=False)
            ])
            if len(indices) < n_examples:
                remaining = n_examples - len(indices)
                remaining_indices = np.random.choice(
                    np.setdiff1d(np.arange(len(X_train)), indices),
                    size=remaining,
                    replace=False
                )
                indices = np.concatenate([indices, remaining_indices])
        
        elif strategy == 'similarity' and test_student is not None:
            X_train_numeric = get_numeric_data(X_train)
            test_vector_numeric = pd.Series(test_student).reindex(X_train_numeric.columns, fill_value=0)
            test_vector_numeric = test_vector_numeric.values.reshape(1, -1)
            X_train_array = X_train_numeric.values
            similarities = cosine_similarity(test_vector_numeric, X_train_array)[0]
            top_k = min(n_examples * 2, len(X_train))
            candidate_indices = np.argsort(similarities)[-top_k:][::-1]
            
            if task == 'classification':
                pass_candidates = [i for i in candidate_indices if y_train.iloc[i] == 1]
                fail_candidates = [i for i in candidate_indices if y_train.iloc[i] == 0]
                n_per_class = n_examples // 2
                indices = []
                indices.extend(pass_candidates[:n_per_class])
                indices.extend(fail_candidates[:n_per_class])
                if len(indices) < n_examples:
                    remaining = [i for i in candidate_indices if i not in indices]
                    indices.extend(remaining[:n_examples - len(indices)])
                indices = np.array(indices[:n_examples])
            else:
                indices = candidate_indices[:n_examples]
        
        elif strategy == 'diverse':
            X_train_numeric = get_numeric_data(X_train)
            X_train_array = X_train_numeric.values
            indices = [np.random.choice(len(X_train))]
            for _ in range(n_examples - 1):
                selected_vectors = X_train_array[indices]
                similarities = cosine_similarity(X_train_array, selected_vectors).max(axis=1)
                remaining_indices = np.setdiff1d(np.arange(len(X_train)), indices)
                if len(remaining_indices) == 0:
                    break
                next_idx = remaining_indices[np.argmin(similarities[remaining_indices])]
                indices.append(next_idx)
            indices = np.array(indices)
        
        elif strategy == 'hybrid':
            if test_student is not None:
                X_train_numeric = get_numeric_data(X_train)
                test_vector_numeric = pd.Series(test_student).reindex(X_train_numeric.columns, fill_value=0)
                test_vector_numeric = test_vector_numeric.values.reshape(1, -1)
                X_train_array = X_train_numeric.values
                similarities = cosine_similarity(test_vector_numeric, X_train_array)[0]
                top_k = min(n_examples * 3, len(X_train))
                candidate_indices = np.argsort(similarities)[-top_k:][::-1]
                indices = [candidate_indices[0]]
                candidate_vectors = X_train_array[candidate_indices]
                for _ in range(n_examples - 1):
                    selected_vectors = X_train_array[indices]
                    similarities_to_selected = cosine_similarity(
                        candidate_vectors, selected_vectors
                    ).max(axis=1)
                    remaining_candidates = np.setdiff1d(
                        np.arange(len(candidate_indices)),
                        [np.where(candidate_indices == idx)[0][0] for idx in indices]
                    )
                    if len(remaining_candidates) == 0:
                        break
                    next_candidate_idx = remaining_candidates[
                        np.argmin(similarities_to_selected[remaining_candidates])
                    ]
                    indices.append(candidate_indices[next_candidate_idx])
                indices = np.array(indices)
            else:
                indices = self.select_examples(X_train, y_train, None, task, 'diverse')
        
        else:
            indices = np.random.choice(len(X_train), size=n_examples, replace=False)
        
        examples = []
        for idx in indices:
            student_dict = X_train.iloc[idx].to_dict()
            if task == 'classification':
                examples.append({
                    'student': student_dict,
                    'label': int(y_train.iloc[idx])
                })
            else:
                examples.append({
                    'student': student_dict,
                    'g3': float(y_train.iloc[idx])
                })
        
        return examples
    
    def _calculate_difficulty_score(
        self,
        test_student: Dict,
        X_train: pd.DataFrame,
        y_train: Optional[pd.Series] = None
    ) -> float:
        """Calculate difficulty score for a test student (0-1, higher = more difficult)"""
        def get_numeric_data(df):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return df
            return df[numeric_cols]
        
        X_train_numeric = get_numeric_data(X_train)
        test_vector = pd.Series(test_student).reindex(X_train_numeric.columns, fill_value=0)
        test_vector = test_vector.values.reshape(1, -1)
        X_train_array = X_train_numeric.values
        similarities = cosine_similarity(test_vector, X_train_array)[0]
        avg_similarity = np.mean(similarities)
        max_similarity = np.max(similarities)
        similarity_std = np.std(similarities)
        difficulty = (
            0.4 * (1 - avg_similarity) +
            0.3 * (1 - max_similarity) +
            0.3 * min(1.0, similarity_std * 2)
        )
        return max(0.0, min(1.0, difficulty))
    
    def _get_adaptive_temperature(self, difficulty: float) -> float:
        """Get adaptive temperature based on difficulty"""
        if not self.use_adaptive_temperature:
            return self.temperature
        return self.min_temperature + (self.max_temperature - self.min_temperature) * difficulty
    
    def _get_dynamic_n_examples(self, difficulty: float) -> int:
        """Get dynamic number of examples based on difficulty"""
        if not self.use_dynamic_examples:
            return self.n_examples
        n_examples = int(
            self.min_examples + 
            (self.max_examples - self.min_examples) * difficulty
        )
        return max(self.min_examples, min(self.max_examples, n_examples))
    
    def _predict_with_self_consistency(
        self,
        prompt: str,
        system_prompt: str,
        task: str = 'classification',
        examples: Optional[List[Dict]] = None
    ) -> Tuple[any, List[str], List[any]]:
        """Predict with self-consistency (multiple sampling and voting)"""
        parse_func = self.template.parse_classification_response if task == 'classification' else self.template.parse_regression_response
        
        if not self.use_self_consistency:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=self.temperature,
                max_tokens=self._get_max_tokens()
            )
            pred = parse_func(response) if response else None
            return pred, [response] if response else [], [pred] if pred is not None else []
        
        all_responses = []
        all_predictions = []
        for _ in range(self.n_consistency_samples):
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=self.consistency_temperature,
                max_tokens=self._get_max_tokens()
            )
            if response:
                all_responses.append(response)
                pred = parse_func(response)
                if pred is not None:
                    all_predictions.append(pred)
        
        if not all_predictions:
            logger.warning("All self-consistency predictions failed, using fallback")
            if task == 'classification' and examples:
                fallback = int(np.mean([ex['label'] for ex in examples]) >= 0.5)
            elif task == 'regression' and examples:
                fallback = np.mean([ex.get('g3', 0) for ex in examples])
            else:
                fallback = None
            return fallback, all_responses, []
        
        final_pred = Counter(all_predictions).most_common(1)[0][0] if task == 'classification' else np.median(all_predictions)
        return final_pred, all_responses, all_predictions
    
    def _is_consistent_with_examples(
        self,
        prediction: any,
        examples: List[Dict],
        test_student: Dict,
        task: str = 'classification'
    ) -> bool:
        """Check if prediction is consistent with few-shot examples pattern"""
        if task == 'classification':
            test_g2 = test_student.get('G2', None)
            test_failures = test_student.get('failures', 0)
            similar_examples = []
            for ex in examples:
                ex_student = ex['student']
                ex_g2 = ex_student.get('G2', None)
                ex_failures = ex_student.get('failures', 0)
                similarity = 0
                if test_g2 is not None and ex_g2 is not None:
                    if abs(test_g2 - ex_g2) <= 2:
                        similarity += 0.5
                if abs(test_failures - ex_failures) <= 1:
                    similarity += 0.5
                if similarity > 0.5:
                    similar_examples.append(ex['label'])
            if len(similar_examples) >= 2:
                majority_label = int(np.mean(similar_examples) >= 0.5)
                return prediction == majority_label
        return True
    
    def _process_prediction_result(
        self,
        pred: any,
        examples: List[Dict],
        test_student: Dict,
        prompt: str,
        system_prompt: str,
        task: str,
        all_responses: List[str],
        all_predictions: List[any]
    ) -> any:
        """Process prediction with optional error correction"""
        if not self.use_error_correction or pred is None:
            return pred
        
        if not self._is_consistent_with_examples(pred, examples, test_student, task=task):
            logger.info(f"Prediction {pred} inconsistent with examples, attempting correction")
            correction_response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.5,
                max_tokens=self._get_max_tokens()
            )
            if correction_response:
                parse_func = self.template.parse_classification_response if task == 'classification' else self.template.parse_regression_response
                correction_pred = parse_func(correction_response)
                if correction_pred is not None and self._is_consistent_with_examples(correction_pred, examples, test_student, task=task):
                    all_responses.append(correction_response)
                    all_predictions.append(correction_pred)
                    return correction_pred
        return pred
    
    def predict_classification(
        self,
        X_test: pd.DataFrame,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        setup: str = 'A',
        strategy: Optional[str] = None
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Predict binary classification (pass/fail) for test data"""
        predictions = []
        detailed_results = []
        system_prompt = "You are an expert educational analyst specializing in student performance prediction. Follow the instructions carefully and provide only the requested output format."
        
        for idx in tqdm(range(len(X_test)), desc="Predicting (classification)"):
            test_student = X_test.iloc[idx].to_dict()
            ground_truth = int(y_test.iloc[idx]) if hasattr(y_test, 'iloc') else int(y_test[idx])
            
            # Calculate difficulty and get adaptive parameters
            if self.use_dynamic_examples or self.use_adaptive_temperature:
                difficulty = self._calculate_difficulty_score(test_student, X_train, y_train)
                adaptive_temp = self._get_adaptive_temperature(difficulty)
                dynamic_n_examples = self._get_dynamic_n_examples(difficulty)
            else:
                difficulty = 0.0
                adaptive_temp = self.temperature
                dynamic_n_examples = self.n_examples
            
            # Select examples
            original_n_examples = self.n_examples
            self.n_examples = dynamic_n_examples
            if strategy in ['similarity', 'hybrid']:
                examples = self.select_examples(X_train, y_train, test_student, task='classification', strategy=strategy)
            else:
                if idx == 0:
                    examples = self.select_examples(X_train, y_train, None, task='classification', strategy=strategy)
            self.n_examples = original_n_examples
            
            # Build prompt and predict
            prompt = self.template.build_classification_prompt(examples=examples, test_student=test_student, setup=setup)
            pred, all_responses, all_predictions = self._predict_with_self_consistency(
                prompt=prompt, system_prompt=system_prompt, task='classification', examples=examples
            )
            
            # Error correction
            pred = self._process_prediction_result(pred, examples, test_student, prompt, system_prompt, 'classification', all_responses, all_predictions)
            
            # Fallback if still None
            if pred is None:
                logger.warning("All predictions failed, using fallback")
                pred = int(np.mean([ex['label'] for ex in examples]) >= 0.5)
            
            predictions.append(pred)
            
            # Save detailed result
            result = {
                'test_index': int(idx),
                'test_student': test_student,
                'prompt': prompt,
                'system_prompt': system_prompt,
                'response': all_responses[0] if all_responses else "",
                'ground_truth': ground_truth,
                'prediction': int(pred),
                'correct': int(pred == ground_truth),
                'few_shot_examples': [{'student': ex['student'], 'label': int(ex['label'])} for ex in examples]
            }
            
            if self.use_self_consistency:
                result['all_responses'] = all_responses
                result['all_predictions'] = [int(p) for p in all_predictions] if all_predictions else []
                result['self_consistency_used'] = True
            
            if self.use_dynamic_examples or self.use_adaptive_temperature:
                result['difficulty_score'] = float(difficulty)
                result['n_examples_used'] = len(examples)
                result['temperature_used'] = float(adaptive_temp)
            
            detailed_results.append(result)
        
        return np.array(predictions), detailed_results
    
    def predict_regression(
        self,
        X_test: pd.DataFrame,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        setup: str = 'A',
        strategy: Optional[str] = None
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Predict regression (G3 grade) for test data"""
        predictions = []
        detailed_results = []
        system_prompt = "You are an expert educational analyst specializing in student performance prediction. Follow the instructions carefully and provide only the requested output format."
        
        for idx in tqdm(range(len(X_test)), desc="Predicting (regression)"):
            test_student = X_test.iloc[idx].to_dict()
            ground_truth = float(y_test.iloc[idx]) if hasattr(y_test, 'iloc') else float(y_test[idx])
            
            # Calculate difficulty and get adaptive parameters
            if self.use_dynamic_examples or self.use_adaptive_temperature:
                difficulty = self._calculate_difficulty_score(test_student, X_train, y_train)
                adaptive_temp = self._get_adaptive_temperature(difficulty)
                dynamic_n_examples = self._get_dynamic_n_examples(difficulty)
            else:
                difficulty = 0.0
                adaptive_temp = self.temperature
                dynamic_n_examples = self.n_examples
            
            # Select examples
            original_n_examples = self.n_examples
            self.n_examples = dynamic_n_examples
            if strategy in ['similarity', 'hybrid']:
                examples = self.select_examples(X_train, y_train, test_student, task='regression', strategy=strategy)
            else:
                if idx == 0:
                    examples = self.select_examples(X_train, y_train, None, task='regression', strategy=strategy)
            self.n_examples = original_n_examples
            
            # Build prompt and predict
            prompt = self.template.build_regression_prompt(examples=examples, test_student=test_student, setup=setup)
            pred, all_responses, all_predictions = self._predict_with_self_consistency(
                prompt=prompt, system_prompt=system_prompt, task='regression', examples=examples
            )
            
            # Error correction
            pred = self._process_prediction_result(pred, examples, test_student, prompt, system_prompt, 'regression', all_responses, all_predictions)
            
            # Fallback if None
            if pred is None:
                logger.warning("All predictions failed, using fallback")
                pred = np.mean([ex.get('g3', 0) for ex in examples])
            
            predictions.append(pred)
            
            # Save detailed result
            result = {
                'test_index': int(idx),
                'test_student': test_student,
                'prompt': prompt,
                'system_prompt': system_prompt,
                'response': all_responses[0] if all_responses else "",
                'ground_truth': ground_truth,
                'prediction': float(pred),
                'error': float(abs(pred - ground_truth)),
                'squared_error': float((pred - ground_truth) ** 2),
                'few_shot_examples': [{'student': ex['student'], 'g3': float(ex.get('g3', 0))} for ex in examples]
            }
            
            if self.use_self_consistency:
                result['all_responses'] = all_responses
                result['all_predictions'] = [float(p) for p in all_predictions] if all_predictions else []
                result['self_consistency_used'] = True
            
            if self.use_dynamic_examples or self.use_adaptive_temperature:
                result['difficulty_score'] = float(difficulty)
                result['n_examples_used'] = len(examples)
                result['temperature_used'] = float(adaptive_temp)
            
            detailed_results.append(result)
        
        return np.array(predictions), detailed_results
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task: str = 'classification',
        setup: str = 'A',
        random_state: int = 42,
        strategy: Optional[str] = None,
        X_encoded: Optional[pd.DataFrame] = None
    ) -> Tuple[float, np.ndarray, np.ndarray, List[Dict]]:
        """
        Evaluate model using the same data split as baseline (single split)
        
        Args:
            X: Feature DataFrame (original, for prompt generation)
            y: Target Series
            task: 'classification' or 'regression'
            setup: 'A' or 'C'
            random_state: Random seed
            strategy: Example selection strategy (overrides self.example_strategy)
            X_encoded: Optional encoded feature DataFrame (for feature selection model training)
            
        Returns:
            Tuple of (score, y_test, y_pred, detailed_results)
        """
        from sklearn.metrics import accuracy_score, mean_squared_error
        
        strategy = strategy or self.example_strategy
        
        print(f"\nStarting few-shot prompt evaluation using the baseline split")
        print(f"Task: {task}, Setup: {setup}")
        print(f"Example selection strategy: {strategy}")
        print(f"Few-shot examples: {self.n_examples}")
        print(f"Chain-of-Thought enabled: {self.use_cot}")
        
        kf = KFold(n_splits=10, shuffle=True, random_state=random_state)
        splits = list(kf.split(X))
        train_idx, test_idx = splits[0]
        
        if hasattr(X, 'iloc'):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        else:
            X_train, X_test = X[train_idx], X[test_idx]
        
        if hasattr(y, 'iloc'):
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        else:
            y_train, y_test = y[train_idx], y[test_idx]
        
        print(f"Train size: {len(X_train)} (used for few-shot examples)")
        print(f"Test size: {len(X_test)} (used for prediction)")
        
        # Feature selection if enabled
        if self.use_feature_selection:
            print(f"\nFeature selection enabled")
            print(f"Model: {self.feature_selection_model.upper()}")
            print(f"Selecting top {self.n_top_features} important features")
            
            if X_encoded is not None:
                if hasattr(X_encoded, 'iloc'):
                    X_train_encoded = X_encoded.iloc[train_idx]
                else:
                    X_train_encoded = X_encoded[train_idx]
            else:
                X_train_encoded = X_train
                non_numeric = X_train.select_dtypes(exclude=[np.number]).columns
                if len(non_numeric) > 0:
                    logger.warning(f"Feature selection using original features with non-numeric columns: {list(non_numeric)}")
            
            self.feature_selector = FeatureSelector(
                model_type=self.feature_selection_model,
                n_features=self.n_top_features,
                random_state=random_state
            )
            self.feature_selector.fit(X_train_encoded, y_train, task=task)
            selected_features = self.feature_selector.get_selected_features()
            print(f"Selected features: {selected_features}")
            self.template.selected_features = selected_features
        else:
            self.template.selected_features = None
        
        # Predict
        if task == 'classification':
            y_pred, detailed_results = self.predict_classification(
                X_test, X_train, y_train, y_test, setup=setup, strategy=strategy
            )
            score = accuracy_score(y_test, y_pred)
            print(f"\nClassification accuracy: {score:.4f}")
        else:
            y_pred, detailed_results = self.predict_regression(
                X_test, X_train, y_train, y_test, setup=setup, strategy=strategy
            )
            score = np.sqrt(mean_squared_error(y_test, y_pred))
            print(f"\nRegression RMSE: {score:.4f}")
        
        return score, y_test.values, y_pred, detailed_results
