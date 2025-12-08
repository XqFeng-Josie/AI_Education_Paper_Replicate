"""
Synthetic student data generation using LLM
"""
import pandas as pd
import json
import logging
import re
import os
from typing import Dict, List, Optional, Tuple
from .llm_client import LlamaClient

logger = logging.getLogger(__name__)


class StudentDataGenerator:
    """Generate synthetic student data using LLM"""
    
    # Data constraints (derived from original data analysis)
    CATEGORICAL_VALUES = {
        'school': ['GP', 'MS'],
        'sex': ['F', 'M'],
        'address': ['R', 'U'],
        'famsize': ['GT3', 'LE3'],
        'Pstatus': ['A', 'T'],
        'Mjob': ['at_home', 'health', 'other', 'services', 'teacher'],
        'Fjob': ['at_home', 'health', 'other', 'services', 'teacher'],
        'reason': ['course', 'home', 'other', 'reputation'],
        'guardian': ['father', 'mother', 'other'],
        'schoolsup': ['no', 'yes'],
        'famsup': ['no', 'yes'],
        'paid': ['no', 'yes'],
        'activities': ['no', 'yes'],
        'nursery': ['no', 'yes'],
        'higher': ['no', 'yes'],
        'internet': ['no', 'yes'],
        'romantic': ['no', 'yes']
    }
    
    NUMERIC_RANGES = {
        'age': (15, 22),
        'Medu': (0, 4),
        'Fedu': (0, 4),
        'traveltime': (1, 4),
        'studytime': (1, 4),
        'failures': (0, 3),
        'famrel': (1, 5),
        'freetime': (1, 5),
        'goout': (1, 5),
        'Dalc': (1, 5),
        'Walc': (1, 5),
        'health': (1, 5),
        'absences': (0, 32),
        'G1': (0, 20),
        'G2': (0, 20),
        'G3': (0, 20)
    }
    
    COLUMN_ORDER = [
        'school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
        'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures',
        'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet',
        'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences',
        'G1', 'G2', 'G3'
    ]
    
    def __init__(self, llm_client: LlamaClient, original_data: pd.DataFrame):
        """
        Initialize data generator
        
        Args:
            llm_client: LLM client
            original_data: Original data (for reference distribution)
        """
        self.llm_client = llm_client
        self.original_data = original_data
        self._analyze_original_distribution()
    
    def _analyze_original_distribution(self):
        """Analyze distribution characteristics of original data"""
        self.distributions = {}
        
        # Frequency distribution of categorical variables
        for col in self.CATEGORICAL_VALUES.keys():
            self.distributions[col] = self.original_data[col].value_counts(normalize=True).to_dict()
        
        # Statistical information of numeric variables
        for col in self.NUMERIC_RANGES.keys():
            self.distributions[col] = {
                'mean': float(self.original_data[col].mean()),
                'std': float(self.original_data[col].std()),
                'min': int(self.original_data[col].min()),
                'max': int(self.original_data[col].max())
            }
    
    def _create_generation_prompt(self, sample_student: Optional[Dict] = None) -> str:
        """
        Create data generation prompt
        
        Args:
            sample_student: Optional sample student data (for guiding generation)
            
        Returns:
            Generation prompt text
        """
        # Build constraint descriptions
        constraints = []
        constraints.append("Categorical variable constraints:")
        for col, values in self.CATEGORICAL_VALUES.items():
            constraints.append(f"  - {col}: {values}")
        
        constraints.append("\nNumeric variable ranges:")
        for col, (min_val, max_val) in self.NUMERIC_RANGES.items():
            constraints.append(f"  - {col}: [{min_val}, {max_val}]")
        
        # Add example (if provided)
        example_text = ""
        if sample_student:
            example_text = f"\nExample student data:\n{json.dumps(sample_student, indent=2, ensure_ascii=False)}"
        
        prompt = f"""You are an expert in educational data generation. Please generate a Portuguese secondary school student data record that conforms to the following constraints.

{chr(10).join(constraints)}

Important requirements:
1. All values must strictly conform to the above constraints
2. Grades G1, G2, G3 should be in the range [0, 20], and G3 is usually related to G1 and G2
3. Age is typically between 15-19 years old
4. Data should reflect realistic combinations of student characteristics (e.g., students with longer study time usually have better grades)
5. Output format must be JSON, containing all 32 fields
{example_text}

Please generate a new student data record in JSON format as follows:
{{
  "school": "GP",
  "sex": "F",
  "age": 17,
  ...
}}

Output only JSON, no other text."""
        
        return prompt
    
    def _parse_llm_response(self, response: str) -> Optional[Dict]:
        """
        Parse LLM response and extract JSON data
        
        Args:
            response: LLM generated text
            
        Returns:
            Parsed student data dictionary, returns None on failure
        """
        # Try to extract JSON
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return data
            except json.JSONDecodeError:
                pass
        
        # If direct parsing fails, try cleaning and parsing
        cleaned = response.strip()
        if cleaned.startswith('```'):
            # Remove code block markers
            cleaned = re.sub(r'```json\s*', '', cleaned)
            cleaned = re.sub(r'```\s*', '', cleaned)
        
        try:
            data = json.loads(cleaned)
            return data
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM response: {response[:200]}")
            return None
    
    def _validate_and_fix_record(self, record: Dict) -> Optional[Dict]:
        """
        Validate and fix generated data record
        
        Args:
            record: Original record
            
        Returns:
            Fixed record, returns None if cannot be fixed
        """
        fixed_record = {}
        
        # Validate and fix each field
        for col in self.COLUMN_ORDER:
            if col not in record:
                # Missing field, use default value or sample from distribution
                if col in self.CATEGORICAL_VALUES:
                    # Sample from original distribution
                    import random
                    values = list(self.CATEGORICAL_VALUES[col])
                    weights = [self.distributions[col].get(v, 0.1) for v in values]
                    fixed_record[col] = random.choices(values, weights=weights)[0]
                elif col in self.NUMERIC_RANGES:
                    min_val, max_val = self.NUMERIC_RANGES[col]
                    fixed_record[col] = int(self.distributions[col]['mean'])
                else:
                    logger.warning(f"Unknown field: {col}")
                    return None
            else:
                value = record[col]
                
                # Validate categorical variables
                if col in self.CATEGORICAL_VALUES:
                    if value not in self.CATEGORICAL_VALUES[col]:
                        # Try to fix (case, spaces, etc.)
                        value_str = str(value).strip().lower()
                        for valid_val in self.CATEGORICAL_VALUES[col]:
                            if valid_val.lower() == value_str:
                                value = valid_val
                                break
                        else:
                            # Cannot fix, use default value
                            import random
                            values = list(self.CATEGORICAL_VALUES[col])
                            weights = [self.distributions[col].get(v, 0.1) for v in values]
                            value = random.choices(values, weights=weights)[0]
                    fixed_record[col] = value
                
                # Validate numeric variables
                elif col in self.NUMERIC_RANGES:
                    try:
                        num_value = int(float(value))
                        # Use both predefined ranges and original data ranges to clamp values
                        min_val, max_val = self.NUMERIC_RANGES[col]
                        dist = self.distributions.get(col, {})
                        min_val = max(min_val, dist.get('min', min_val))
                        max_val = min(max_val, dist.get('max', max_val))
                        num_value = max(min_val, min(max_val, num_value))  # Clip to range
                        fixed_record[col] = num_value
                    except (ValueError, TypeError):
                        # Cannot convert, use mean
                        fixed_record[col] = int(self.distributions[col]['mean'])
                
                else:
                    fixed_record[col] = value
        
        # Ensure grade reasonableness (G3 is usually related to G1 and G2)
        if 'G1' in fixed_record and 'G2' in fixed_record and 'G3' in fixed_record:
            g1, g2, g3 = fixed_record['G1'], fixed_record['G2'], fixed_record['G3']
            # G3 is usually close to the average of G1 and G2, allow some variation
            expected_g3 = int((g1 + g2) / 2)
            if abs(g3 - expected_g3) > 5:  # If difference is too large, adjust G3
                fixed_record['G3'] = max(0, min(20, expected_g3))
        
        return fixed_record
    
    def generate_one(self, sample_student: Optional[Dict] = None) -> Optional[Dict]:
        """
        Generate one student data record
        
        Args:
            sample_student: Optional sample student data
            
        Returns:
            Generated student data dictionary, returns None on failure
        """
        prompt = self._create_generation_prompt(sample_student)
        system_prompt = "You are a professional educational data analyst, skilled in generating student records that conform to real data distributions."
        
        response = self.llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.8,
            max_tokens=512
        )
        
        if not response:
            return None
        
        record = self._parse_llm_response(response)
        if not record:
            return None
        
        # Validate and fix
        fixed_record = self._validate_and_fix_record(record)
        return fixed_record
    
    def generate_batch(
        self, 
        n: int, 
        progress_callback=None,
        output_path: Optional[str] = None,
        resume: bool = False
    ) -> List[Dict]:
        """
        Generate student data in batch with real-time saving and resume support
        
        Args:
            n: Number of records to generate
            progress_callback: Progress callback function (optional)
            output_path: Output file path for real-time saving (optional)
            resume: Whether to resume from existing file (default: False)
            
        Returns:
            List of generated student data
        """
        records = []
        failed_count = 0
        max_failures = n * 2  # Allow up to 2x failures
        start_index = 0
        
        # Resume from existing file if requested
        if resume and output_path:
            existing_records, start_index = self.load_existing_records(output_path)
            if start_index > 0:
                logger.info(f"Resuming generation: {start_index} records already exist, cleaning invalid rows")
                cleaned_records = self.clean_existing_records(existing_records, output_path=output_path)
                records = cleaned_records
                start_index = len(cleaned_records)
                if start_index < len(existing_records):
                    logger.info(f"Removed/adjusted invalid rows: kept {start_index}/{len(existing_records)}")
                logger.info(f"Generating {max(0, n - start_index)} additional records to reach target {n}")
            else:
                records = existing_records
        
        # Randomly select some examples from original data
        sample_students = self.original_data.sample(min(10, len(self.original_data))).to_dict('records')
        
        for i in range(start_index, n):
            if failed_count >= max_failures:
                logger.error(f"Too many failures, stopping generation")
                break
            
            # Randomly select an example (for guiding generation)
            sample = sample_students[i % len(sample_students)] if sample_students else None
            
            record = self.generate_one(sample_student=sample)
            if record:
                records.append(record)
                
                # Real-time saving: append to file immediately
                if output_path:
                    try:
                        self.append_to_csv(record, output_path)
                    except Exception as e:
                        logger.error(f"Failed to save record {i+1} to file: {e}")
                
                if progress_callback:
                    progress_callback(i + 1, n)
            else:
                failed_count += 1
                logger.warning(f"Failed to generate record {i+1}")
        
        logger.info(f"Successfully generated {len(records)}/{n} records")
        return records
    
    def save_to_csv(self, records: List[Dict], output_path: str):
        """
        Save generated data to CSV file
        
        Args:
            records: List of student data records
            output_path: Output file path
        """
        df = pd.DataFrame(records)
        
        # Ensure column order is correct
        df = df[self.COLUMN_ORDER]
        
        # Save as semicolon-separated CSV (consistent with original format)
        df.to_csv(output_path, sep=';', index=False)
        logger.info(f"Saved {len(records)} records to {output_path}")
    
    def append_to_csv(self, record: Dict, output_path: str):
        """
        Append a single record to CSV file (for real-time saving)
        
        Args:
            record: Single student data record
            output_path: Output file path
        """
        # Check if file exists
        file_exists = os.path.exists(output_path)
        
        # Ensure all columns exist in record
        complete_record = {}
        for col in self.COLUMN_ORDER:
            complete_record[col] = record.get(col, None)
        
        df = pd.DataFrame([complete_record])
        
        # Ensure column order is correct
        df = df[self.COLUMN_ORDER]
        
        # Append to CSV (header only if file doesn't exist)
        df.to_csv(output_path, sep=';', index=False, mode='a', header=not file_exists)
    
    def load_existing_records(self, output_path: str) -> Tuple[List[Dict], int]:
        """
        Load existing records from CSV file (for resume functionality)
        
        Args:
            output_path: Output file path
            
        Returns:
            Tuple of (existing_records_list, count)
        """
        if not os.path.exists(output_path):
            return [], 0
        
        try:
            df = pd.read_csv(output_path, sep=';')
            records = df.to_dict('records')
            count = len(records)
            logger.info(f"Loaded {count} existing records from {output_path}")
            return records, count
        except Exception as e:
            logger.warning(f"Failed to load existing records from {output_path}: {e}")
            return [], 0

    def clean_existing_records(self, records: List[Dict], output_path: Optional[str] = None) -> List[Dict]:
        """
        Clean already generated records by re-validating and fixing; drop those that cannot be fixed.
        If output_path is provided, overwrite the file with cleaned records.
        """
        cleaned = []
        for idx, record in enumerate(records):
            fixed = self._validate_and_fix_record(record)
            if fixed:
                cleaned.append(fixed)
            else:
                logger.warning(f"Dropping invalid record at index {idx}")
        if output_path is not None:
            # Overwrite with cleaned data to ensure subsequent resume uses valid rows
            self.save_to_csv(cleaned, output_path)
        return cleaned

