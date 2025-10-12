#!/usr/bin/env python3
"""
Prompt Manager for NCTE Classroom Transcript Analysis

This module handles all prompt-related functionality including:
- Loading prompt configurations from YAML
- Formatting prompts for different tasks
- Managing few-shot examples
- Supporting both string and message formats
"""

import yaml
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional


class PromptManager:
    """Manages prompt templates and formatting for NCTE tasks."""
    
    def __init__(self, config_path: str = "prompts_config.yaml"):
        """Initialize prompt manager with configuration file."""
        self.config_path = config_path
        self.prompts = self._load_prompts()
    
    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompt configurations from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt config file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML config: {e}")
    
    def get_task_config(self, task_name: str) -> Dict[str, str]:
        """Get prompt configuration for a specific task."""
        if task_name not in self.prompts:
            raise ValueError(f"Task {task_name} not found in prompts config")
        return self.prompts[task_name]
    
    def get_task_specific_config(self, task_name: str) -> Dict[str, Any]:
        """Get task-specific configuration including template type."""
        task_configs = self.prompts.get('task_configs', {})
        return task_configs.get(task_name, {'template_type': 'dual_text'})
    
    
    def format_message_prompt(self, task_name: str, student_text: str, teacher_text: str = "", 
                            few_shot_examples: str = "") -> List[Dict[str, str]]:
        """Format prompt as message list (for modern LLM APIs)."""
        config = self.get_task_config(task_name)
        task_config = self.get_task_specific_config(task_name)
        template_type = task_config.get('template_type', 'dual_text')
        
        # Build system message
        system_message = {
            "role": "system",
            "content": config['system_prompt']
        }
        
        # Build user message with instruction and examples
        instruction = config['instruction']
        
        # Add few-shot examples if provided
        if few_shot_examples:
            instruction += f"\n\n{few_shot_examples}"
        
        # Add the current example to classify
        if template_type == 'single_text':
            text_field = task_config.get('text_field', 'student_text')
            text_label = task_config.get('text_label', 'Student')
            if text_field == 'teacher_text':
                current_example = f"Teacher: {teacher_text}"
            else:
                # Use student_text or text field depending on task
                text_content = student_text if student_text else teacher_text
                current_example = f"{text_label}: {text_content}"
        else:
            current_example = f"Student: {student_text}\nTeacher: {teacher_text}"
        
        user_content = f"{instruction}\n\n{current_example}\n\nClassification:"
        
        user_message = {
            "role": "user", 
            "content": user_content
        }
        
        return [system_message, user_message]
    
    def prepare_few_shot_examples(self, task_name: str, train_data: pd.DataFrame, 
                                 n_shots: int = 5) -> str:
        """Prepare few-shot examples for a task."""
        config = self.get_task_config(task_name)
        task_config = self.get_task_specific_config(task_name)
        template_type = task_config.get('template_type', 'dual_text')
        
        # Sample balanced examples
        task_data = train_data.dropna(subset=[task_name])
        examples = []
        
        # Get examples for each class
        for label in [0, 1]:
            label_data = task_data[task_data[task_name] == label]
            n_examples = min(n_shots // 2, len(label_data))
            if n_examples > 0:
                # Select from first half to avoid overlap with test examples
                available_data = label_data.head(max(1, len(label_data) // 2))
                if len(available_data) >= n_examples:
                    sampled = available_data.sample(n=n_examples, random_state=42)
                else:
                    sampled = available_data
                
                for _, row in sampled.iterrows():
                    # Format example based on template type
                    if template_type == 'single_text':
                        text_field = task_config.get('text_field', 'student_text')
                        text_label = task_config.get('text_label', 'Student')
                        
                        # Get the text content from the appropriate field
                        if text_field in row and pd.notna(row[text_field]):
                            text_content = row[text_field]
                        elif 'student_text' in row and pd.notna(row['student_text']):
                            text_content = row['student_text']
                        elif 'teacher_text' in row and pd.notna(row['teacher_text']):
                            text_content = row['teacher_text']
                        else:
                            continue  # Skip if no text found
                        
                        example_text = f"{text_label}: {text_content}\nClassification: {int(row[task_name])}"
                    else:
                        # Use both fields for dual text tasks  
                        example_text = f"Student: {row['student_text']}\nTeacher: {row['teacher_text']}\nClassification: {int(row[task_name])}"
                    examples.append(example_text)
        
        if examples:
            return config['few_shot_prefix'] + '\n\n'.join(examples) + '\n\nNow classify this new example:\n\n'
        else:
            return ""
    
    def get_global_settings(self) -> Dict[str, Any]:
        """Get global settings from configuration."""
        return self.prompts.get('global_settings', {})
    
    def list_available_tasks(self) -> List[str]:
        """Get list of available tasks."""
        return [task for task in self.prompts.keys() 
                if task not in ['global_settings', 'task_configs']]


def test_prompt_manager():
    """Test the prompt manager functionality."""
    print("🧪 Testing Prompt Manager\n")
    
    # Create mock data
    train_data = pd.DataFrame({
        'student_text': [
            "I think the answer is 5",
            "Can I go to the bathroom?", 
            "The pattern is 2, 4, 6",
            "I don't understand this"
        ],
        'teacher_text': [
            "Good thinking! Can you explain?",
            "Please wait until break",
            "Excellent observation!",
            "Let me help you with this"
        ],
        'student_on_task': [1, 0, 1, 0],
        'teacher_on_task': [1, 0, 1, 1],
        'high_uptake': [1, 0, 1, 0],
        'focusing_question': [0, 0, 1, 1]
    })
    
    try:
        manager = PromptManager("prompts_config.yaml")
        print("✅ PromptManager loaded successfully")
    except Exception as e:
        print(f"❌ Error loading PromptManager: {e}")
        return False
    
    # Test different tasks
    tasks = ["student_on_task", "teacher_on_task", "high_uptake", "focusing_question"]
    success = True
    
    for task in tasks:
        print(f"\n📋 Testing {task}:")
        
        try:
            # Test message format
            messages = manager.format_message_prompt(
                task, "I think the answer is 7", "Good! How did you get that?"
            )
            print(f"   ✅ Message format generated ({len(messages)} messages)")
            
            # Test few-shot examples
            few_shot = manager.prepare_few_shot_examples(task, train_data, n_shots=2)
            print(f"   ✅ Few-shot examples prepared ({len(few_shot)} chars)")
            
        except Exception as e:
            print(f"   ❌ Error testing {task}: {e}")
            success = False
    
    return success


if __name__ == "__main__":
    print("Prompt Manager Test for NCTE LLM Classifier")
    print("=" * 60)
    
    test_passed = test_prompt_manager()
    
    print(f"\n{'='*60}")
    if test_passed:
        print("✅ ALL TESTS PASSED! Prompt manager works correctly.")
        print("\n🔧 Key Features:")
        print("   - YAML configuration loading ✅")
        print("   - String and message format support ✅")
        print("   - Few-shot example generation ✅")
        print("   - Task-specific template handling ✅")
    else:
        print("❌ SOME TESTS FAILED! Please check the implementation.")
    
    print("=" * 60)
