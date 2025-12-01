"""
Instructor Agent - Posts content, responds to questions
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class InstructorAgent:
    """
    Teacher Agent responsible for course organization
    """
    
    def __init__(self, llm_client, prompt_config_path=None):
        """
        Args:
            llm_client: LlamaClient instance
            prompt_config_path: Path to instructor prompts YAML
        """
        self.llm = llm_client
        
        # Load prompts
        if prompt_config_path is None:
            prompt_config_path = Path(__file__).parent.parent / 'prompts' / 'instructor_prompts.yaml'
        
        with open(prompt_config_path, 'r') as f:
            self.prompts = yaml.safe_load(f)
        
        self.system_prompt = self.prompts['system_prompt']
        
        # Course structure
        self.week_topics = {
            1: "Introduction to Computing - Overview and Getting Started",
            2: "Basic Programming Concepts - Variables and Data Types",
            3: "Control Flow and Functions",
            4: "Assignment Week - Complete and Submit TMA 1",
            5: "Data Structures - Lists and Arrays",
            6: "Object-Oriented Programming Basics",
            7: "File I/O and Error Handling",
            8: "Course Review and Advanced Topics"
        }
    
    def post_weekly_content(self, week_num: int) -> Dict[str, Any]:
        """
        Post learning materials for the week
        
        Returns:
            dict with materials posted
        """
        logger.info(f"Instructor posting content for Week {week_num}")
        
        week_topic = self.week_topics.get(week_num, "Advanced Topics")
        
        prompt = self.prompts['weekly_content_prompt'].format(
            week_num=week_num,
            week_topic=week_topic
        )
        
        # Generate content description
        content_description = self.llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.7,
            max_tokens=256
        )
        
        if content_description is None:
            # Fallback
            content_description = f"Week {week_num}: {week_topic}. Materials include lectures, readings, and practice exercises."
        
        return {
            'week': week_num,
            'topic': week_topic,
            'description': content_description,
            'resources_available': [
                'lecture_videos',
                'reading_materials',
                'forum_discussion',
                'practice_quiz' if week_num < 4 else 'assignment_portal'
            ]
        }
    
    def respond_to_forum_question(self, student_question: str) -> str:
        """
        Respond to a student question in the forum
        
        Args:
            student_question: The question asked
            
        Returns:
            Instructor's response
        """
        prompt = self.prompts['forum_response_prompt'].format(
            student_question=student_question
        )
        
        response = self.llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.7,
            max_tokens=128
        )
        
        if response is None:
            response = "Thank you for your question. Please refer to the course materials or contact me directly for assistance."
        
        return response

