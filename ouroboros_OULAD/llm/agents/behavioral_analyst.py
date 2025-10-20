"""
Behavioral Analyst Agent
"""

from typing import Dict, Any
from .base_agent import BaseAgent
from ..prompt import BEHAVIORAL_ANALYST_SYSTEM_PROMPT, BEHAVIORAL_ANALYST_USER_PROMPT


class BehavioralAnalystAgent(BaseAgent):
    """Agent specialized in learning behavior pattern analysis"""
    
    def __init__(self, llm_wrapper, weight: float = 0.25):
        super().__init__(
            name="Behavioral Analyst",
            role="Analyze student learning behavior patterns",
            llm_wrapper=llm_wrapper,
            system_prompt=BEHAVIORAL_ANALYST_SYSTEM_PROMPT,
            weight=weight
        )
    
    def analyze(self, student_narrative: str, **kwargs) -> Dict[str, Any]:
        """
        Analyze student's behavioral patterns
        
        Args:
            student_narrative: Student behavior description
            
        Returns:
            Behavioral analysis with risk assessment
        """
        self.logger.info(f"Analyzing student with {self.name}")
        
        # Format prompt
        user_prompt = BEHAVIORAL_ANALYST_USER_PROMPT.format(
            student_narrative=student_narrative
        )
        
        # Get LLM response
        try:
            response = self.llm.generate_json(user_prompt, self.system_prompt)
            return self._parse_response(response)
        
        except Exception as e:
            self.logger.error(f"Error in {self.name} analysis: {e}")
            return {
                'agent_name': self.name,
                'error': str(e),
                'risk_score': 5,
                'confidence': 'Low',
                'behavioral_trend': 'unknown',
                'disengagement_signals': [],
                'positive_behaviors': []
            }





