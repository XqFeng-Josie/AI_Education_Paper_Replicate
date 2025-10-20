"""
Time Series Analyst Agent
"""

from typing import Dict, Any
from .base_agent import BaseAgent
from ..prompt import TIME_SERIES_ANALYST_SYSTEM_PROMPT, TIME_SERIES_ANALYST_USER_PROMPT


class TimeSeriesAnalystAgent(BaseAgent):
    """Agent specialized in temporal pattern analysis"""
    
    def __init__(self, llm_wrapper, weight: float = 0.20):
        super().__init__(
            name="Time Series Analyst",
            role="Analyze temporal trends in student behavior",
            llm_wrapper=llm_wrapper,
            system_prompt=TIME_SERIES_ANALYST_SYSTEM_PROMPT,
            weight=weight
        )
    
    def analyze(self, student_narrative: str, **kwargs) -> Dict[str, Any]:
        """
        Analyze temporal patterns in student behavior
        
        Args:
            student_narrative: Student behavior description
            
        Returns:
            Temporal analysis with risk assessment
        """
        self.logger.info(f"Analyzing student with {self.name}")
        
        # Format prompt
        user_prompt = TIME_SERIES_ANALYST_USER_PROMPT.format(
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
                'engagement_trajectory': 'unknown',
                'warning_signals': []
            }





