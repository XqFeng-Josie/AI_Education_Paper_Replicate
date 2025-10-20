"""
Academic Advisor Agent
"""

from typing import Dict, Any
from .base_agent import BaseAgent
from ..prompt import ACADEMIC_ADVISOR_SYSTEM_PROMPT, ACADEMIC_ADVISOR_USER_PROMPT


class AcademicAdvisorAgent(BaseAgent):
    """Agent specialized in academic performance analysis"""
    
    def __init__(self, llm_wrapper, weight: float = 0.25):
        super().__init__(
            name="Academic Advisor",
            role="Analyze student academic performance and VLE engagement",
            llm_wrapper=llm_wrapper,
            system_prompt=ACADEMIC_ADVISOR_SYSTEM_PROMPT,
            weight=weight
        )
    
    def analyze(self, student_narrative: str, **kwargs) -> Dict[str, Any]:
        """
        Analyze student's academic engagement
        
        Args:
            student_narrative: Student behavior description
            
        Returns:
            Academic analysis with risk assessment
        """
        self.logger.info(f"Analyzing student with {self.name}")
        
        # Format prompt
        user_prompt = ACADEMIC_ADVISOR_USER_PROMPT.format(
            student_narrative=student_narrative
        )
        
        # Get LLM response
        try:
            response = self.llm.generate_json(user_prompt, self.system_prompt)
            return self._parse_response(response)
        
        except Exception as e:
            self.logger.error(f"Error in {self.name} analysis: {e}")
            # Return default response
            return {
                'agent_name': self.name,
                'error': str(e),
                'risk_score': 5,
                'confidence': 'Low',
                'engagement_assessment': 'Error in analysis',
                'red_flags': [],
                'strengths': []
            }





