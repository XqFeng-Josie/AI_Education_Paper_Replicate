"""
Peer Comparison Agent
"""

from typing import Dict, Any
from .base_agent import BaseAgent
from ..prompt import PEER_COMPARATOR_SYSTEM_PROMPT, PEER_COMPARATOR_USER_PROMPT


class PeerComparatorAgent(BaseAgent):
    """Agent specialized in peer comparison analysis"""
    
    def __init__(self, llm_wrapper, weight: float = 0.20):
        super().__init__(
            name="Peer Comparator",
            role="Compare student performance with peers",
            llm_wrapper=llm_wrapper,
            system_prompt=PEER_COMPARATOR_SYSTEM_PROMPT,
            weight=weight
        )
    
    def analyze(self, student_narrative: str, peer_context: str = "", **kwargs) -> Dict[str, Any]:
        """
        Analyze student relative to peers
        
        Args:
            student_narrative: Student behavior description
            peer_context: Peer comparison context
            
        Returns:
            Peer comparison analysis with risk assessment
        """
        self.logger.info(f"Analyzing student with {self.name}")
        
        # Format prompt
        user_prompt = PEER_COMPARATOR_USER_PROMPT.format(
            student_narrative=student_narrative,
            peer_context=peer_context if peer_context else "Peer comparison data not available"
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
                'relative_performance': 'unknown',
                'comparison_insights': []
            }





