"""
Decision Maker Agent
"""

import json
from typing import Dict, Any, List
from .base_agent import BaseAgent
from ..prompt import DECISION_MAKER_SYSTEM_PROMPT, DECISION_MAKER_USER_PROMPT


class DecisionMakerAgent(BaseAgent):
    """Agent that synthesizes all analyses and makes final decision"""
    
    def __init__(self, llm_wrapper, weight: float = 0.10):
        super().__init__(
            name="Decision Maker",
            role="Aggregate insights and make final risk assessment",
            llm_wrapper=llm_wrapper,
            system_prompt=DECISION_MAKER_SYSTEM_PROMPT,
            weight=weight
        )
    
    def make_decision(self, 
                     student_narrative: str,
                     agent_analyses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Make final decision based on all agent analyses
        
        Args:
            student_narrative: Student behavior description
            agent_analyses: Dictionary of analyses from each agent
            
        Returns:
            Final risk assessment and recommendations
        """
        self.logger.info(f"Making final decision with {self.name}")
        
        # Format agent analyses for prompt
        academic_analysis = json.dumps(agent_analyses.get('academic_advisor', {}), indent=2)
        behavioral_analysis = json.dumps(agent_analyses.get('behavioral_analyst', {}), indent=2)
        peer_analysis = json.dumps(agent_analyses.get('peer_comparator', {}), indent=2)
        temporal_analysis = json.dumps(agent_analyses.get('time_series_analyst', {}), indent=2)
        
        # Format prompt
        user_prompt = DECISION_MAKER_USER_PROMPT.format(
            student_narrative=student_narrative,
            academic_analysis=academic_analysis,
            behavioral_analysis=behavioral_analysis,
            peer_analysis=peer_analysis,
            temporal_analysis=temporal_analysis
        )
        
        # Get LLM response
        try:
            response = self.llm.generate_json(user_prompt, self.system_prompt)
            
            # Add aggregated risk score
            response['aggregated_risk_score'] = self._calculate_weighted_risk(agent_analyses)
            
            return response
        
        except Exception as e:
            self.logger.error(f"Error in {self.name} decision: {e}")
            return {
                'agent_name': self.name,
                'error': str(e),
                'final_risk_level': 'No Risk',
                'confidence': 'Low',
                'explanation': 'Error in final decision making'
            }
    
    def _calculate_weighted_risk(self, agent_analyses: Dict[str, Dict[str, Any]]) -> float:
        """Calculate weighted average risk score from all agents"""
        total_weighted_score = 0
        total_weight = 0
        
        for agent_name, analysis in agent_analyses.items():
            if 'risk_score' in analysis and 'agent_weight' in analysis:
                total_weighted_score += analysis['risk_score'] * analysis['agent_weight']
                total_weight += analysis['agent_weight']
        
        if total_weight > 0:
            return total_weighted_score / total_weight
        else:
            return 5.0  # Default medium risk
    
    def analyze(self, student_narrative: str, **kwargs) -> Dict[str, Any]:
        """Not used for decision maker - use make_decision instead"""
        raise NotImplementedError("Use make_decision() for DecisionMakerAgent")





