"""
Base agent class for multi-agent system
"""

import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, 
                 name: str,
                 role: str,
                 llm_wrapper,
                 system_prompt: str,
                 weight: float = 1.0):
        """
        Initialize agent
        
        Args:
            name: Agent name
            role: Agent's role description
            llm_wrapper: LLM wrapper instance
            system_prompt: System prompt for this agent
            weight: Weight for this agent's output in final decision (0-1)
        """
        self.name = name
        self.role = role
        self.llm = llm_wrapper
        self.system_prompt = system_prompt
        self.weight = weight
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def analyze(self, student_narrative: str, **kwargs) -> Dict[str, Any]:
        """
        Analyze student data and return assessment
        
        Args:
            student_narrative: Text description of student behavior
            **kwargs: Additional context
            
        Returns:
            Dictionary containing agent's analysis
        """
        pass
    
    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate agent response"""
        # Add agent metadata
        response['agent_name'] = self.name
        response['agent_weight'] = self.weight
        
        # Validate required fields
        if 'risk_score' not in response:
            self.logger.warning(f"Agent {self.name} response missing risk_score")
            response['risk_score'] = 5  # Default medium risk
        
        if 'confidence' not in response:
            self.logger.warning(f"Agent {self.name} response missing confidence")
            response['confidence'] = "Medium"
        
        return response
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            'name': self.name,
            'role': self.role,
            'weight': self.weight
        }





