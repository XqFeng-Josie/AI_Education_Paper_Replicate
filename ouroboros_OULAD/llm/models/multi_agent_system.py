"""
Multi-Agent System for At-Risk Student Prediction
"""

import logging
from typing import Dict, Any, List, Optional
import yaml
import os

from ..agents.academic_advisor import AcademicAdvisorAgent
from ..agents.behavioral_analyst import BehavioralAnalystAgent
from ..agents.peer_comparator import PeerComparatorAgent
from ..agents.time_series_analyst import TimeSeriesAnalystAgent
from ..agents.decision_maker import DecisionMakerAgent

logger = logging.getLogger(__name__)


class MultiAgentSystem:
    """Coordinates multiple agents for student risk assessment"""
    
    def __init__(self, 
                 llm_wrapper,
                 config_path: Optional[str] = None):
        """
        Initialize multi-agent system
        
        Args:
            llm_wrapper: LLM wrapper instance to be shared by all agents
            config_path: Path to agent configuration file
        """
        self.llm = llm_wrapper
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()
        
        # Initialize agents
        self.agents = self._initialize_agents()
        self.decision_maker = DecisionMakerAgent(
            llm_wrapper=self.llm,
            weight=self.config['agents']['decision_maker']['weight']
        )
        
        logger.info(f"Initialized MultiAgentSystem with {len(self.agents)} agents")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration if config file not found"""
        return {
            'agents': {
                'academic_advisor': {'weight': 0.25},
                'behavioral_analyst': {'weight': 0.25},
                'peer_comparator': {'weight': 0.20},
                'time_series_analyst': {'weight': 0.20},
                'decision_maker': {'weight': 0.10}
            },
            'coordination': {
                'mode': 'sequential',
                'share_intermediate_results': True
            }
        }
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all specialist agents"""
        agents = {}
        
        # Academic Advisor
        agents['academic_advisor'] = AcademicAdvisorAgent(
            llm_wrapper=self.llm,
            weight=self.config['agents']['academic_advisor']['weight']
        )
        
        # Behavioral Analyst
        agents['behavioral_analyst'] = BehavioralAnalystAgent(
            llm_wrapper=self.llm,
            weight=self.config['agents']['behavioral_analyst']['weight']
        )
        
        # Peer Comparator
        agents['peer_comparator'] = PeerComparatorAgent(
            llm_wrapper=self.llm,
            weight=self.config['agents']['peer_comparator']['weight']
        )
        
        # Time Series Analyst
        agents['time_series_analyst'] = TimeSeriesAnalystAgent(
            llm_wrapper=self.llm,
            weight=self.config['agents']['time_series_analyst']['weight']
        )
        
        return agents
    
    def predict(self, 
                student_narrative: str,
                peer_context: str = "",
                return_intermediate: bool = True) -> Dict[str, Any]:
        """
        Predict student risk level using multi-agent system
        
        Args:
            student_narrative: Text description of student behavior
            peer_context: Peer comparison context
            return_intermediate: Whether to return intermediate agent results
            
        Returns:
            Dictionary containing final prediction and optionally intermediate results
        """
        logger.info("Starting multi-agent prediction")
        
        # Step 1: Collect analyses from all specialist agents
        agent_analyses = {}
        
        for agent_name, agent in self.agents.items():
            try:
                logger.info(f"Running {agent_name} analysis")
                
                if agent_name == 'peer_comparator':
                    analysis = agent.analyze(
                        student_narrative=student_narrative,
                        peer_context=peer_context
                    )
                else:
                    analysis = agent.analyze(student_narrative=student_narrative)
                
                agent_analyses[agent_name] = analysis
                logger.info(f"{agent_name} completed with risk_score: {analysis.get('risk_score', 'N/A')}")
            
            except Exception as e:
                logger.error(f"Error in {agent_name}: {e}")
                agent_analyses[agent_name] = {
                    'agent_name': agent_name,
                    'error': str(e),
                    'risk_score': 5,  # Default medium risk
                    'confidence': 'Low'
                }
        
        # Step 2: Decision maker synthesizes all analyses
        try:
            logger.info("Running decision maker")
            final_decision = self.decision_maker.make_decision(
                student_narrative=student_narrative,
                agent_analyses=agent_analyses
            )
        except Exception as e:
            logger.error(f"Error in decision maker: {e}")
            final_decision = {
                'error': str(e),
                'final_risk_level': 'No Risk',
                'confidence': 'Low'
            }
        
        # Prepare result
        result = {
            'final_decision': final_decision,
            'student_narrative': student_narrative
        }
        
        if return_intermediate:
            result['agent_analyses'] = agent_analyses
        
        logger.info(f"Multi-agent prediction complete: {final_decision.get('final_risk_level', 'Unknown')}")
        
        return result
    
    def predict_batch(self,
                     student_narratives: List[str],
                     peer_contexts: Optional[List[str]] = None,
                     return_intermediate: bool = False) -> List[Dict[str, Any]]:
        """
        Predict risk for multiple students
        
        Args:
            student_narratives: List of student behavior descriptions
            peer_contexts: Optional list of peer contexts
            return_intermediate: Whether to return intermediate results
            
        Returns:
            List of prediction results
        """
        if peer_contexts is None:
            peer_contexts = [""] * len(student_narratives)
        
        results = []
        
        for i, (narrative, peer_context) in enumerate(zip(student_narratives, peer_contexts)):
            logger.info(f"Processing student {i+1}/{len(student_narratives)}")
            
            try:
                result = self.predict(
                    student_narrative=narrative,
                    peer_context=peer_context,
                    return_intermediate=return_intermediate
                )
                results.append(result)
            
            except Exception as e:
                logger.error(f"Error processing student {i}: {e}")
                results.append({
                    'error': str(e),
                    'final_decision': {
                        'final_risk_level': 'No Risk',
                        'confidence': 'Low'
                    }
                })
        
        return results
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the multi-agent system"""
        return {
            'num_agents': len(self.agents),
            'agents': {name: agent.get_info() for name, agent in self.agents.items()},
            'decision_maker': self.decision_maker.get_info(),
            'configuration': self.config
        }





