"""
Agent implementations for multi-agent system
"""

from .base_agent import BaseAgent
from .academic_advisor import AcademicAdvisorAgent
from .behavioral_analyst import BehavioralAnalystAgent
from .peer_comparator import PeerComparatorAgent
from .time_series_analyst import TimeSeriesAnalystAgent
from .decision_maker import DecisionMakerAgent

__all__ = [
    'BaseAgent',
    'AcademicAdvisorAgent',
    'BehavioralAnalystAgent',
    'PeerComparatorAgent',
    'TimeSeriesAnalystAgent',
    'DecisionMakerAgent'
]





