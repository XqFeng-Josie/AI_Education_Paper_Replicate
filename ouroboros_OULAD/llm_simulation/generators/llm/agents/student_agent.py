"""
Student Agent - Makes learning decisions based on personality
"""

import yaml
import logging
from pathlib import Path
import json
import re
from collections import defaultdict, Counter
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class StudentAgent:
    """
    Student Agent with unique personality
    """
    
    PERSONALITY_TYPES = {
        'high_performing': {
            'motivation': 'high',
            'time_management': 'organized',
            'prior_knowledge': 'strong',
            'learning_style': 'visual, reading, hands-on',
            'social_tendency': 'active',
            'max_daily_actions': 5,
            'typical_actions': 3
        },
        'average': {
            'motivation': 'medium',
            'time_management': 'moderate',
            'prior_knowledge': 'moderate',
            'learning_style': 'visual, reading',
            'social_tendency': 'moderate',
            'max_daily_actions': 3,
            'typical_actions': 2
        },
        'struggling': {
            'motivation': 'medium',
            'time_management': 'procrastinator',
            'prior_knowledge': 'weak',
            'learning_style': 'hands-on',
            'social_tendency': 'active (help-seeking)',
            'max_daily_actions': 3,
            'typical_actions': 1
        },
        'at_risk': {
            'motivation': 'low',
            'time_management': 'poor',
            'prior_knowledge': 'weak',
            'learning_style': 'passive',
            'social_tendency': 'passive',
            'max_daily_actions': 1,
            'typical_actions': 0
        }
    }
    
    DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    def __init__(self, student_id: str, personality_type: str, llm_client, prompt_config_path=None):
        """
        Args:
            student_id: Unique student ID
            personality_type: One of: high_performing, average, struggling, at_risk
            llm_client: LlamaClient instance
            prompt_config_path: Path to student prompts YAML
        """
        self.id = student_id
        self.personality_type = personality_type
        self.profile = self.PERSONALITY_TYPES[personality_type]
        self.llm = llm_client
        
        # Load prompts
        if prompt_config_path is None:
            prompt_config_path = Path(__file__).parent.parent / 'prompts' / 'student_prompts.yaml'
        
        with open(prompt_config_path, 'r') as f:
            self.prompts = yaml.safe_load(f)
        
        # Build system prompt
        behavior_desc = self.prompts['behavior_descriptions'][personality_type]
        self.behavior_description = behavior_desc.strip()
        self.system_prompt = self.prompts['system_prompt_template'].format(
            motivation=self.profile['motivation'],
            time_management=self.profile['time_management'],
            prior_knowledge=self.profile['prior_knowledge'],
            learning_style=self.profile['learning_style'],
            social_tendency=self.profile['social_tendency'],
            behavior_description=behavior_desc
        )
        
        # Track history
        self.action_history = []
        self.last_active_day = 0
        self.generated_week_plans: Dict[int, Dict[int, List[str]]] = {}
    
    def plan_weekly_actions(
        self,
        week_num: int,
        start_day: int,
        week_content: Dict[str, Any],
        tma_deadline_day: int = 28
    ) -> Dict[int, List[str]]:
        """
        Generate a 7-day action plan for the specified week.
        """
        end_day = start_day + 6
        deadline_window = self._format_deadline_window(start_day, end_day, tma_deadline_day)
        history_summary = self._summarize_history(upto_day=start_day - 1, window=14)
        weekend_summary = self._summarize_preceding_weekend(start_day)
        personality_anchor = self._personality_anchor_text()
        
        week_topic = week_content.get('topic', f"Week {week_num}")
        week_summary = week_content.get('description', '').strip()
        if len(week_summary) > 480:
            week_summary = week_summary[:477].rstrip() + "..."
        
        prompt = self.prompts['weekly_action_prompt'].format(
            week_num=week_num,
            start_day=start_day,
            end_day=end_day,
            week_topic=week_topic,
            week_summary=week_summary if week_summary else "Core learning materials released.",
            deadline_window=deadline_window,
            history_summary=history_summary,
            weekend_summary=weekend_summary,
            personality_anchor=personality_anchor
        )
        
        response = self.llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.85,
            max_tokens=768
        )
        
        if response is None:
            raise RuntimeError(f"LLM generation failed for week {week_num} of student {self.id}")
        
        try:
            weekly_plan = self._parse_weekly_plan(response, start_day, end_day)
        except Exception as parse_error:
            logger.error(
                "Failed to parse weekly plan for %s week %s. Raw response:\n%s",
                self.id,
                week_num,
                response
            )
            raise
        self._record_weekly_plan(week_num, weekly_plan)
        return weekly_plan
    
    def decide_assignment_submission(self, day_num: int, week_num: int) -> bool:
        """
        Decide whether to submit TMA 1 assignment
        
        Returns:
            True if will submit, False otherwise
        """
        days_remaining = 28 - day_num
        
        # Count recent activities
        recent_activities = [
            a for a in self.action_history
            if isinstance(a, dict) and isinstance(a.get('day'), int) and a['day'] >= day_num - 7
        ]
        activity_count = sum(1 for a in recent_activities if a['action'] != 'no_activity')
        assignment_focus = sum(1 for a in recent_activities if a['action'] == 'work_on_assignment')
        
        # Personality hints
        hints = {
            'high_performing': {
                'motivation_hint': 'You are highly motivated and committed to success',
                'time_management_hint': 'You started early and are well-prepared',
                'prior_knowledge_hint': 'You have strong understanding of the material'
            },
            'average': {
                'motivation_hint': 'You are reasonably motivated to do well',
                'time_management_hint': 'You have been working on it with decent time',
                'prior_knowledge_hint': 'You understand most of the material'
            },
            'struggling': {
                'motivation_hint': 'You are trying hard despite difficulties',
                'time_management_hint': 'You left it a bit late but are working on it',
                'prior_knowledge_hint': 'The material is challenging for you'
            },
            'at_risk': {
                'motivation_hint': 'You have low motivation or other priorities',
                'time_management_hint': 'You have barely started or haven\'t engaged much',
                'prior_knowledge_hint': 'You are struggling with the material'
            }
        }
        
        hint_set = hints[self.personality_type]
        
        assignment_focus_hint = f"{assignment_focus} assignment-focused actions in the past week"
        
        prompt = self.prompts['assignment_decision_prompt'].format(
            day_num=day_num,
            week_num=week_num,
            days_remaining=days_remaining,
            personality_type=self.personality_type,
            recent_activities=f"{activity_count} activities in past week",
            assignment_focus_hint=assignment_focus_hint,
            **hint_set
        )
        
        response = self.llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.7,
            max_tokens=32
        )
        
        # Parse yes/no
        if response:
            response_lower = response.lower().strip()
            if 'yes' in response_lower:
                return True
            elif 'no' in response_lower:
                return False
        
        # Fallback: based on personality and activity
        if self.personality_type == 'high_performing':
            return True
        elif self.personality_type == 'average':
            return activity_count > 5  # If somewhat active, submit
        elif self.personality_type == 'struggling':
            return activity_count > 8  # Need more effort to submit
        else:  # at_risk
            return False
    # --- New helper methods ---
    def _parse_weekly_plan(self, response: str, start_day: int, end_day: int) -> Dict[int, List[str]]:
        """
        Parse weekly JSON plan from LLM response.
        """
        cleaned = response.strip()
        if not cleaned:
            raise ValueError("Weekly plan response was empty")
        
        try:
            plan = json.loads(cleaned)
        except json.JSONDecodeError:
            json_match = re.search(r'\[[\s\S]*\]', cleaned)
            if not json_match:
                raise ValueError(f"Unable to parse weekly plan JSON: {cleaned}")
            plan = json.loads(json_match.group(0))
        
        if not isinstance(plan, list):
            raise ValueError("Weekly plan must be a JSON array")
        
        day_plan: Dict[int, List[str]] = {}
        valid_actions = {
            'view_lecture','read_resource','check_homepage','browse_subpage',
            'read_forum','post_forum','take_quiz','work_on_assignment',
            'check_glossary','visit_external_link','do_nothing'
        }
        
        fallback_day_pointer = start_day
        day_name_offsets = {name.lower(): idx for idx, name in enumerate(self.DAY_NAMES)}
        
        def _resolve_day(entry: Dict[str, Any]) -> int | None:
            alt_keys = ('day', 'course_day', 'day_number', 'day_index')
            for key in alt_keys:
                value = entry.get(key)
                if isinstance(value, int):
                    return value
                # Some models return numeric strings
                if isinstance(value, str) and value.isdigit():
                    return int(value)
            day_name = entry.get('day_name')
            if isinstance(day_name, str):
                normalized = day_name.strip().lower()
                if normalized in day_name_offsets:
                    return start_day + day_name_offsets[normalized]
            return None
        
        for entry_idx, entry in enumerate(plan):
            if not isinstance(entry, dict):
                continue
            
            day = _resolve_day(entry)
            if day is None:
                day = fallback_day_pointer
                fallback_day_pointer = min(fallback_day_pointer + 1, end_day + 1)
            actions = entry.get('actions', [])
            
            if not isinstance(day, int):
                continue
            
            if day < start_day or day > end_day:
                continue
            if not isinstance(actions, list):
                actions = []
            
            normalized_actions = []
            for act in actions:
                act_str = str(act)
                if act_str in valid_actions:
                    if act_str != 'do_nothing':
                        normalized_actions.append(act_str)
                else:
                    # Ignore invalid actions silently
                    continue
            
            day_plan[day] = normalized_actions
        
        # Fill missing days
        for day in range(start_day, end_day + 1):
            day_plan.setdefault(day, [])
        
        return day_plan
    
    def _record_weekly_plan(self, week_num: int, weekly_plan: Dict[int, List[str]]):
        """
        Persist generated actions into history for later prompts/decisions.
        """
        self.generated_week_plans[week_num] = weekly_plan
        for day, actions in sorted(weekly_plan.items()):
            if actions:
                for action in actions:
                    self.action_history.append({'day': day, 'week': week_num, 'action': action})
                    self.last_active_day = max(self.last_active_day, day)
            else:
                # Record explicit inactivity marker for summaries
                self.action_history.append({'day': day, 'week': week_num, 'action': 'no_activity'})
    
    def _summarize_history(self, upto_day: int, window: int = 14) -> str:
        """
        Create a compact description of the past `window` days before `upto_day`.
        """
        if upto_day <= 0 or not self.action_history:
            return "No activity yet."
        
        start_day = max(1, upto_day - window + 1)
        relevant = [
            a for a in self.action_history
            if isinstance(a, dict) and isinstance(a.get('day'), int) and start_day <= a['day'] <= upto_day
        ]
        if not relevant:
            return "No activity yet."
        
        by_day: Dict[int, List[str]] = defaultdict(list)
        for record in relevant:
            day_num = record.get('day')
            if isinstance(day_num, int):
                by_day[day_num].append(record.get('action', ''))
        
        summary_parts = []
        for day in sorted(by_day.keys()):
            actions = [act for act in by_day[day] if act != 'no_activity']
            if not actions:
                summary_parts.append(f"Day {day} ({self._day_name(day)}): inactive")
                continue
            counts = Counter(actions)
            action_summary = ', '.join(f"{act}×{count}" for act, count in counts.most_common())
            summary_parts.append(f"Day {day} ({self._day_name(day)}): {action_summary}")
        
        return '; '.join(summary_parts)
    
    def _summarize_preceding_weekend(self, start_day: int) -> str:
        """
        Summarize behaviour over the weekend immediately preceding start_day.
        """
        if start_day <= 2:
            return "No prior weekend."
        
        saturday = start_day - ((start_day - 1) % 7) - 1  # last Saturday
        sunday = saturday + 1
        if saturday < 1:
            return "No prior weekend."
        
        weekend_days = [d for d in (saturday, sunday) if d < start_day]
        if not weekend_days:
            return "No prior weekend."
        
        parts = []
        for day in weekend_days:
            actions = [
                a.get('action')
                for a in self.action_history
                if isinstance(a, dict)
                and a.get('day') == day
                and a.get('action') not in (None, 'no_activity')
            ]
            if not actions:
                parts.append(f"{self._day_name(day)}: inactive")
            else:
                counts = Counter(actions)
                parts.append(f"{self._day_name(day)}: " + ', '.join(f"{act}×{c}" for act, c in counts.most_common()))
        return '; '.join(parts) if parts else "Weekend inactive."
    
    def _personality_anchor_text(self) -> str:
        return (
            f"{self.personality_type} persona — motivation {self.profile['motivation']}, "
            f"time management {self.profile['time_management']}, "
            f"typical active days target {self.profile['typical_actions']} actions."
        )
    
    def _format_deadline_window(self, start_day: int, end_day: int, deadline_day: int) -> str:
        if end_day < deadline_day:
            days_start = max(0, deadline_day - start_day)
            days_end = max(0, deadline_day - end_day)
            return f"{days_end}–{days_start} days remaining"
        if start_day <= deadline_day <= end_day:
            return "deadline occurs this week"
        return "deadline already passed"
    
    def _day_name(self, day_num: int) -> str:
        return self.DAY_NAMES[(day_num - 1) % 7]

