"""
Map Agent Actions to OULAD-style VLE Events
"""

import random
from typing import Dict, Any, List


class ActionToVLEMapper:
    """
    Converts high-level agent actions to VLE activity events
    """
    
    # Map actions to OULAD activity types and click patterns
    ACTION_MAPPING = {
        'view_lecture': {
            'activity_type': 'oucontent',
            'click_range': (1, 3),
            'description': 'View course lecture content'
        },
        'read_resource': {
            'activity_type': 'resource',
            'click_range': (1, 2),
            'description': 'Read PDF or document resource'
        },
        'check_homepage': {
            'activity_type': 'homepage',
            'click_range': (1, 1),
            'description': 'Check course homepage'
        },
        'browse_subpage': {
            'activity_type': 'subpage',
            'click_range': (1, 2),
            'description': 'Browse course subpage'
        },
        'post_forum': {
            'activity_type': 'forumng',
            'click_range': (3, 6),
            'description': 'Post in discussion forum'
        },
        'read_forum': {
            'activity_type': 'forumng',
            'click_range': (1, 3),
            'description': 'Read forum discussions'
        },
        'take_quiz': {
            'activity_type': 'quiz',
            'click_range': (5, 15),
            'description': 'Complete a quiz'
        },
        'work_on_assignment': {
            'activity_type': 'oucollaborate',
            'click_range': (3, 10),
            'description': 'Work on assignment'
        },
        'check_glossary': {
            'activity_type': 'glossary',
            'click_range': (1, 2),
            'description': 'Check glossary'
        },
        'visit_external_link': {
            'activity_type': 'url',
            'click_range': (1, 1),
            'description': 'Visit external link'
        },
        'do_nothing': {
            'activity_type': None,
            'click_range': (0, 0),
            'description': 'No activity'
        }
    }
    
    def __init__(self, random_seed=42):
        """
        Args:
            random_seed: Random seed for reproducibility
        """
        self.rng = random.Random(random_seed)
    
    def convert_action_to_vle_event(
        self,
        student_id: str,
        action: str,
        day: int
    ) -> Dict[str, Any]:
        """
        Convert a single action to VLE event
        
        Args:
            student_id: Student ID
            action: Action name (e.g., 'view_lecture')
            day: Course day (1-56)
            
        Returns:
            VLE event dict with:
            - id_student
            - date (course day)
            - activity_type
            - sum_click
        """
        mapping = self.ACTION_MAPPING.get(action)
        
        if mapping is None or mapping['activity_type'] is None:
            # Invalid action or 'do_nothing'
            return None
        
        # Generate clicks
        min_clicks, max_clicks = mapping['click_range']
        clicks = self.rng.randint(min_clicks, max_clicks)
        
        return {
            'id_student': student_id,
            'date': day,
            'activity_type': mapping['activity_type'],
            'sum_click': clicks,
            'action_source': action,  # For debugging
            # Note: week, day_of_week, module_code will be added by course_simulator
        }
    
    def convert_daily_actions(
        self,
        student_id: str,
        actions: List[str],
        day: int
    ) -> List[Dict[str, Any]]:
        """
        Convert a list of daily actions to VLE events
        
        Args:
            student_id: Student ID
            actions: List of action names
            day: Course day
            
        Returns:
            List of VLE events
        """
        events = []
        
        for action in actions:
            event = self.convert_action_to_vle_event(student_id, action, day)
            if event is not None:
                events.append(event)
        
        return events
    
    def convert_to_oulad_format(self, events: List[Dict[str, Any]], module_code: str = None) -> List[Dict[str, Any]]:
        """
        Convert internal events to OULAD studentVle format
        
        OULAD format:
        - code_module: e.g., 'BBB'
        - code_presentation: e.g., '2014J'
        - id_student: student ID
        - id_site: VLE resource ID (we'll generate fake IDs)
        - date: course day
        - sum_click: click count
        
        Args:
            events: List of internal VLE events
            module_code: Module code (extracted from student_id if not provided)
            
        Returns:
            List of OULAD-format events
        """
        # Activity type to fake site ID mapping
        site_id_mapping = {
            'oucontent': 1000,
            'resource': 2000,
            'homepage': 3000,
            'subpage': 4000,
            'forumng': 5000,
            'quiz': 6000,
            'oucollaborate': 7000,
            'glossary': 8000,
            'url': 9000,
            'no_activity': 0  # Special marker for days with no activity
        }
        
        oulad_events = []
        
        for event in events:
            if event is None:
                continue
            
            activity_type = event['activity_type']
            
            # Handle 'no_activity' events specially
            if activity_type == 'no_activity':
                # For no_activity events, we still record them but with sum_click=0
                # This allows tracking of inactive days while keeping the data format consistent
                student_id = event['id_student']
                event_module_code = module_code if module_code else (
                    student_id.split('_')[0] if '_' in student_id else 'BBB'
                )
                
                oulad_event = {
                    'code_module': event_module_code,
                    'code_presentation': '2014J',
                    'id_student': student_id,
                    'id_site': 0,  # Special site ID for no_activity
                    'date': event['date'],
                    'sum_click': 0,  # No clicks on inactive days
                    'activity_type': 'no_activity'  # Extra field for our use
                }
                oulad_events.append(oulad_event)
                continue
            
            base_site_id = site_id_mapping.get(activity_type, 10000)
            
            # Generate a fake but consistent site ID
            # (In real OULAD, different resources have different IDs)
            site_id = base_site_id + self.rng.randint(1, 100)
            
            # Extract module from student_id for each event
            # Format: BBB_llm_student_0001
            student_id = event['id_student']
            event_module_code = module_code if module_code else (
                student_id.split('_')[0] if '_' in student_id else 'BBB'
            )
            
            oulad_event = {
                'code_module': event_module_code,
                'code_presentation': '2014J',
                'id_student': student_id,
                'id_site': site_id,
                'date': event['date'],
                'sum_click': event['sum_click'],
                'activity_type': activity_type  # Extra field for our use
            }
            
            oulad_events.append(oulad_event)
        
        return oulad_events

