"""
Convert student behavioral data to natural language descriptions for LLM input
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime


class BehaviorToTextConverter:
    """Convert structured student behavior data to narrative text"""
    
    def __init__(self):
        self.activity_descriptions = {
            'oucontent': 'course content materials',
            'resource': 'resource files',
            'url': 'external URLs',
            'homepage': 'course homepage',
            'subpage': 'course subpages',
            'glossary': 'glossary',
            'forumng': 'discussion forums',
            'oucollaborate': 'collaboration tools',
            'quiz': 'quizzes',
            'questionnaire': 'questionnaires',
            'page': 'content pages',
            'dataplus': 'data tools',
            'dualpane': 'dual-pane viewer',
            'externalquiz': 'external quizzes',
            'folder': 'folders',
            'htmlactivity': 'HTML activities',
            'ouwiki': 'wiki',
            'repeatactivity': 'repeated activities',
            'sharedsubpage': 'shared subpages'
        }
    
    def convert_student_to_narrative(self, 
                                     student_data: Dict[str, Any],
                                     vle_data: pd.DataFrame = None,
                                     include_demographics: bool = True,
                                     include_temporal: bool = True,
                                     include_statistics: bool = True,
                                     day_window: int = None) -> str:
        """
        Convert a student's data to narrative text
        
        Args:
            student_data: Dictionary containing student information
            vle_data: DataFrame with VLE interaction data
            include_demographics: Include demographic information
            include_temporal: Include temporal patterns
            include_statistics: Include statistical summaries
            day_window: Number of days to include (None for all)
            
        Returns:
            Natural language description of student behavior
        """
        narrative_parts = []
        
        # Header
        student_id = student_data.get('id_student', 'Unknown')
        narrative_parts.append(f"Student Profile (ID: {student_id})")
        narrative_parts.append("=" * 50)
        
        # Demographics
        if include_demographics and 'demographics' in student_data:
            demo_text = self._generate_demographic_text(student_data['demographics'])
            narrative_parts.append(f"\n**Demographics:**\n{demo_text}")
        
        # Course information
        if 'course_info' in student_data:
            course_text = self._generate_course_info_text(student_data['course_info'])
            narrative_parts.append(f"\n**Course Information:**\n{course_text}")
        
        # Learning behavior summary
        if include_statistics and 'vle_statistics' in student_data:
            stats_text = self._generate_statistics_text(student_data['vle_statistics'])
            narrative_parts.append(f"\n**Learning Activity Summary:**\n{stats_text}")
        
        # Temporal patterns
        if include_temporal and vle_data is not None:
            temporal_text = self._generate_temporal_patterns(vle_data, day_window)
            narrative_parts.append(f"\n**Temporal Engagement Patterns:**\n{temporal_text}")
        
        # Activity type breakdown
        if vle_data is not None and 'activity_type' in vle_data.columns:
            activity_text = self._generate_activity_breakdown(vle_data)
            narrative_parts.append(f"\n**Activity Type Breakdown:**\n{activity_text}")
        
        # Recent behavior
        if vle_data is not None:
            recent_text = self._generate_recent_behavior(vle_data, days=7)
            narrative_parts.append(f"\n**Recent Activity (Last 7 Days):**\n{recent_text}")
        
        return "\n".join(narrative_parts)
    
    def _generate_demographic_text(self, demo: Dict[str, Any]) -> str:
        """Generate demographic description"""
        lines = []
        
        if 'gender' in demo:
            lines.append(f"- Gender: {demo['gender']}")
        
        if 'age_band' in demo:
            lines.append(f"- Age: {demo['age_band']}")
        
        if 'highest_education' in demo:
            lines.append(f"- Education Level: {demo['highest_education']}")
        
        if 'region' in demo:
            lines.append(f"- Region: {demo['region']}")
        
        if 'num_of_prev_attempts' in demo:
            attempts = demo['num_of_prev_attempts']
            lines.append(f"- Previous Course Attempts: {attempts}")
        
        if 'disability' in demo:
            disability = "Yes" if demo['disability'] == 'Y' else "No"
            lines.append(f"- Has Disability: {disability}")
        
        return "\n".join(lines)
    
    def _generate_course_info_text(self, course_info: Dict[str, Any]) -> str:
        """Generate course information text"""
        lines = []
        
        if 'code_module' in course_info:
            lines.append(f"- Module: {course_info['code_module']}")
        
        if 'code_presentation' in course_info:
            lines.append(f"- Presentation: {course_info['code_presentation']}")
        
        if 'current_day' in course_info:
            lines.append(f"- Days into Course: {course_info['current_day']}")
        
        if 'assessment_name' in course_info:
            lines.append(f"- Target Assessment: {course_info['assessment_name']}")
        
        if 'days_until_assessment' in course_info:
            lines.append(f"- Days Until Assessment: {course_info['days_until_assessment']}")
        
        return "\n".join(lines)
    
    def _generate_statistics_text(self, stats: Dict[str, Any]) -> str:
        """Generate statistical summary"""
        lines = []
        
        # Overall engagement
        if 'total_clicks' in stats:
            lines.append(f"- Total VLE Interactions: {stats['total_clicks']:,}")
        
        if 'active_days' in stats:
            lines.append(f"- Active Days: {stats['active_days']}")
        
        if 'total_days' in stats:
            engagement_rate = (stats['active_days'] / stats['total_days'] * 100) if stats['total_days'] > 0 else 0
            lines.append(f"- Engagement Rate: {engagement_rate:.1f}%")
        
        # Login patterns
        if 'last_login' in stats:
            lines.append(f"- Days Since Last Login: {stats['last_login']}")
        
        if 'first_login' in stats:
            lines.append(f"- Days Until First Login: {stats['first_login']}")
        
        # Materials access
        if 'unique_materials' in stats:
            lines.append(f"- Unique Materials Accessed: {stats['unique_materials']}")
        
        if 'avg_clicks_per_day' in stats:
            lines.append(f"- Average Clicks per Active Day: {stats['avg_clicks_per_day']:.1f}")
        
        # Consecutive days
        if 'max_consecutive_days' in stats:
            lines.append(f"- Longest Consecutive Study Streak: {stats['max_consecutive_days']} days")
        
        return "\n".join(lines)
    
    def _generate_temporal_patterns(self, vle_data: pd.DataFrame, day_window: int = None) -> str:
        """Generate temporal pattern description"""
        if vle_data is empty or len(vle_data) == 0:
            return "- No VLE activity recorded"
        
        if day_window:
            vle_data = vle_data[vle_data['date'] >= -day_window]
        
        # Group by date
        daily_activity = vle_data.groupby('date')['sum_click'].sum().sort_index()
        
        if len(daily_activity) == 0:
            return "- No activity in the specified time window"
        
        lines = []
        
        # Activity trend
        if len(daily_activity) >= 3:
            recent_avg = daily_activity[-7:].mean() if len(daily_activity) >= 7 else daily_activity.mean()
            earlier_avg = daily_activity[:-7].mean() if len(daily_activity) > 7 else daily_activity.mean()
            
            if recent_avg > earlier_avg * 1.2:
                trend = "increasing (student is becoming more engaged)"
            elif recent_avg < earlier_avg * 0.8:
                trend = "decreasing (warning: student engagement is declining)"
            else:
                trend = "stable"
            
            lines.append(f"- Engagement Trend: {trend}")
        
        # Regularity
        gaps = daily_activity.index.to_series().diff().dropna()
        if len(gaps) > 0:
            avg_gap = gaps.mean()
            max_gap = gaps.max()
            
            if avg_gap <= 1.5:
                regularity = "very regular (daily engagement)"
            elif avg_gap <= 3:
                regularity = "moderately regular (active every 2-3 days)"
            else:
                regularity = f"irregular (average gap: {avg_gap:.1f} days, max gap: {max_gap:.0f} days)"
            
            lines.append(f"- Study Pattern: {regularity}")
        
        # Peak activity
        peak_day = daily_activity.idxmax()
        peak_clicks = daily_activity.max()
        lines.append(f"- Peak Activity: Day {peak_day} with {peak_clicks:.0f} clicks")
        
        return "\n".join(lines)
    
    def _generate_activity_breakdown(self, vle_data: pd.DataFrame) -> str:
        """Generate activity type breakdown"""
        if 'activity_type' not in vle_data.columns:
            return "- Activity breakdown not available"
        
        activity_summary = vle_data.groupby('activity_type')['sum_click'].sum().sort_values(ascending=False)
        
        if len(activity_summary) == 0:
            return "- No activities recorded"
        
        lines = []
        total_clicks = activity_summary.sum()
        
        for activity_type, clicks in activity_summary.head(5).items():
            percentage = (clicks / total_clicks * 100) if total_clicks > 0 else 0
            activity_name = self.activity_descriptions.get(activity_type, activity_type)
            lines.append(f"  • {activity_name}: {clicks:.0f} clicks ({percentage:.1f}%)")
        
        return "\n".join(lines)
    
    def _generate_recent_behavior(self, vle_data: pd.DataFrame, days: int = 7) -> str:
        """Generate recent behavior description"""
        if len(vle_data) == 0:
            return "- No recent activity"
        
        recent_data = vle_data.nlargest(days, 'date')
        
        if len(recent_data) == 0:
            return "- No activity in recent days"
        
        lines = []
        
        # Days active
        days_active = recent_data['date'].nunique()
        lines.append(f"- Active on {days_active} out of last {days} days")
        
        # Total interactions
        total_clicks = recent_data['sum_click'].sum()
        lines.append(f"- Total Interactions: {total_clicks:.0f}")
        
        # Average per day
        avg_clicks = total_clicks / days_active if days_active > 0 else 0
        lines.append(f"- Average Clicks per Active Day: {avg_clicks:.1f}")
        
        return "\n".join(lines)
    
    def convert_batch_to_narratives(self, 
                                    student_list: List[Dict[str, Any]], 
                                    vle_data_dict: Dict[int, pd.DataFrame] = None) -> List[str]:
        """
        Convert multiple students to narratives
        
        Args:
            student_list: List of student data dictionaries
            vle_data_dict: Dictionary mapping student_id to their VLE data
            
        Returns:
            List of narrative texts
        """
        narratives = []
        
        for student_data in student_list:
            student_id = student_data.get('id_student')
            vle_data = vle_data_dict.get(student_id) if vle_data_dict else None
            
            narrative = self.convert_student_to_narrative(
                student_data=student_data,
                vle_data=vle_data
            )
            narratives.append(narrative)
        
        return narratives


def create_peer_comparison_context(target_student: Dict[str, Any], 
                                   peer_students: List[Dict[str, Any]],
                                   num_peers: int = 5) -> str:
    """
    Create peer comparison context
    
    Args:
        target_student: Target student data
        peer_students: List of peer student data
        num_peers: Number of peers to include
        
    Returns:
        Peer comparison text
    """
    lines = ["\n**Peer Comparison:**"]
    
    # Calculate statistics from peers
    if len(peer_students) > 0:
        peer_clicks = [s.get('vle_statistics', {}).get('total_clicks', 0) for s in peer_students]
        peer_active_days = [s.get('vle_statistics', {}).get('active_days', 0) for s in peer_students]
        
        target_clicks = target_student.get('vle_statistics', {}).get('total_clicks', 0)
        target_active_days = target_student.get('vle_statistics', {}).get('active_days', 0)
        
        # Percentile calculation
        percentile_clicks = (sum(1 for c in peer_clicks if c < target_clicks) / len(peer_clicks) * 100) if peer_clicks else 0
        percentile_days = (sum(1 for d in peer_active_days if d < target_active_days) / len(peer_active_days) * 100) if peer_active_days else 0
        
        lines.append(f"- Total Clicks Percentile: {percentile_clicks:.1f}%")
        lines.append(f"- Active Days Percentile: {percentile_days:.1f}%")
        
        # Comparison to average
        avg_peer_clicks = np.mean(peer_clicks) if peer_clicks else 0
        avg_peer_days = np.mean(peer_active_days) if peer_active_days else 0
        
        clicks_ratio = (target_clicks / avg_peer_clicks) if avg_peer_clicks > 0 else 0
        days_ratio = (target_active_days / avg_peer_days) if avg_peer_days > 0 else 0
        
        if clicks_ratio < 0.5:
            lines.append(f"- ⚠️ WARNING: Student has {clicks_ratio:.1%} of average peer activity")
        elif clicks_ratio < 0.8:
            lines.append(f"- Student has {clicks_ratio:.1%} of average peer activity (below average)")
        else:
            lines.append(f"- Student has {clicks_ratio:.1%} of average peer activity")
    
    return "\n".join(lines)





