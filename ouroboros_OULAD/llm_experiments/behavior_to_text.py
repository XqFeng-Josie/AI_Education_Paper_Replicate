"""
BehaviorToTextConverter: Convert student numerical features to LLM-readable text
"""

import logging
from typing import Dict, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class BehaviorToTextConverter:
    """
    Convert student behavior data from numerical features to natural language descriptions
    """
    
    def __init__(self, 
                 include_peer_context: bool = True,
                 days_to_cutoff: int = 0):
        """
        Initialize converter
        
        Args:
            include_peer_context: Whether to include peer comparison context
            days_to_cutoff: Days before assignment deadline (for temporal context)
        """
        self.include_peer_context = include_peer_context
        self.days_to_cutoff = days_to_cutoff
        self.cohort_stats = None
        
    def set_cohort_statistics(self, train_df: pd.DataFrame):
        """
        Compute cohort statistics for peer comparison
        
        Args:
            train_df: Training DataFrame containing all students
        """
        self.cohort_stats = self._compute_cohort_statistics(train_df)
        logger.info(f"Computed cohort statistics from {len(train_df)} students")
    
    def convert_to_text(self, row: pd.Series) -> str:
        """
        Convert student's feature data to text description
        
        Args:
            row: Student's feature data (pandas Series)
            
        Returns:
            Detailed natural language description of student behavior
        """
        student_id = row.get('id_student', 'unknown')
        
        # Start with header
        narrative = f"Student ID: {student_id}\n"
        narrative += f"**Prediction Time**: {self.days_to_cutoff} days before assignment deadline\n\n"
        
        # 1. Academic Performance & VLE Activity
        narrative += self._describe_vle_activity(row)
        
        # 2. Behavioral Patterns (temporal analysis)
        narrative += self._describe_behavioral_patterns(row)
        
        # 3. Engagement Trends
        narrative += self._describe_engagement_trends(row)
        
        # 4. Demographics (optional, if available)
        narrative += self._describe_demographics(row)
        
        return narrative.strip()
    
    def _describe_vle_activity(self, row: pd.Series) -> str:
        """Describe VLE (Virtual Learning Environment) activity"""
        text = "## Academic Performance & VLE Activity\n"
        
        # Total clicks
        sum_click = row.get('sum_click', 0)
        text += f"- **Total VLE Clicks**: {int(sum_click)}\n"
        
        # Peer comparison for total clicks
        if self.include_peer_context and self.cohort_stats:
            percentile = self._get_percentile(sum_click, self.cohort_stats.get('sum_click_values', []))
            median = self.cohort_stats.get('sum_click_median', 0)
            mean = self.cohort_stats.get('sum_click_mean', 0)
            
            comparison = self._get_comparison_text(sum_click, median)
            text += f"  - Cohort median: {int(median)}, mean: {int(mean)}\n"
            text += f"  - Student percentile: {int(percentile)}% (clicks are {comparison} cohort median)\n"
        
        # Active days
        num_days = row.get('num_days', 0)
        text += f"- **Active Learning Days**: {int(num_days)}\n"
        
        # Consecutive days
        consecutive_days = row.get('consecutive_days', 0)
        if consecutive_days > 0:
            text += f"- **Consecutive Active Days**: {int(consecutive_days)} days of continuous engagement\n"
        
        # Last login
        last_login_days = row.get('last_login_days', None)
        if last_login_days is not None and last_login_days >= 0:
            text += f"- **Last Login**: {int(last_login_days)} days before deadline\n"
            if last_login_days > 7:
                text += f"  - ⚠️ Warning: No activity in the last {int(last_login_days)} days\n"
        
        text += "\n"
        return text
    
    def _describe_behavioral_patterns(self, row: pd.Series) -> str:
        """Describe day-by-day behavioral patterns"""
        text = "## Behavioral Patterns\n"
        
        # Find all day columns (e.g., day_-30_sum_click, day_-29_sum_click, ...)
        day_columns = sorted([col for col in row.index if col.startswith('day_') and col.endswith('_sum_click')])
        
        if not day_columns:
            text += "- No day-by-day activity data available\n\n"
            return text
        
        # Extract day activities
        day_activities = []
        for col in day_columns:
            try:
                day_num = int(col.split('_')[1])  # Extract day number from "day_-30_sum_click"
                clicks = row.get(col, 0)
                day_activities.append((day_num, clicks))
            except:
                continue
        
        if not day_activities:
            text += "- No day-by-day activity data available\n\n"
            return text
        
        # Sort by day
        day_activities = sorted(day_activities, key=lambda x: x[0])
        
        # Recent activity (last 7 days)
        recent_days = [clicks for day, clicks in day_activities[-7:]]
        if recent_days:
            avg_recent = sum(recent_days) / len(recent_days)
            active_recent_days = sum(1 for c in recent_days if c > 0)
            text += f"- **Recent Activity (Last 7 Days)**:\n"
            text += f"  - Average: {avg_recent:.1f} clicks/day\n"
            text += f"  - Active days: {active_recent_days}/7 days\n"
        
        # Activity distribution
        all_clicks = [clicks for _, clicks in day_activities]
        active_days_count = sum(1 for c in all_clicks if c > 0)
        max_clicks_day = max(all_clicks) if all_clicks else 0
        avg_clicks = sum(all_clicks) / len(all_clicks) if all_clicks else 0
        
        text += f"- **Overall Activity Pattern**:\n"
        text += f"  - Average activity: {avg_clicks:.1f} clicks/day\n"
        text += f"  - Peak day activity: {int(max_clicks_day)} clicks\n"
        text += f"  - Active days ratio: {active_days_count}/{len(all_clicks)} days ({active_days_count/len(all_clicks)*100:.1f}%)\n"
        
        text += "\n"
        return text
    
    def _describe_engagement_trends(self, row: pd.Series) -> str:
        """Describe temporal engagement trends (increasing/decreasing/stable)"""
        text = "## Engagement Trends\n"
        
        # Find all day columns
        day_columns = sorted([col for col in row.index if col.startswith('day_') and col.endswith('_sum_click')])
        
        if len(day_columns) < 14:
            text += "- Insufficient data for trend analysis\n\n"
            return text
        
        # Extract activities
        day_activities = []
        for col in day_columns:
            try:
                day_num = int(col.split('_')[1])
                clicks = row.get(col, 0)
                day_activities.append((day_num, clicks))
            except:
                continue
        
        if not day_activities:
            text += "- Insufficient data for trend analysis\n\n"
            return text
        
        day_activities = sorted(day_activities, key=lambda x: x[0])
        
        # Compare early vs late period
        n_days = len(day_activities)
        early_period = [clicks for _, clicks in day_activities[:n_days//2]]
        late_period = [clicks for _, clicks in day_activities[n_days//2:]]
        
        early_avg = sum(early_period) / len(early_period) if early_period else 0
        late_avg = sum(late_period) / len(late_period) if late_period else 0
        
        # Determine trend
        if late_avg > early_avg * 1.3:
            trend = "**Increasing** ↗️"
            interpretation = "Student engagement is growing over time (positive signal)"
        elif late_avg < early_avg * 0.7:
            trend = "**Decreasing** ↘️"
            interpretation = "Student engagement is declining over time (warning sign)"
        else:
            trend = "**Stable** →"
            interpretation = "Student maintains consistent engagement"
        
        text += f"- **Trend**: {trend}\n"
        text += f"  - Early period average: {early_avg:.1f} clicks/day\n"
        text += f"  - Late period average: {late_avg:.1f} clicks/day\n"
        text += f"  - Interpretation: {interpretation}\n"
        
        text += "\n"
        return text
    
    def _describe_demographics(self, row: pd.Series) -> str:
        """Describe demographic information"""
        # Check if demographic data exists
        demog_fields = ['highest_education', 'age_band', 'num_of_prev_attempts', 
                       'studied_credits', 'gender', 'region', 'disability']
        
        has_demog = any(field in row.index for field in demog_fields)
        
        if not has_demog:
            return ""
        
        text = "## Demographics\n"
        
        if 'highest_education' in row.index:
            edu = row.get('highest_education', 'Unknown')
            text += f"- **Education Level**: {edu}\n"
        
        if 'age_band' in row.index:
            age = row.get('age_band', 'Unknown')
            text += f"- **Age Band**: {age}\n"
        
        if 'num_of_prev_attempts' in row.index:
            prev_attempts = row.get('num_of_prev_attempts', 0)
            if prev_attempts > 0:
                text += f"- **Previous Attempts**: {int(prev_attempts)} (indicates re-taking course)\n"
        
        if 'studied_credits' in row.index:
            credits = row.get('studied_credits', 0)
            text += f"- **Studied Credits**: {int(credits)}\n"
        
        text += "\n"
        return text
    
    def _compute_cohort_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Compute cohort statistics for peer comparison
        
        Args:
            df: DataFrame containing all students in cohort
            
        Returns:
            Dictionary of statistics
        """
        stats = {}
        
        # Total clicks statistics
        if 'sum_click' in df.columns:
            stats['sum_click_median'] = df['sum_click'].median()
            stats['sum_click_mean'] = df['sum_click'].mean()
            stats['sum_click_std'] = df['sum_click'].std()
            stats['sum_click_values'] = df['sum_click'].values.tolist()
        
        # Active days statistics
        if 'num_days' in df.columns:
            stats['num_days_median'] = df['num_days'].median()
            stats['num_days_mean'] = df['num_days'].mean()
        
        return stats
    
    def _get_percentile(self, value: float, cohort_values: list) -> float:
        """
        Calculate percentile of value in cohort
        
        Args:
            value: Student's value
            cohort_values: List of all cohort values
            
        Returns:
            Percentile (0-100)
        """
        if not cohort_values:
            return 50.0
        
        cohort_array = np.array(cohort_values)
        percentile = (cohort_array < value).sum() / len(cohort_array) * 100
        return percentile
    
    def _get_comparison_text(self, value: float, reference: float) -> str:
        """
        Generate comparison text (above/below/similar to)
        
        Args:
            value: Student's value
            reference: Reference value (e.g., median)
            
        Returns:
            Comparison text
        """
        if value > reference * 1.2:
            return "significantly above"
        elif value > reference * 1.05:
            return "above"
        elif value < reference * 0.8:
            return "significantly below"
        elif value < reference * 0.95:
            return "below"
        else:
            return "similar to"

