"""
8-Week Course Simulator with LLM Agents
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import os
import json

from ..llm_client import LlamaClient
from ..agents import InstructorAgent, StudentAgent
from .action_to_vle_mapper import ActionToVLEMapper

logger = logging.getLogger(__name__)


class CourseSimulator:
    """
    Simulate an 8-week university course with Instructor and Student agents
    """
    
    def __init__(
        self,
        n_students: int,
        personality_distribution: Dict[str, float] = None,
        llama_server_url: str = "http://localhost:8000",
        random_seed: int = 42,
        module_code: str = "BBB",
        stream_dir: Path | None = None,
        resume: bool = False,
        id_offset: int = 0
    ):
        """
        Args:
            n_students: Number of students to simulate
            personality_distribution: Dict with proportions for each personality type
            llama_server_url: URL of Llama server
            random_seed: Random seed
            module_code: Module code (BBB, DDD, EEE, FFF)
            stream_dir: If provided, write raw events/interaction logs incrementally here
            resume: If True, resume from checkpoint in stream_dir
            id_offset: Starting offset for student indices to ensure unique IDs across appends
        """
        self.n_students = n_students
        self.random_seed = random_seed
        self.module_code = module_code
        self.stream_dir = Path(stream_dir) if stream_dir else None
        self.resume = resume
        self.id_offset = max(0, int(id_offset))
        
        # Default personality distribution (from data-driven config)
        if personality_distribution is None:
            personality_distribution = {
                'high_performing': 0.087,
                'average': 0.084,
                'struggling': 0.171,
                'at_risk': 0.659
            }
        
        self.personality_dist = personality_distribution
        
        # Initialize LLM client
        logger.info(f"Connecting to Llama server at {llama_server_url}")
        self.llm_client = LlamaClient(base_url=llama_server_url)
        
        if not self.llm_client.is_healthy():
            raise RuntimeError(
                f"Llama server not healthy at {llama_server_url}. "
                "Please start the server: python llm/server/llama_server.py"
            )
        
        # Initialize agents
        logger.info("Initializing agents...")
        self.instructor = InstructorAgent(self.llm_client)
        self.students = self._create_students()
        
        # VLE mapper
        self.vle_mapper = ActionToVLEMapper(random_seed=random_seed)
        
        # Storage
        self.vle_events = []
        self.interaction_log = []
        
        # Streaming files and checkpoint
        self._ckpt = {'week': 1, 'day_of_week': 1, 'current_student_idx': 0}
        self._events_fp = None
        self._interactions_fp = None
        self._ckpt_path = None
        if self.stream_dir:
            self.stream_dir.mkdir(parents=True, exist_ok=True)
            self._ckpt_path = self.stream_dir / f"{self.module_code}_checkpoint.json"
            events_path = self.stream_dir / f"{self.module_code}_events_raw.ndjson"
            interactions_path = self.stream_dir / f"{self.module_code}_interactions.ndjson"
            # Open append mode
            self._events_fp = open(events_path, 'a', buffering=1)
            self._interactions_fp = open(interactions_path, 'a', buffering=1)
            # Load checkpoint if resume
            if self.resume and self._ckpt_path.exists():
                try:
                    self._ckpt = json.loads(self._ckpt_path.read_text())
                    # Ensure all required fields exist
                    if 'current_student_idx' not in self._ckpt:
                        self._ckpt['current_student_idx'] = 0
                    logger.info(f"Resuming {self.module_code} from checkpoint: {self._ckpt}")
                except Exception as e:
                    logger.warning(f"Failed to read checkpoint, starting fresh: {e}")
                    self._ckpt = {'week': 1, 'day_of_week': 1, 'current_student_idx': 0}
    
    def _stream_event_batch(self, events: List[Dict[str, Any]]):
        if not self._events_fp:
            return
        for ev in events:
            self._events_fp.write(json.dumps(ev, ensure_ascii=False) + "\n")
    
    def _stream_interaction(self, record: Dict[str, Any]):
        if not self._interactions_fp:
            return
        self._interactions_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    def _save_checkpoint(self, week: int, day_of_week: int, current_student_idx: int = 0):
        if not self._ckpt_path:
            return
        self._ckpt = {
            'week': week,
            'day_of_week': day_of_week,
            'current_student_idx': current_student_idx,
            'module_code': self.module_code,
            'n_students': self.n_students,
            'id_offset': self.id_offset
        }
        try:
            self._ckpt_path.write_text(json.dumps(self._ckpt, indent=2))
        except Exception as e:
            logger.warning(f"Failed to write checkpoint: {e}")
    
    def _close_streams(self):
        try:
            if self._events_fp:
                self._events_fp.close()
            if self._interactions_fp:
                self._interactions_fp.close()
        except Exception:
            pass
    
    def _create_students(self) -> List[StudentAgent]:
        """Create N student agents with personality distribution"""
        import numpy as np
        
        students = []
        
        # Sample personalities
        personality_types = list(self.personality_dist.keys())
        proportions = list(self.personality_dist.values())
        
        # Normalize proportions to ensure they sum to exactly 1.0
        proportions = np.array(proportions)
        proportions = proportions / proportions.sum()
        
        rng = np.random.default_rng(self.random_seed)
        sampled_personalities = rng.choice(
            personality_types,
            size=self.n_students,
            p=proportions
        )
        
        # Create agents
        for i, personality in enumerate(sampled_personalities):
            global_index = self.id_offset + i
            student_id = f"{self.module_code}_llm_student_{global_index:06d}"
            student = StudentAgent(
                student_id=student_id,
                personality_type=personality,
                llm_client=self.llm_client
            )
            students.append(student)
        
        # Log distribution
        from collections import Counter
        dist_count = Counter(sampled_personalities)
        logger.info(f"Created {self.n_students} students:")
        for ptype, count in dist_count.items():
            logger.info(f"  - {ptype}: {count} ({count/self.n_students*100:.1f}%)")
        
        return students
    
    def simulate_8_weeks(self) -> Dict[str, Any]:
        """
        Run 8-week simulation
        
        Returns:
            dict with simulation results
        """
        logger.info("=" * 60)
        logger.info("Starting 8-Week Course Simulation with LLM Agents")
        logger.info("=" * 60)
        
        tma1_deadline_day = 28  # End of Week 4
        
        # Prepare instructor content upfront
        weekly_contents: Dict[int, Dict[str, Any]] = {}
        logger.info(f"\n{'='*60}")
        logger.info("Preparing instructor weekly content")
        logger.info(f"{'='*60}")
        for week in range(1, 9):
            weekly_content = self.instructor.post_weekly_content(week)
            weekly_contents[week] = weekly_content
            logger.info(f"Week {week}: {weekly_content['topic']}")
            interaction_record = {
                'type': 'instructor_post',
                'week': week,
                'content': weekly_content
            }
            self.interaction_log.append(interaction_record)
            self._stream_interaction(interaction_record)
        
        total_submissions = 0
        
        # Simulate student trajectories one student at a time
        for student_idx, student in enumerate(tqdm(self.students, desc="Simulating students", leave=False)):
            logger.info(f"\n{'-'*58}")
            logger.info(f"Simulating full trajectory for {student.id} ({student.personality_type})")
            
            start_week = 1
            if self.resume:
                ckpt_student = self._ckpt.get('current_student_idx', 0)
                ckpt_week = self._ckpt.get('week', 1)
                if student_idx < ckpt_student:
                    logger.info(f"Skipping {student.id} (already completed via checkpoint)")
                    continue
                if student_idx == ckpt_student:
                    start_week = ckpt_week
                    logger.info(f"Resuming {student.id} from week {start_week}")
            
            # 按学生粒度存储：先收集该学生的所有事件，完成后再写入
            student_events = []  # 该学生的所有事件
            student_interactions = []  # 该学生的所有交互记录
            student_completed = False  # 标记学生是否成功完成
            
            try:
                for week in range(start_week, 9):
                    start_day = (week - 1) * 7 + 1
                    end_day = start_day + 6
                    
                    if self.stream_dir:
                        self._save_checkpoint(week, 1, student_idx)
                    
                    weekly_content = weekly_contents[week]
                    
                    try:
                        weekly_plan = student.plan_weekly_actions(
                            week_num=week,
                            start_day=start_day,
                            week_content=weekly_content,
                            tma_deadline_day=tma1_deadline_day
                        )
                    except Exception as e:
                        logger.error(f"  ❌ Failed to generate weekly plan for {student.id} (week {week}): {e}")
                        error_record = {
                            'type': 'student_week_plan_error',
                            'student_id': student.id,
                            'week': week,
                            'error': str(e),
                            'status': 'error'
                        }
                        student_interactions.append(error_record)
                        # Abort remaining weeks for this student to allow graceful continuation
                        raise  # 抛出异常，让外层catch处理
                
                    # Log weekly plan
                    weekly_record = {
                        'type': 'student_week_plan',
                        'student_id': student.id,
                        'week': week,
                        'start_day': start_day,
                        'end_day': end_day,
                        'actions_by_day': weekly_plan
                    }
                    student_interactions.append(weekly_record)
                    
                    weekly_action_counter = 0
                    
                    for day_offset in range(7):
                        day = start_day + day_offset
                        day_actions = weekly_plan.get(day, [])
                        day_of_week = day_offset + 1
                        
                        events = self.vle_mapper.convert_daily_actions(
                            student_id=student.id,
                            actions=day_actions,
                            day=day
                        )
                        
                        if not events:
                            no_activity_event = {
                                'id_student': student.id,
                                'date': day,
                                'week': week,
                                'day_of_week': day_of_week,
                                'module_code': self.module_code,
                                'activity_type': 'no_activity',
                                'sum_click': 0,
                                'action_source': 'no_activity',
                                'generation_status': 'success'
                            }
                            events = [no_activity_event]
                        else:
                            for event in events:
                                event['week'] = week
                                event['day_of_week'] = day_of_week
                                event['module_code'] = self.module_code
                                event['generation_status'] = 'success'
                        
                        weekly_action_counter += len(day_actions) if day_actions else 1
                        student_events.extend(events)  # 收集到学生事件列表
                        
                        interaction = {
                            'type': 'student_daily_actions',
                            'student_id': student.id,
                            'day': day,
                            'week': week,
                            'day_of_week': day_of_week,
                            'actions': day_actions,
                            'events_count': len(events),
                            'status': 'success'
                        }
                        student_interactions.append(interaction)
                    
                    logger.info(f"  Week {week}: {weekly_action_counter} logged actions/events for {student.id}")
                    
                    # Handle TMA submission decision at the end of Week 4
                    if week == 4:
                        will_submit = student.decide_assignment_submission(
                            day_num=tma1_deadline_day,
                            week_num=week
                        )
                        submission_record = {
                            'type': 'assignment_submission',
                            'student_id': student.id,
                            'day': tma1_deadline_day,
                            'week': week,
                            'submitted': bool(will_submit)
                        }
                        student_interactions.append(submission_record)
                        
                        if will_submit:
                            total_submissions += 1
                            submission_event = {
                                'id_student': student.id,
                                'date': tma1_deadline_day,
                                'week': week,
                                'day_of_week': 7,
                                'module_code': self.module_code,
                                'activity_type': 'oucollaborate',
                                'sum_click': 10,
                                'action_source': 'submit_assignment',
                                'generation_status': 'success'
                            }
                            student_events.append(submission_event)
                            logger.info(f"    → {student.id} submitted TMA 1")
                        else:
                            logger.info(f"    → {student.id} did NOT submit TMA 1")
                    
                    # Advance checkpoint to next week for this student
                    if self.stream_dir:
                        next_week = week + 1 if week < 8 else 1
                        self._save_checkpoint(next_week, 1, student_idx)
                
                # 学生成功完成所有8周
                student_completed = True
                
            except Exception as e:
                # 学生生成失败，不保存任何数据
                logger.error(f"  ❌ Student {student.id} failed, not saving any data: {e}")
                student_completed = False
            
            # 按学生粒度存储：只有成功完成的学生才写入数据
            if student_completed and student_events:
                # 验证学生是否有完整的56天数据
                student_days = set(ev.get('date') for ev in student_events if ev.get('date') is not None)
                expected_days = set(range(1, 57))
                if student_days == expected_days:
                    # 写入该学生的所有事件
                    self._stream_event_batch(student_events)
                    # 写入该学生的所有交互记录
                    for interaction in student_interactions:
                        self._stream_interaction(interaction)
                    
                    # 更新内存中的事件列表（用于非流式模式）
                    self.vle_events.extend(student_events)
                    self.interaction_log.extend(student_interactions)
                    
                    logger.info(f"  ✅ {student.id} completed successfully, saved {len(student_events)} events")
                else:
                    missing_days = expected_days - student_days
                    logger.warning(f"  ⚠️  {student.id} missing days {sorted(list(missing_days))[:10]}, not saving")
                    student_completed = False
            
            # 更新checkpoint：移动到下一个学生
            if self.stream_dir:
                if student_completed:
                    # 学生完成，移动到下一个
                    self._save_checkpoint(week=1, day_of_week=1, current_student_idx=student_idx + 1)
                else:
                    # 学生失败，保持当前checkpoint，下次继续尝试
                    logger.warning(f"  ⚠️  {student.id} failed, checkpoint remains at student_idx={student_idx}")
        
        if self.n_students > 0:
            submission_rate = total_submissions / self.n_students
            logger.info(f"\nTMA 1 submissions: {total_submissions}/{self.n_students} ({submission_rate:.1%})")
        
        logger.info(f"\n{'='*60}")
        logger.info("Simulation Complete!")
        logger.info(f"{'='*60}")
        logger.info(f"Total VLE events generated: {len(self.vle_events)}")
        # Final checkpoint mark completion
        self._save_checkpoint(week=9, day_of_week=1, current_student_idx=0)
        self._close_streams()
        
        return {
            'n_students': self.n_students,
            'total_events': len(self.vle_events),
            'total_interactions': len(self.interaction_log),
            'vle_events': self.vle_events,
            'interaction_log': self.interaction_log
        }
    
    def export_vle_logs(self, output_path: Path):
        """
        Export VLE events in OULAD format
        
        Args:
            output_path: Path to save CSV file
        """
        import pandas as pd
        
        # Convert to OULAD format
        oulad_events = self.vle_mapper.convert_to_oulad_format(self.vle_events)
        
        if not oulad_events:
            logger.warning("No VLE events to export")
            return
        
        # Save as CSV
        df = pd.DataFrame(oulad_events)
        df.to_csv(output_path, index=False)
        
        logger.info(f"✅ Exported {len(df)} VLE events to {output_path}")
        
        # Statistics
        logger.info(f"\nVLE Event Statistics:")
        logger.info(f"  Activity types:")
        for atype, count in df['activity_type'].value_counts().items():
            logger.info(f"    - {atype}: {count}")
        
        # Clicks per student
        clicks_per_student = df.groupby('id_student')['sum_click'].sum()
        logger.info(f"\n  Clicks per student:")
        logger.info(f"    - Mean: {clicks_per_student.mean():.1f}")
        logger.info(f"    - Median: {clicks_per_student.median():.1f}")
        logger.info(f"    - Min: {clicks_per_student.min()}")
        logger.info(f"    - Max: {clicks_per_student.max()}")
    
    def export_interaction_log(self, output_path: Path):
        """Export full interaction log as JSON"""
        with open(output_path, 'w') as f:
            json.dump(self.interaction_log, f, indent=2)
        
        logger.info(f"✅ Exported interaction log to {output_path}")

