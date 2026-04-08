"""
Construction Superintendent OpenEnv Environment.

Implements the OpenEnv standard API:
- reset(task_level, seed) -> Observation
- step(action) -> Tuple[Observation, reward, done, info]
- state() -> Dict (full internal state)

Domain: AI agent manages a real construction project schedule,
responding to disruptions (weather, material shortages, equipment failures)
to minimise delay and cost overrun.
"""

from __future__ import annotations

import copy
import uuid
from typing import Any, Dict, List, Optional, Tuple

from env.disruption import DISRUPTION_FACTORIES, DisruptionEvent
from env.models import (
    Action,
    ActionType,
    DisruptionObservation,
    DisruptionType,
    Observation,
    ProjectMetrics,
    StepResult,
    TaskObservation,
    TaskStatus,
)
from env.project import PROJECT_FACTORIES, TaskNode, ProjectTemplate
from env.scheduler import apply_schedule

# =============================================================================
# Reward weights
# =============================================================================

DELAY_PENALTY_PER_DAY = -2.0
COST_PENALTY_PER_1K = -0.5
TASK_COMPLETION_BONUS = 5.0
ON_TIME_BONUS = 30.0
PROJECT_COMPLETION_BONUS = 50.0
DISRUPTION_RESOLUTION_BONUS = 3.0
BUDGET_OVERRUN_PENALTY = -10.0
INVALID_ACTION_PENALTY = -1.0
EXPEDITE_COST_PER_DAY_PER_RESOURCE = 800.0   # extra $ per day per extra unit

# =============================================================================
# Class
# =============================================================================

class ConstructionEnv:
    """
    OpenEnv-compliant construction project management environment.

    The simulation runs as an event-driven loop:
    1. On reset(), the initial schedule is computed and the first disruption is queued.
    2. Each step(), the agent submits ONE action.
    3. The simulation advances to the next event (disruption fires or a task completes).
    4. The episode ends when all tasks complete or max_steps is reached.

    Observation: Full project state (tasks, disruptions, metrics)
    Actions: expedite_task, delay_task, reassign_resources, noop
    Reward: Dense signal each step (-delay, -cost, +completions, +on-time bonus)
    """

    def __init__(self) -> None:
        self._task_level: str = "easy"
        
        self._seed: Optional[int] = None
        self._template: Optional[ProjectTemplate] = None
        
        self._tasks: Dict[str, TaskNode] = {}
        self._disruptions: List[DisruptionEvent] = []
        
        self._current_day: int = 0
        self._episode_step: int = 0
        self._original_end_day: int = 0
        self._budget_used: float = 0.0
        self._expedite_extra_cost: float = 0.0 # track resource cost separately
        self._done: bool = False
        
        self._session_id: str = str(uuid.uuid4())
        self._history: List[Dict] = []
        
    # =========================================================================
    # OpenEnv API
    # =========================================================================

    def reset(self, task_level: str = "easy", seed: Optional[int] = None) -> Observation:
        """Reset environment and return initial observation."""
        if task_level not in PROJECT_FACTORIES:
            raise ValueError(f"Unknown task_level '{task_level}'. Choose: easy | medium | hard")
            
        self._task_level = task_level
        self._seed = seed
        self._session_id = str(uuid.uuid4())
        
        self._current_day = 0
        self._episode_step = 0
        self._budget_used = 0.0
        self._expedite_extra_cost = 0.0
        self._done = False
        self._history = []
        
        # Clone fresh Project Template
        self._template = PROJECT_FACTORIES[task_level]()
        self._tasks = {t.id: t.clone() for t in self._template.tasks}
        
        # Reset all task states
        for task in self._tasks.values():
            task.status = "pending"
            task.extra_resources = 0
            task.delay_days = 0
            task.progress_pct = 0.0
            
        # Compute baseline schedule
        self._original_end_day = apply_schedule(self._tasks)
        
        # Clone disruptions for this episode
        self._disruptions = copy.deepcopy(DISRUPTION_FACTORIES[task_level]())
        
        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Apply action, advance simulation, return (obs, reward, done, info)."""
        if self._done:
            obs = self._build_observation()
            obs.terminal_message = "Episode already finished. Call reset()."
            return obs, 0.0, True, {"warning": "episode_done"}
            
        self._episode_step += 1
        
        reward_components: Dict[str, float] = {
            "delay_penalty": 0.0,
            "cost_penalty": 0.0,
            "completion_bonus": 0.0,
            "disruption_resolution_bonus": 0.0,
            "project_completion_bonus": 0.0,
            "invalid_action_penalty": 0.0,
        }
        
        info: Dict[str, Any] = {"action": action.dict(), "events": []}
        
        # --- Apply agent action ---
        valid_action, msg = self._apply_action(action, reward_components, info)
        if not valid_action:
            reward_components["invalid_action_penalty"] = INVALID_ACTION_PENALTY
            
        # --- Advance simulation clock ---
        days_advanced = self._advance_simulation(info)
        
        # --- Budget tracking ---
        step_cost = self._compute_step_cost(days_advanced)
        self._budget_used += step_cost
        
        if self._template:
            if self._budget_used > self._template.budget:
                overrun = self._budget_used - self._template.budget
                budget_penalty = BUDGET_OVERRUN_PENALTY * (overrun / 10_000)
                reward_components["cost_penalty"] += budget_penalty
                
        # --- Recompute schedule after changes ---
        new_end_day = apply_schedule(self._tasks)
        
        # --- Delay penalty (per day of current delay) ---
        current_delay = max(0, new_end_day - self._original_end_day)
        reward_components["delay_penalty"] = DELAY_PENALTY_PER_DAY * current_delay
        
        # --- Cost penalty for extra resources ---
        if self._expedite_extra_cost > 0:
            reward_components["cost_penalty"] += COST_PENALTY_PER_1K * (self._expedite_extra_cost / 1000.0)
            self._expedite_extra_cost = 0.0  # reset after applying penalty
            
        # --- Project completion check ---
        if self._all_tasks_complete():
            reward_components["project_completion_bonus"] = PROJECT_COMPLETION_BONUS
            if new_end_day <= self._original_end_day:
                reward_components["project_completion_bonus"] += ON_TIME_BONUS
            self._done = True
            
        # --- Max steps check ---
        max_steps = self._template.max_steps if self._template else 20
        if self._episode_step >= max_steps and not self._done:
            self._done = True
            info["timeout"] = True
            
        total_reward = sum(reward_components.values())
        
        info.update({
            "reward_breakdown": reward_components,
            "reward_detail": msg,
            "current_day": self._current_day,
            "projected_end_day": new_end_day,
            "original_end_day": self._original_end_day,
            "delay_days": current_delay,
            "budget_used": self._budget_used,
            "budget_total": self._template.budget if self._template else 0,
            "step": self._episode_step,
        })
        
        self._history.append({"step": self._episode_step, "action": action.dict(), "reward": total_reward})
        
        obs = self._build_observation()
        return obs, total_reward, self._done, info

    def state(self) -> Dict[str, Any]:
        """Return full internal environment state (for debugging / grading)."""
        projected_end = apply_schedule(self._tasks) if self._tasks else 0
        return {
            "session_id": self._session_id,
            "task_level": self._task_level,
            "current_projected_end_day": projected_end,
            "episode_step": self._episode_step,
            "current_day": self._current_day,
            "original_end_day": self._original_end_day,
            "budget_used": self._budget_used,
            "budget_total": self._template.budget if self._template else 0,
            "tasks": [self._task_dict(t) for t in self._tasks.values()],
            "disruptions": [self._disruption_dict(d) for d in self._disruptions],
            "history": self._history,
        }

    # =========================================================================
    # Action Application
    # =========================================================================

    def _apply_action(
        self, 
        action: Action, 
        reward_components: Dict[str, float], 
        info: Dict
    ) -> Tuple[bool, str]:
        
        at = action.action_type
        
        if at == ActionType.NOOP or at == "noop":
            return True, "Agent chose NO-OP - simulation advances."
            
        if at == ActionType.EXPEDITE_TASK or at == "expedite_task":
            return self._do_expedite(action, reward_components, info)
            
        if at == ActionType.DELAY_TASK or at == "delay_task":
            return self._do_delay(action, info)
            
        if at == ActionType.REASSIGN_RESOURCES or at == "reassign_resources":
            return self._do_reassign(action, info)
            
        return False, f"Unknown action type: {at}"

    def _do_expedite(self, action: Action, reward_components: Dict, info: Dict) -> Tuple[bool, str]:
        tid = action.task_id
        if not tid in self._tasks:
            return False, f"Task '{tid}' not found."
            
        task = self._tasks[tid]
        if task.status == "completed":
            return False, f"Task '{tid}' is already completed."
            
        extra = min(action.days or 1, 5)  # cap at 5 extra resource units
        task.extra_resources += extra
        
        # Extra cost is captured via daily_cost() accumulation - no upfront charge needed.
        # But we do record a step penalty to guide the agent.
        surcharge_per_day = extra * EXPEDITE_COST_PER_DAY_PER_RESOURCE
        self._expedite_extra_cost += surcharge_per_day * task.current_duration()
        
        # Bonus for resolving a disruption with expediting
        active = [d for d in self._disruptions if d.affected_task_id == tid and d.active and not d.resolved]
        for d in active:
            d.resolved = True
            reward_components["disruption_resolution_bonus"] += DISRUPTION_RESOLUTION_BONUS
            info["events"].append(f"Disruption [{d.id}] resolved via expedite on ({tid}).")
            
        return True, f"Expedited ({tid}) with +{extra} resource units (+${surcharge_per_day:.0f}/day)."

    def _do_delay(self, action: Action, info: Dict) -> Tuple[bool, str]:
        tid = action.task_id
        if not tid in self._tasks:
            return False, f"Task '{tid}' not found."
            
        task = self._tasks[tid]
        days = action.days or 1
        
        task.delay_days += days
        
        # Mark disruptions on this task as resolved (accepted)
        active = [d for d in self._disruptions if d.affected_task_id == tid and d.active and not d.resolved]
        for d in active:
            d.resolved = True
            info["events"].append(f"Accepted delay on ({tid}): disruption ({d.id}) absorbed.")
            
        return True, f"Delayed task ({tid}) by {days} days (accepted disruption)."

    def _do_reassign(self, action: Action, info: Dict) -> Tuple[bool, str]:
        src = action.task_id
        dst = action.target_task_id
        
        if not src or not dst:
            return False, "reassign_resources requires task_id and target_task_id."
            
        if src not in self._tasks or dst not in self._tasks:
            return False, f"Task '{src}' or '{dst}' not found."
            
        src_task = self._tasks[src]
        dst_task = self._tasks[dst]
        
        if src_task.resources <= 1:
            return False, f"Cannot remove resources from ({src}) - already at minimum."
            
        src_task.resources -= 1
        dst_task.resources += 1
        
        info["events"].append(f"Moved 1 resource from ({src}) to ({dst}).")
        return True, f"Reassigned 1 resource from ({src}) -> ({dst}) - ({dst}) will finish faster."

    # =========================================================================
    # Simulation advancement
    # =========================================================================

    def _advance_simulation(self, info: Dict) -> int:
        """
        Advance clock to the next meaningful event:
        - Task completion
        - Next disruption fire date
        Returns number of days advanced.
        """
        start_day = self._current_day
        
        # Find next event day
        next_event_day = self._current_day + 1
        
        # Check pending disruptions
        pending = [d for d in self._disruptions if not d.active and not d.resolved]
        if pending:
            nearest_fire = min(d.fire_on_day for d in pending)
            next_event_day = max(self._current_day + 1, nearest_fire)
            
        # Also look at task end days (don't skip past completions)
        in_progress_ends = []
        for task in self._tasks.values():
            if task.status in ["in_progress", "pending"] and task.end_day > self._current_day:
                in_progress_ends.append(task.end_day)
                
        if in_progress_ends:
            next_event_day = min(next_event_day, min(in_progress_ends))
            
        self._current_day = next_event_day
        
        # Mark tasks as in-progress / completed
        deps_done = all(t.status == "completed" for t in self._tasks.values())
        
        for task in self._tasks.values():
            if task.status == "completed":
                continue
                
            deps_done = all(
                self._tasks[dep].status == "completed" 
                for dep in task.dependencies
            )
            
            if deps_done and task.start_day <= self._current_day:
                if self._current_day >= task.end_day:
                    task.status = "completed"
                    task.progress_pct = 100.0
                    info["events"].append(f"Task ({task.id}) {task.name} completed on day {self._current_day}.")
                else:
                    task.status = "in_progress"
                    elapsed = self._current_day - task.start_day
                    task.progress_pct = min(99.0, 100 * elapsed / max(1, task.current_duration()))
            elif not deps_done:
                task.status = "blocked"
                
        # Fire pending disruptions
        for d in self._disruptions:
            if not d.active and not d.resolved and d.fire_on_day <= self._current_day:
                d.active = True
                affected = self._tasks.get(d.affected_task_id)
                if affected and affected.status != "completed":
                    affected.status = "disrupted"
                    affected.delay_days += d.delay_days
                    apply_schedule(self._tasks)
                    info["events"].append(
                        f"DISRUPTION [{d.id}]: {d.description} (+{d.delay_days}d delay on {d.affected_task_id})"
                    )
                    
        return self._current_day - start_day

    # =========================================================================
    # Helpers
    # =========================================================================

    def _all_tasks_complete(self) -> bool:
        return all(t.status == "completed" for t in self._tasks.values())

    def _compute_step_cost(self, days: int) -> float:
        cost = 0.0
        for task in self._tasks.values():
            if task.status in ["in_progress", "disrupted"]:
                cost += task.daily_cost() * days
        return cost

    def _build_observation(self) -> Observation:
        new_end_day = apply_schedule(self._tasks)
        current_delay = max(0, new_end_day - self._original_end_day)
        
        task_obs = []
        for task in self._tasks.values():
            task_obs.append(TaskObservation(
                id=task.id,
                name=task.name,
                original_duration=task.duration,
                current_duration=task.current_duration(),
                current_start_day=task.start_day,
                current_end_day=task.end_day,
                status=TaskStatus(task.status),
                dependencies=task.dependencies,
                is_on_critical_path=task.is_on_critical_path,
                delay_from_original=task.delay_days,
                resources=task.resources,
                cost_per_day=task.daily_cost(),
                progress_pct=task.progress_pct,
            ))
            
        disruption_obs = []
        for d in self._disruptions:
            if d.active or (d.fire_on_day <= self._current_day + 5): # show upcoming disruptions
                disruption_obs.append(DisruptionObservation(
                    id=d.id,
                    type=DisruptionType(d.type),
                    affected_task_id=d.affected_task_id,
                    remaining_delay_days=d.delay_days if not d.resolved else 0,
                    total_delay_days=d.delay_days,
                    description=d.description,
                    resolved=d.resolved,
                ))
                
        tasks_done = sum(1 for t in self._tasks.values() if t.status == "completed")
        disruptions_resolved = sum(1 for d in self._disruptions if d.resolved)
        disruptions_active = sum(1 for d in self._disruptions if d.active and not d.resolved)
        
        metrics = ProjectMetrics(
            original_end_day=self._original_end_day,
            current_projected_end_day=new_end_day,
            delay_days=current_delay,
            budget_total=self._template.budget if self._template else 0.0,
            budget_used=round(self._budget_used, 2),
            budget_remaining=round((self._template.budget if self._template else 0.0) - self._budget_used, 2),
            tasks_total=len(self._tasks),
            tasks_completed=tasks_done,
            disruptions_encountered=sum(1 for d in self._disruptions if d.active),
            disruptions_resolved=disruptions_resolved,
            on_critical_path_delayed=any(
                t.is_on_critical_path and t.delay_days > 0 
                for t in self._tasks.values()
            )
        )
        
        return Observation(
            current_day=self._current_day,
            episode_step=self._episode_step,
            task_level=self._task_level,
            tasks=task_obs,
            active_disruptions=[d for d in disruption_obs if not d.resolved],
            metrics=metrics,
            available_actions=[a.value for a in ActionType]
        )

    @staticmethod
    def _task_dict(task: TaskNode) -> Dict:
        return {
            "id": task.id,
            "name": task.name,
            "duration": task.duration,
            "current_duration": task.current_duration(),
            "delay_days": task.delay_days,
            "start_day": task.start_day,
            "end_day": task.end_day,
            "status": task.status,
            "extra_resources": task.extra_resources,
            "is_on_critical_path": task.is_on_critical_path,
        }
        
    @staticmethod
    def _disruption_dict(d: DisruptionEvent) -> Dict:
        return {
            "id": d.id,
            "type": d.type,
            "affected_task_id": d.affected_task_id,
            "delay_days": d.delay_days,
            "fire_on_day": d.fire_on_day,
            "active": d.active,
            "resolved": d.resolved,
        }