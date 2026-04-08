"""
Typed Pydantic models for the Construction Superintendent OpenEnv environment.
Defines types: Action, Observation, Reward, StepResult.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# =============================================================================
# Enumerations
# =============================================================================

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DISRUPTED = "disrupted"
    BLOCKED = "blocked"

class DisruptionType(str, Enum):
    WEATHER = "weather"
    MATERIAL_SHORTAGE = "material_shortage"
    EQUIPMENT_FAILURE = "equipment_failure"
    LABOR_SHORTAGE = "labor_shortage"

class ActionType(str, Enum):
    EXPEDITE_TASK = "expedite_task"       # add resources = finish faster, costs money
    DELAY_TASK = "delay_task"             # accept delay, push task start forward
    REASSIGN_RESOURCES = "reassign_resources" # move workers from one task to another
    NOOP = "noop"                         # do nothing, advance to next event

# =============================================================================
# Action
# =============================================================================

class Action(BaseModel):
    """Agent action submitted to step()."""

    action_type: ActionType = Field(
        ..., description="Type of scheduling action to take."
    )
    task_id: Optional[str] = Field(
        None, description="Primary task to act upon."
    )
    target_task_id: Optional[str] = Field(
        None, description="Secondary task (used by reassign_resources)."
    )
    days: Optional[int] = Field(
        None, ge=1, le=30, description="Number of days (for delay_task or expedite_task)."
    )

    class Config:
        use_enum_values = True

# =============================================================================
# Observation sub-models
# =============================================================================

class TaskObservation(BaseModel):
    id: str
    name: str
    original_duration: int
    current_duration: int
    current_start_day: int
    current_end_day: int
    status: TaskStatus
    dependencies: List[str]
    is_on_critical_path: bool
    delay_from_original: int
    resources: int = Field(description="Number of resource units assigned.")
    cost_per_day: float
    progress_pct: float = Field(0.0, description="Completion percentage 0-100.")

class DisruptionObservation(BaseModel):
    id: str
    type: DisruptionType
    affected_task_id: str
    remaining_delay_days: int
    total_delay_days: int
    description: str
    resolved: bool = False

class ProjectMetrics(BaseModel):
    original_end_day: int
    current_projected_end_day: int
    delay_days: int
    budget_total: float
    budget_used: float
    budget_remaining: float
    tasks_total: int
    tasks_completed: int
    disruptions_encountered: int
    disruptions_resolved: int
    on_critical_path_delayed: bool

class Observation(BaseModel):
    """Full environment observation returned by reset() and step()."""

    current_day: int
    episode_step: int
    task_level: str = Field(description="easy | medium | hard")
    tasks: List[TaskObservation]
    active_disruptions: List[DisruptionObservation]
    metrics: ProjectMetrics
    available_actions: List[str] = Field(
        description="Human-readable list of valid action_type values."
    )
    terminal_message: Optional[str] = None

# =============================================================================
# Reward
# =============================================================================

class Reward(BaseModel):
    """Structured reward breakdown (returned in info)."""

    value: float = Field(description="Scalar reward for this step.")
    delay_penalty: float
    cost_penalty: float
    completion_bonus: float
    disruption_resolution_bonus: float
    explanation: str

# =============================================================================
# Step result (the canonical OpenEnv return)
# =============================================================================

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]

# =============================================================================
# Reset / State
# =============================================================================

class ResetRequest(BaseModel):
    task_level: str = Field("easy", description="easy | medium | hard")
    seed: Optional[int] = Field(None, description="RNG seed for reproducibility.")

class GradeResult(BaseModel):
    task_level: str
    score: float = Field(description="Normalised score 0.0-1.0")
    breakdown: Dict[str, float]
    passed: bool
    explanation: str