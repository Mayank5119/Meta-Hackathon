"""
Construction project templates (task DAGs) for the three difficulty levels.
Each template defines the initial task graph, budget, and base schedule.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class TaskNode:
    id: str
    name: str
    duration: int               # planned duration in days
    dependencies: List[str]
    resources: int = 2          # default worker-units assigned
    cost_per_day: float = 1000.0
    # runtime state (populated by scheduler)
    status: str = "pending"
    start_day: int = 0
    end_day: int = 0
    progress_pct: float = 0.0
    extra_resources: int = 0    # added via expedite action
    delay_days: int = 0         # cumulative delay applied
    is_on_critical_path: bool = False

    def current_duration(self) -> int:
        """Duration after accounting for expediting."""
        base = self.duration + self.delay_days
        if self.extra_resources > 0:
            reduction = min(self.extra_resources * 1, base // 2)
            return max(1, base - reduction)
        return max(1, base)

    def daily_cost(self) -> float:
        # cost_per_day is the TOTAL daily rate for the task at baseline crew size.
        # Extra resources (expediting) add a flat $400/day surcharge per unit.
        extra_surcharge = self.extra_resources * 400.0
        return self.cost_per_day + extra_surcharge

    def clone(self) -> "TaskNode":
        return copy.deepcopy(self)


@dataclass
class ProjectTemplate:
    name: str
    description: str
    tasks: List[TaskNode]
    budget: float
    max_steps: int = 20         # episode cap

    def task_map(self) -> Dict[str, TaskNode]:
        return {t.id: t for t in self.tasks}

    def clone(self) -> "ProjectTemplate":
        return copy.deepcopy(self)


# ---------------------------------------------------------------------------
# Task-level definitions
# ---------------------------------------------------------------------------

def easy_project() -> ProjectTemplate:
    """
    5 tasks, linear critical path, 1 disruption event.
    Optimal agent reduces project delay to ≤ 2 days.
    """
    return ProjectTemplate(
        name="Simple Residential Build",
        description=(
            "A small single-family home with a linear construction sequence. "
            "One weather disruption will hit early framing."
        ),
        budget=90_000,
        max_steps=15,
        tasks=[
            TaskNode("T1", "Site Preparation",   3, [],        resources=3, cost_per_day=800),
            TaskNode("T2", "Foundation Pouring", 5, ["T1"],    resources=4, cost_per_day=1_500),
            TaskNode("T3", "Structural Framing", 7, ["T2"],    resources=5, cost_per_day=2_000),
            TaskNode("T4", "Roofing",            4, ["T3"],    resources=3, cost_per_day=1_800),
            TaskNode("T5", "Final Inspection",   2, ["T4"],    resources=1, cost_per_day=500),
        ],
    )


def medium_project() -> ProjectTemplate:
    """
    8 tasks, parallel MEP tracks, 3 disruption events.
    Agent must handle cascading delays across parallel paths.
    """
    return ProjectTemplate(
        name="Commercial Office Build",
        description=(
            "A two-story commercial office with parallel MEP installation. "
            "Three disruptions: rain, material shortage, equipment failure."
        ),
        budget=250_000,
        max_steps=25,
        tasks=[
            TaskNode("T1", "Site Preparation",        3, [],                      resources=3, cost_per_day=800),
            TaskNode("T2", "Foundation",              5, ["T1"],                  resources=4, cost_per_day=1_500),
            TaskNode("T3", "Structural Framing",      7, ["T2"],                  resources=6, cost_per_day=2_500),
            TaskNode("T4", "Roofing",                 4, ["T3"],                  resources=4, cost_per_day=1_800),
            TaskNode("T5", "Electrical Rough-In",     5, ["T3"],                  resources=3, cost_per_day=1_200),
            TaskNode("T6", "Plumbing Rough-In",       4, ["T3"],                  resources=3, cost_per_day=1_100),
            TaskNode("T7", "Insulation & Drywall",    5, ["T4", "T5", "T6"],      resources=5, cost_per_day=1_600),
            TaskNode("T8", "Final Inspection",        2, ["T7"],                  resources=1, cost_per_day=500),
        ],
    )


def hard_project() -> ProjectTemplate:
    """
    10 tasks, complex DAG with joins & forks, 5 disruptions, tight budget.
    Agent must balance time vs cost trade-offs.
    """
    return ProjectTemplate(
        name="Multi-Story Mixed-Use Complex",
        description=(
            "A five-story mixed-use building with complex dependency graph. "
            "Five disruptions across critical and non-critical paths. "
            "Tight budget — over-expediting causes cost overrun penalty."
        ),
        budget=500_000,
        max_steps=35,
        tasks=[
            TaskNode("T1",  "Site Survey & Preparation",    4,  [],                         resources=4, cost_per_day=900),
            TaskNode("T2",  "Foundation & Basement",        8,  ["T1"],                     resources=6, cost_per_day=2_500),
            TaskNode("T3",  "Structural Framing",           10, ["T2"],                     resources=8, cost_per_day=3_000),
            TaskNode("T4",  "Roofing & Waterproofing",      5,  ["T3"],                     resources=4, cost_per_day=2_200),
            TaskNode("T5",  "Electrical Systems",           7,  ["T3"],                     resources=4, cost_per_day=1_800),
            TaskNode("T6",  "Plumbing & HVAC",              6,  ["T3"],                     resources=4, cost_per_day=2_000),
            TaskNode("T7",  "Interior Walls & Insulation",  5,  ["T4", "T5", "T6"],         resources=5, cost_per_day=1_500),
            TaskNode("T8",  "Finishing & Fixtures",         8,  ["T7"],                     resources=6, cost_per_day=2_000),
            TaskNode("T9",  "Landscaping & Exterior",       4,  ["T4"],                     resources=3, cost_per_day=800),
            TaskNode("T10", "Final Inspection & Handover",  3,  ["T8", "T9"],               resources=2, cost_per_day=600),
        ],
    )


PROJECT_FACTORIES = {
    "easy":   easy_project,
    "medium": medium_project,
    "hard":   hard_project,
}
