"""
Deterministic agent graders for all three task levels.

Each grader:
- takes the environment final state dict
- applies a task-specific scoring rubric
- is reproducible and has clear pass/fail criteria
"""

from __future__ import annotations

from typing import Any, Dict

from env.models import GradeResult

# ---------------------------------------------------------------------------
# Easy grader
# ---------------------------------------------------------------------------

def grade_easy(state: Dict[str, Any]) -> GradeResult:
    """
    Easy task: 5-task linear project, 1 weather disruption.

    Scoring (out of 1.0):
    - 0.40 : Project completed (all tasks done)
    - 0.30 : Delay <= 2 days from original end date  (full credit)
             Delay <= 5 days                         (partial: 0.15)
    - 0.20 : Budget not exceeded
    - 0.10 : Disruption was actively resolved (expedite or reassign used)
    """
    breakdown: Dict[str, float] = {}
    tasks = state.get("tasks", [])
    total = len(tasks)
    done = sum([1 for t in tasks if t["status"] == "completed"])

    orig_end = state.get("original_end_day", 21)
    final_delay = max(0, state.get("current_projected_end_day", orig_end) - orig_end)

    # Re-check via disruptions
    disruptions = state.get("disruptions", [])
    resolved = sum([1 for d in disruptions if d["resolved"]])

    # Completion
    if done == total:
        breakdown["completion"] = 0.40
    else:
        breakdown["completion"] = 0.40 * (done / max(total, 1))

    # Delay
    if final_delay <= 2:
        breakdown["delay"] = 0.30
    elif final_delay <= 5:
        breakdown["delay"] = 0.15
    else:
        breakdown["delay"] = max(0.0, 0.30 - 0.05 * max(0, final_delay - 5))

    # Budget
    budget_total = state.get("budget_total", 90_000)
    budget_used = state.get("budget_used", 0)
    breakdown["budget"] = 0.20 if budget_used <= budget_total else 0.0

    # Disruption handling
    breakdown["disruption_handling"] = 0.10 if resolved == 1 else 0.0

    score = round(min(1.0, sum(breakdown.values())), 4)
    passed = score >= 0.60

    return GradeResult(
        task_level="easy",
        score=score,
        breakdown=breakdown,
        passed=passed,
        explanation=(
            f"**Project:** {done}/{total} tasks complete.\n"
            f"**Delay:** {final_delay}d vs original.\n"
            f"**Budget:** ${budget_used:,.0f} / ${budget_total:,.0f}.\n"
            f"**Disruptions resolved:** {resolved}.\n"
            f"**Score:** {score:.2f} ({'PASS' if passed else 'FAIL'})."
        )
    )

# ---------------------------------------------------------------------------
# Medium grader
# ---------------------------------------------------------------------------

def grade_medium(state: Dict[str, Any]) -> GradeResult:
    """
    Medium task: 8-task parallel project, 3 disruptions.

    Scoring:
    - 0.35 : Project completed
    - 0.25 : Delay <= 3d (full), <= 8d (partial 0.10)
    - 0.20 : >= 2 disruptions resolved
    - 0.20 : Budget within 110% of limit
    """
    breakdown: Dict[str, float] = {}
    tasks = state.get("tasks", [])
    total = len(tasks)
    done = sum([1 for t in tasks if t["status"] == "completed"])

    orig_end = state.get("original_end_day", 31)
    final_delay = max(0, state.get("current_projected_end_day", orig_end) - orig_end)

    disruptions = state.get("disruptions", [])
    total_disruptions = len(disruptions)
    resolved = sum([1 for d in disruptions if d["resolved"]])

    budget_total = state.get("budget_total", 250_000)
    budget_used = state.get("budget_used", 0)

    # Completion
    breakdown["completion"] = 0.35 * (done / max(total, 1))

    # Delay
    if final_delay <= 3:
        breakdown["delay"] = 0.25
    elif final_delay <= 8:
        breakdown["delay"] = 0.10
    else:
        breakdown["delay"] = max(0.0, 0.10 - 0.02 * max(0, final_delay - 8))

    # Disruption resolution
    if total_disruptions > 0:
        breakdown["disruption_handling"] = 0.20 * min(1.0, resolved / max(2, total_disruptions))
    else:
        breakdown["disruption_handling"] = 0.20

    # Budget
    if budget_used <= budget_total:
        breakdown["budget"] = 0.20
    elif budget_used <= budget_total * 1.10:
        breakdown["budget"] = 0.10
    else:
        breakdown["budget"] = 0.0

    score = round(min(1.0, sum(breakdown.values())), 4)
    passed = score >= 0.55

    return GradeResult(
        task_level="medium",
        score=score,
        breakdown=breakdown,
        passed=passed,
        explanation=(
            f"**Project:** {done}/{total} tasks complete.\n"
            f"**Delay:** {final_delay}d vs original.\n"
            f"**Budget:** ${budget_used:,.0f} / ${budget_total:,.0f}.\n"
            f"**Disruptions resolved:** {resolved}.\n"
            f"**Score:** {score:.2f} ({'PASS' if passed else 'FAIL'})."
        )
    )

# ---------------------------------------------------------------------------
# Hard grader
# ---------------------------------------------------------------------------

def grade_hard(state: Dict[str, Any]) -> GradeResult:
    """
    Hard task: 10-task complex DAG, 5 disruptions, tight budget.

    Scoring:
    - 0.30 : Project completed
    - 0.25 : Delay <= 5d (full), <= 12d (partial 0.10)
    - 0.25 : >= 3 disruptions resolved
    - 0.20 : Budget within 105%
    """
    breakdown: Dict[str, float] = {}
    tasks = state.get("tasks", [])
    total = len(tasks)
    done = sum([1 for t in tasks if t["status"] == "completed"])

    orig_end = state.get("original_end_day", 40)
    final_delay = max(0, state.get("current_projected_end_day", orig_end) - orig_end)

    disruptions = state.get("disruptions", [])
    total_disruptions = len(disruptions)
    resolved = sum([1 for d in disruptions if d["resolved"]])

    budget_total = state.get("budget_total", 500_000)
    budget_used = state.get("budget_used", 0)

    # Completion
    breakdown["completion"] = 0.30 * (done / max(total, 1))

    # Delay
    if final_delay <= 5:
        breakdown["delay"] = 0.25
    elif final_delay <= 12:
        breakdown["delay"] = 0.10
    else:
        breakdown["delay"] = max(0.0, 0.10 - 0.02 * max(0, final_delay - 12))

    # Disruption resolution (need >= 3 for full credit)
    if total_disruptions > 0:
        breakdown["disruption_handling"] = 0.25 * min(1.0, resolved / 3)
    else:
        breakdown["disruption_handling"] = 0.25

    # Budget
    if budget_used <= budget_total:
        breakdown["budget"] = 0.20
    elif budget_used <= budget_total * 1.05:
        breakdown["budget"] = 0.10
    else:
        breakdown["budget"] = 0.0

    score = round(min(1.0, sum(breakdown.values())), 4)
    passed = score >= 0.50

    return GradeResult(
        task_level="hard",
        score=score,
        breakdown=breakdown,
        passed=passed,
        explanation=(
            f"**Project:** {done}/{total} tasks complete.\n"
            f"**Delay:** {final_delay}d vs original.\n"
            f"**Budget:** ${budget_used:,.0f} / ${budget_total:,.0f}.\n"
            f"**Disruptions resolved:** {resolved}/{total_disruptions}.\n"
            f"**Score:** {score:.2f} ({'PASS' if passed else 'FAIL'})."
        )
    )

# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}

def grade(task_level: str, state: Dict[str, Any]) -> GradeResult:
    """Run the appropriate grader for the given task level."""
    if task_level not in GRADERS:
        raise ValueError(f"Unknown task_level {task_level}")
    return GRADERS[task_level](state)