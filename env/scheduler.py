"""
Critical Path Method (CPM) scheduler for the construction environment.
Computes earliest-start times, latest-start times, and the critical path.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from env.project import TaskNode


def topological_sort(tasks: Dict[str, TaskNode]) -> List[str]:
    """Kahn's algorithm — returns tasks in dependency order."""
    in_degree: Dict[str, int] = {tid: 0 for tid in tasks}
    for task in tasks.values():
        for dep in task.dependencies:
            in_degree[task.id] += 1

    queue = [tid for tid, deg in in_degree.items() if deg == 0]
    result: List[str] = []

    while queue:
        queue.sort()          # deterministic ordering
        current = queue.pop(0)
        result.append(current)
        for task in tasks.values():
            if current in task.dependencies:
                in_degree[task.id] -= 1
                if in_degree[task.id] == 0:
                    queue.append(task.id)

    if len(result) != len(tasks):
        raise ValueError("Cycle detected in task dependency graph.")

    return result


def compute_early_schedule(tasks: Dict[str, TaskNode]) -> Dict[str, Tuple[int, int]]:
    """
    Forward pass: compute Earliest Start (ES) and Earliest Finish (EF) for each task.
    Returns {task_id: (early_start, early_finish)}.
    """
    order = topological_sort(tasks)
    es: Dict[str, int] = {}
    ef: Dict[str, int] = {}

    for tid in order:
        task = tasks[tid]
        if not task.dependencies:
            es[tid] = 0
        else:
            es[tid] = max(ef[dep] for dep in task.dependencies)
        ef[tid] = es[tid] + task.current_duration()

    return {tid: (es[tid], ef[tid]) for tid in tasks}


def compute_late_schedule(
    tasks: Dict[str, TaskNode],
    project_duration: int,
    early: Dict[str, Tuple[int, int]],
) -> Dict[str, Tuple[int, int]]:
    """
    Backward pass: compute Latest Start (LS) and Latest Finish (LF).
    Returns {task_id: (late_start, late_finish)}.
    """
    order = topological_sort(tasks)
    lf: Dict[str, int] = {}
    ls: Dict[str, int] = {}

    for tid in reversed(order):
        task = tasks[tid]
        successors = [
            t for t in tasks.values() if tid in t.dependencies
        ]
        if not successors:
            lf[tid] = project_duration
        else:
            lf[tid] = min(ls[s.id] for s in successors)
        ls[tid] = lf[tid] - task.current_duration()

    return {tid: (ls[tid], lf[tid]) for tid in tasks}


def find_critical_path(tasks: Dict[str, TaskNode]) -> Set[str]:
    """
    Returns the set of task IDs on the critical path (float == 0).
    """
    early = compute_early_schedule(tasks)
    project_duration = max(ef for _, ef in early.values())
    late = compute_late_schedule(tasks, project_duration, early)

    critical: Set[str] = set()
    for tid in tasks:
        total_float = late[tid][0] - early[tid][0]
        if total_float == 0:
            critical.add(tid)

    return critical


def apply_schedule(tasks: Dict[str, TaskNode]) -> int:
    """
    Apply computed earliest-start schedule to all tasks.
    Updates task.start_day, task.end_day, task.is_on_critical_path.
    Returns projected project end day.
    """
    early = compute_early_schedule(tasks)
    critical = find_critical_path(tasks)

    for tid, task in tasks.items():
        task.start_day = early[tid][0]
        task.end_day = early[tid][1]
        task.is_on_critical_path = tid in critical

    return max(ef for _, ef in early.values())
