"""
Critical Path Method (CPM) scheduler.

Calculates start_day, end_day, and is_on_critical_path for a DAG of TaskNodes.
Handles forward pass (early start/finish) and backward pass (late start/finish).
"""

from typing import Dict, List, Set

from env.project import TaskNode

def apply_schedule(tasks: Dict[str, TaskNode]) -> int:
    """
    Apply CPM to the task dictionary in-place.
    Updates start_day, end_day, and is_on_critical_path for all tasks.
    Returns the final projected end day of the project.
    """
    if not tasks:
        return 0

    # 1. Forward Pass (Calculate Early Start and Early Finish)
    # Reset computed fields
    for task in tasks.values():
        task.start_day = 0
        task.end_day = 0
        task.is_on_critical_path = False

    resolved: Set[str] = set()
    pending = set(tasks.keys())
    
    # Iteratively resolve tasks where all dependencies are already resolved
    while pending:
        progress_made = False
        for tid in list(pending):
            task = tasks[tid]
            
            if all(dep in resolved for dep in task.dependencies):
                # Calculate start_day (max of all dependency end_days)
                if not task.dependencies:
                    task.start_day = 0
                else:
                    task.start_day = max(tasks[dep].end_day for dep in task.dependencies)
                    
                task.end_day = task.start_day + task.current_duration()
                resolved.add(tid)
                pending.remove(tid)
                progress_made = True
                
        if not progress_made and pending:
            # Circular dependency detected
            raise ValueError(f"Circular dependency detected in tasks: {pending}")

    # The project end day is the maximum end_day of all tasks
    project_end_day = max((task.end_day for task in tasks.values()), default=0)

    # 2. Backward Pass (Calculate Late Start, Late Finish, and Critical Path)
    # First, map which tasks act as dependencies for others
    is_dependency_for: Dict[str, List[str]] = {tid: [] for tid in tasks}
    for tid, task in tasks.items():
        for dep in task.dependencies:
            if dep in is_dependency_for:
                is_dependency_for[dep].append(tid)

    late_finish: Dict[str, int] = {}
    late_start: Dict[str, int] = {}
    
    resolved_backward: Set[str] = set()
    pending_backward = set(tasks.keys())
    
    while pending_backward:
        progress_made = False
        for tid in list(pending_backward):
            dependent_tasks = is_dependency_for[tid]
            
            # Can process if all tasks that depend on THIS task are already processed
            if all(dep in resolved_backward for dep in dependent_tasks):
                if not dependent_tasks:
                    # Terminal node
                    lf = project_end_day
                else:
                    # Minimum of the late starts of all dependent tasks
                    lf = min(late_start[dep] for dep in dependent_tasks)
                    
                late_finish[tid] = lf
                late_start[tid] = lf - tasks[tid].current_duration()
                
                # Check critical path condition: Early Start == Late Start (Float == 0)
                if tasks[tid].start_day == late_start[tid]:
                    tasks[tid].is_on_critical_path = True
                    
                resolved_backward.add(tid)
                pending_backward.remove(tid)
                progress_made = True
                
        if not progress_made and pending_backward:
             break
             
    return project_end_day