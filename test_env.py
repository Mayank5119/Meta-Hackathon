"""
Local smoke test script for the Construction Superintendent OpenEnv.
Validates the core environment logic without requiring an LLM or API keys.
Uses basic heuristics to ensure all difficulty levels run successfully.
"""

import sys
import traceback

from env.construction_env import ConstructionEnv
from env.models import Action, ActionType
from graders.grader import grade

def heuristic_agent(obs) -> Action:
    """
    A simple hardcoded heuristic agent for testing.
    - If a critical path task is disrupted, expedite it.
    - Otherwise, NOOP.
    """
    # Find which tasks are currently on the critical path
    critical_task_ids = {t.id for t in obs.tasks if t.is_on_critical_path}
    
    # Check for active disruptions affecting those critical tasks
    for d in obs.active_disruptions:
        if d.affected_task_id in critical_task_ids:
            # Expedite the critical task to absorb the delay
            return Action(
                action_type=ActionType.EXPEDITE_TASK, 
                task_id=d.affected_task_id, 
                days=2
            )
            
    # Default action if no critical disruptions are active
    return Action(action_type=ActionType.NOOP)

def run_smoke_test():
    env = ConstructionEnv()
    levels = ["easy", "medium", "hard"]
    
    all_passed = True

    print("Starting local smoke tests...\n")

    for level in levels:
        try:
            obs = env.reset(task_level=level, seed=42)
            
            # Extract initial info for logging
            m = obs.metrics
            print(f"[{level}] reset OK - {m.tasks_total} tasks, original_end={m.original_end_day}d, budget=${m.budget_total:,.0f}")
            
            done = False
            total_reward = 0.0
            steps = 0
            
            while not done:
                action = heuristic_agent(obs)
                obs, reward, done, info = env.step(action)
                total_reward += reward
                steps += 1
                
            # Grade the completed episode
            final_state = env.state()
            result = grade(level, final_state)
            
            print(f"steps={steps} reward={total_reward:.2f} score={result.score:.4f} passed={result.passed}")
            
            if not result.passed:
                all_passed = False
                
        except Exception as e:
            print(f"[{level}] FAILED with exception:")
            traceback.print_exc()
            all_passed = False

    print("-" * 40)
    if all_passed:
        print("All levels OK.")
        sys.exit(0)
    else:
        print("Some tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    run_smoke_test()