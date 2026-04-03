"""Quick smoke test for all three task levels."""
from env.construction_env import ConstructionEnv
from env.models import Action, ActionType
from graders.grader import grade

env = ConstructionEnv()

for level in ['easy', 'medium', 'hard']:
    obs = env.reset(task_level=level, seed=42)
    print(f"[{level}] reset OK — {obs.metrics.tasks_total} tasks, "
          f"original_end={obs.metrics.original_end_day}d, "
          f"budget=${obs.metrics.budget_total:,.0f}")

    done = False
    step = 0
    total_reward = 0.0

    while not done and step < 10:
        action = Action(action_type=ActionType.NOOP)
        if obs.active_disruptions:
            d = obs.active_disruptions[0]
            task = next((t for t in obs.tasks if t.id == d.affected_task_id), None)
            if task and task.is_on_critical_path:
                action = Action(action_type=ActionType.EXPEDITE_TASK,
                                task_id=d.affected_task_id, days=2)
            else:
                action = Action(action_type=ActionType.DELAY_TASK,
                                task_id=d.affected_task_id,
                                days=d.remaining_delay_days)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step += 1

    result = grade(level, env.state())
    print(f"  steps={step} reward={total_reward:.1f} score={result.score:.4f} passed={result.passed}")
    print(f"  {result.explanation}")
    print()

print("All levels OK.")
