"""
Training loop for the DQN agent on ConstructionEnv.

CLI usage:
    python -m agent.train --task_level easy --episodes 300

The trained checkpoint is saved to agent/checkpoints/dqn_<level>.pt and
automatically picked up by the Gradio UI and run_episode_dqn().
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.construction_env import ConstructionEnv
from graders.grader import grade
from agent.pytorch_agent import DQNAgent, encode_observation

DEFAULT_CHECKPOINT = "agent/checkpoints/dqn_{level}.pt"


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_training(
    task_level: str = "easy",
    num_episodes: int = 300,
    seed: Optional[int] = 42,
    lr: float = 1e-3,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.99,
    batch_size: int = 64,
    buffer_size: int = 20_000,
    target_update_freq: int = 20,
    save_path: Optional[str] = None,
    progress_callback: Optional[Callable[[int, float, float, float], None]] = None,
    stop_flag: Optional[List] = None,
) -> Dict:
    """
    Train a DQN agent on the ConstructionEnv.

    Args:
        task_level:        "easy" | "medium" | "hard"
        num_episodes:      Total training episodes
        seed:              Base seed; episode i uses seed+i (None → random each episode)
        progress_callback: Called every episode with (episode, reward, epsilon, loss)
        stop_flag:         A single-element list [False]; set [0]=True to abort cleanly
        save_path:         Override checkpoint path (None → DEFAULT_CHECKPOINT)

    Returns:
        dict with keys: rewards, losses, grades, final_epsilon, total_updates, checkpoint
    """
    if save_path is None:
        save_path = DEFAULT_CHECKPOINT.format(level=task_level)
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    env = ConstructionEnv()
    agent = DQNAgent(
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size,
        buffer_size=buffer_size,
        target_update_freq=target_update_freq,
    )

    rewards_history: List[float] = []
    loss_history:    List[float] = []
    grade_history:   List[tuple] = []

    for ep in range(num_episodes):
        if stop_flag and stop_flag[0]:
            break

        ep_seed = (seed + ep) if seed is not None else None
        obs = env.reset(task_level=task_level, seed=ep_seed)
        task_ids = [t.id for t in obs.tasks]
        state = encode_observation(obs, task_ids)

        total_reward = 0.0
        ep_losses: List[float] = []
        done = False

        while not done:
            action_idx, action = agent.select_action(state, task_ids)
            next_obs, reward, done, _ = env.step(action)
            next_state = encode_observation(next_obs, task_ids)

            agent.store(state, action_idx, float(reward), next_state, done)
            loss = agent.update()
            if loss is not None:
                ep_losses.append(loss)

            state = next_state
            total_reward += float(reward)

        agent.decay_epsilon()
        if (ep + 1) % 100 == 0:
            agent.step_lr_scheduler()

        avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        rewards_history.append(total_reward)
        loss_history.append(avg_loss)

        # Grade periodically (every 50 episodes and at the end)
        if (ep + 1) % 50 == 0 or ep == num_episodes - 1:
            final_state = env.state()
            grade_result = grade(task_level, final_state)
            grade_history.append((ep + 1, grade_result.score, grade_result.passed))

        if progress_callback:
            progress_callback(ep + 1, total_reward, agent.epsilon, avg_loss)

    agent.save(save_path)

    return {
        "rewards":       rewards_history,
        "losses":        loss_history,
        "grades":        grade_history,
        "final_epsilon": agent.epsilon,
        "total_updates": agent.update_count,
        "checkpoint":    save_path,
    }


# ---------------------------------------------------------------------------
# Greedy inference with a trained checkpoint
# ---------------------------------------------------------------------------

def run_episode_dqn(
    task_level: str = "easy",
    seed: int = 42,
    checkpoint_path: Optional[str] = None,
) -> Dict:
    """
    Run a single episode using a pre-trained DQN (greedy policy, epsilon=0).

    Returns:
        dict with steps, total_reward, final_obs, grade
    """
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT.format(level=task_level)

    agent = DQNAgent(epsilon_start=0.0, epsilon_end=0.0)
    if os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
    else:
        print(f"[WARNING] Checkpoint not found: {checkpoint_path}. Running untrained agent.")

    env = ConstructionEnv()
    obs = env.reset(task_level=task_level, seed=seed)
    task_ids = [t.id for t in obs.tasks]

    steps_log: List[Dict] = []
    total_reward = 0.0
    done = False

    while not done:
        state = encode_observation(obs, task_ids)
        _, action = agent.greedy_action(state, task_ids)
        next_obs, reward, done, info = env.step(action)

        steps_log.append({
            "step":   obs.episode_step,
            "day":    obs.current_day,
            "action": (
                f"{action.action_type} "
                f"{action.task_id or ''} "
                f"{'days='+str(action.days) if action.days else ''}"
            ).strip(),
            "reward": round(float(reward), 2),
            "events": " | ".join(info.get("events", [])),
        })
        total_reward += float(reward)
        obs = next_obs

    final_state = env.state()
    grade_result = grade(task_level, final_state)

    return {
        "steps":        steps_log,
        "total_reward": round(total_reward, 2),
        "final_obs":    obs,
        "grade":        grade_result,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN agent on ConstructionEnv.")
    parser.add_argument("--task_level", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epsilon_decay", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    def print_progress(ep, reward, epsilon, loss):
        if ep % 50 == 0:
            print(f"  Ep {ep:4d} | reward={reward:8.2f} | ε={epsilon:.3f} | loss={loss:.4f}")

    print(f"Training DQN on '{args.task_level}' for {args.episodes} episodes...")
    results = run_training(
        task_level=args.task_level,
        num_episodes=args.episodes,
        seed=args.seed,
        lr=args.lr,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        progress_callback=print_progress,
    )
    print(f"\nDone. Checkpoint: {results['checkpoint']}")
    if results["grades"]:
        last = results["grades"][-1]
        print(f"Final grade (ep {last[0]}): {last[1]:.4f} ({'PASS' if last[2] else 'FAIL'})")
