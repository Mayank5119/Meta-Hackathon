Here is the complete code for `agent/train.py` extracted from the video:

```python
"""
Training script for the DQN Agent on the Construction Superintendent OpenEnv.
Includes state flattening and action mapping logic to interface the Neural Net
with the OpenEnv Pydantic models.
"""

import os
import numpy as np
import torch
from typing import List, Dict, Tuple

from env.construction_env import ConstructionEnv
from env.models import Action, ActionType, Observation
from agent.pytorch_agent import ConstructionDQNAgent

# =============================================================================
# Environment Wrappers / Converters
# =============================================================================

# For simplicity in this baseline, we hardcode max tasks to 10 (hard level)
MAX_TASKS = 10

def flatten_observation(obs: Observation) -> np.ndarray:
    """Convert complex Observation into a flat numpy array for the Neural Network."""
    features = []
    
    # 1. Global metrics (normalized roughly)
    features.append(obs.current_day / 50.0)
    features.append(obs.metrics.delay_days / 20.0)
    features.append(obs.metrics.budget_used / max(1.0, obs.metrics.budget_total))
    features.append(obs.metrics.disruptions_encountered / 5.0)
    features.append(obs.metrics.disruptions_resolved / 5.0)
    
    # 2. Task metrics (fixed size array up to MAX_TASKS)
    task_dict = {t.id: t for t in obs.tasks}
    for i in range(1, MAX_TASKS + 1):
        t_id = f"T{i}"
        if t_id in task_dict:
            t = task_dict[t_id]
            features.extend([
                t.progress_pct / 100.0,
                1.0 if t.status == "in_progress" else 0.0,
                1.0 if t.status == "disrupted" else 0.0,
                1.0 if t.is_on_critical_path else 0.0,
                t.delay_from_original / 10.0,
                t.resources / 10.0
            ])
        else:
            features.extend([0.0] * 6) # Pad missing tasks
            
    # 3. Active disruption flag
    features.append(1.0 if len(obs.active_disruptions) > 0 else 0.0)
    
    return np.array(features, dtype=np.float32)

def get_action_space_size() -> int:
    """
    Action space:
    0: NOOP
    1-10: Expedite T1..T10
    11-20: Delay T1..T10
    (Ignoring reassign for basic DQN to keep action space small: 21 actions)
    """
    return 1 + (MAX_TASKS * 2)

def map_int_to_action(action_idx: int) -> Action:
    """Convert NN integer output back to OpenEnv Action."""
    if action_idx == 0:
        return Action(action_type=ActionType.NOOP)
        
    elif 1 <= action_idx <= MAX_TASKS:
        task_idx = action_idx
        return Action(action_type=ActionType.EXPEDITE_TASK, task_id=f"T{task_idx}", days=2)
        
    elif 11 <= action_idx <= 20:
        task_idx = action_idx - 10
        return Action(action_type=ActionType.DELAY_TASK, task_id=f"T{task_idx}", days=2)
        
    # Fallback
    return Action(action_type=ActionType.NOOP)

def get_valid_actions_mask(obs: Observation) -> np.ndarray:
    """Return boolean mask of valid actions for the current state."""
    mask = np.zeros(get_action_space_size(), dtype=np.float32)
    mask[0] = 1.0 # NOOP always valid
    
    active_task_ids = {t.id for t in obs.tasks if t.status != "completed"}
    
    for i in range(1, MAX_TASKS + 1):
        t_id = f"T{i}"
        if t_id in active_task_ids:
            mask[i] = 1.0       # Can expedite
            mask[i + 10] = 1.0  # Can delay
            
    return mask

# =============================================================================
# Training Loop
# =============================================================================

def train():
    print("Starting DQN Training on Construction OpenEnv...")
    env = ConstructionEnv()
    
    # Calculate state dimension based on our flatten function
    dummy_obs = env.reset("hard")
    state_dim = len(flatten_observation(dummy_obs))
    action_dim = get_action_space_size()
    
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    
    agent = ConstructionDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=1e-4,
        batch_size=64,
        memory_capacity=20000,
        epsilon_decay=5000, # slower decay for longer training
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    num_episodes = 1000
    task_levels = ["easy", "medium", "hard"]
    
    os.makedirs("checkpoints", exist_ok=True)
    
    for episode in range(num_episodes):
        # Rotate through difficulty levels to train a robust agent
        level = task_levels[episode % len(task_levels)]
        obs = env.reset(task_level=level)
        
        state = flatten_observation(obs)
        total_reward = 0.0
        done = False
        
        while not done:
            mask = get_valid_actions_mask(obs)
            action_idx = agent.select_action(state, valid_actions_mask=mask)
            env_action = map_int_to_action(action_idx)
            
            next_obs, reward, done, info = env.step(env_action)
            next_state = flatten_observation(next_obs)
            
            # Store transition
            agent.memory.push(state, action_idx, reward, next_state, done)
            
            # Optimize
            agent.optimize_model()
            
            state = next_state
            total_reward += reward
            
        # Update target network
        if episode % agent.target_update == 0:
            agent.update_target_network()
            
        if (episode + 1) % 50 == 0:
            current_eps = agent.epsilon_end + (agent.epsilon_start - agent.epsilon_end) * np.exp(-1. * agent.steps_done / agent.epsilon_decay)
            print(f"Episode {episode+1}/{num_episodes} | Level: {level:<6} | Reward: {total_reward:.2f} | Epsilon: {current_eps:.3f}")
            
        if (episode + 1) % 250 == 0:
            ckpt_path = f"checkpoints/dqn_agent_ep{episode+1}.pth"
            agent.save(ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")
            
    print("Training complete!")

if __name__ == "__main__":
    train()
```