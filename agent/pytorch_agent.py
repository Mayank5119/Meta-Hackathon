"""
PyTorch Double Dueling DQN Agent — Construction Superintendent RL Environment.

Observation encoding (128-dimensional flat vector):
  MAX_TASKS=10  × 10 features each  = 100
  MAX_DISRUPT=5 × 5  features each  =  25
  Project metrics                   =   3
                                    ─────
                                     128

Action space (61 discrete):
  0          → NOOP
  1–30       → EXPEDITE task_idx (0-9), days (1-3)  idx = 1  + task_idx*3 + (days-1)
  31–60      → DELAY    task_idx (0-9), days (1-3)  idx = 31 + task_idx*3 + (days-1)
"""
from __future__ import annotations

import os
import random
from collections import deque, namedtuple
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env.models import Action, ActionType, Observation

# ---------------------------------------------------------------------------
# Observation encoding constants
# ---------------------------------------------------------------------------

MAX_TASKS = 10
MAX_DISRUPTIONS = 5

# Per-task features:
#   status one-hot (5): pending/in_progress/completed/disrupted/blocked
#   is_on_critical_path (1)
#   progress_pct / 100      (1)
#   delay_from_original / 10 (1, capped at 1)
#   resources / 10          (1)
#   cost_per_day / 10 000   (1)
TASK_FEATURES = 10

# Per-disruption features:
#   type one-hot (4): weather/material_shortage/equipment_failure/labor_shortage
#   total_delay_days / 10  (1)
DISRUPTION_FEATURES = 5

# Project metrics:
#   delay_days / 20         (1)
#   budget_used / budget_total normalised (1)
#   tasks_completed / tasks_total         (1)
METRIC_FEATURES = 3

OBS_SIZE = MAX_TASKS * TASK_FEATURES + MAX_DISRUPTIONS * DISRUPTION_FEATURES + METRIC_FEATURES  # 128
ACTION_SIZE = 61

_STATUS_IDX = {
    "pending": 0, "in_progress": 1, "completed": 2, "disrupted": 3, "blocked": 4,
}
_DISRUPTION_TYPE_IDX = {
    "weather": 0, "material_shortage": 1, "equipment_failure": 2, "labor_shortage": 3,
}


# ---------------------------------------------------------------------------
# Observation encoder
# ---------------------------------------------------------------------------

def encode_observation(obs: Observation, task_ids: List[str]) -> torch.Tensor:
    """Encode a full Observation into a fixed-size float32 tensor of shape (OBS_SIZE,)."""
    vec = np.zeros(OBS_SIZE, dtype=np.float32)

    # ── Task features ────────────────────────────────────────────────────────
    task_map = {t.id: t for t in obs.tasks}
    for i, tid in enumerate(task_ids[:MAX_TASKS]):
        task = task_map.get(tid)
        if task is None:
            continue
        base = i * TASK_FEATURES
        status_str = task.status.value if hasattr(task.status, "value") else str(task.status)
        vec[base + _STATUS_IDX.get(status_str, 0)] = 1.0
        vec[base + 5] = 1.0 if task.is_on_critical_path else 0.0
        vec[base + 6] = task.progress_pct / 100.0
        vec[base + 7] = min(task.delay_from_original / 10.0, 1.0)
        vec[base + 8] = min(task.resources / 10.0, 1.0)
        vec[base + 9] = min(task.cost_per_day / 10_000.0, 1.0)

    # ── Disruption features ──────────────────────────────────────────────────
    offset = MAX_TASKS * TASK_FEATURES
    for i, d in enumerate(obs.active_disruptions[:MAX_DISRUPTIONS]):
        base = offset + i * DISRUPTION_FEATURES
        dtype_str = d.type.value if hasattr(d.type, "value") else str(d.type)
        vec[base + _DISRUPTION_TYPE_IDX.get(dtype_str, 0)] = 1.0
        vec[base + 4] = min(d.total_delay_days / 10.0, 1.0)

    # ── Project metrics ──────────────────────────────────────────────────────
    offset2 = offset + MAX_DISRUPTIONS * DISRUPTION_FEATURES
    m = obs.metrics
    vec[offset2]     = min(m.delay_days / 20.0, 1.0)
    vec[offset2 + 1] = min(m.budget_used / max(m.budget_total, 1), 2.0) / 2.0
    vec[offset2 + 2] = m.tasks_completed / max(m.tasks_total, 1)

    return torch.FloatTensor(vec)


# ---------------------------------------------------------------------------
# Action decoder
# ---------------------------------------------------------------------------

def decode_action(action_idx: int, task_ids: List[str]) -> Action:
    """Map a discrete action index to an Action object."""
    if action_idx == 0 or not task_ids:
        return Action(action_type=ActionType.NOOP)

    if 1 <= action_idx <= 30:
        task_idx = (action_idx - 1) // 3
        days = (action_idx - 1) % 3 + 1
        if task_idx < len(task_ids):
            return Action(action_type=ActionType.EXPEDITE_TASK, task_id=task_ids[task_idx], days=days)

    elif 31 <= action_idx <= 60:
        task_idx = (action_idx - 31) // 3
        days = (action_idx - 31) % 3 + 1
        if task_idx < len(task_ids):
            return Action(action_type=ActionType.DELAY_TASK, task_id=task_ids[task_idx], days=days)

    return Action(action_type=ActionType.NOOP)


# ---------------------------------------------------------------------------
# Network — Double Dueling DQN
# ---------------------------------------------------------------------------

class DQNNetwork(nn.Module):
    """Dueling DQN architecture: shared encoder → separate value & advantage streams."""

    def __init__(self, obs_size: int = OBS_SIZE, action_size: int = ACTION_SIZE):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, action_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared(x)
        value = self.value_stream(shared)
        advantage = self.advantage_stream(shared)
        # Q(s,a) = V(s) + A(s,a) - mean_a'[A(s,a')]
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    def __init__(self, capacity: int = 20_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """
    Double Dueling DQN with experience replay and epsilon-greedy exploration.

    Training loop (external):
        state = encode_observation(obs, task_ids)
        action_idx, action = agent.select_action(state, task_ids)
        next_obs, reward, done, _ = env.step(action)
        next_state = encode_observation(next_obs, task_ids)
        agent.store(state, action_idx, reward, next_state, done)
        loss = agent.update()
        agent.decay_epsilon()

    Inference:
        action_idx, action = agent.greedy_action(state, task_ids)
    """

    def __init__(
        self,
        obs_size: int = OBS_SIZE,
        action_size: int = ACTION_SIZE,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.99,
        buffer_size: int = 20_000,
        batch_size: int = 64,
        target_update_freq: int = 20,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_count = 0

        self.policy_net = DQNNetwork(obs_size, action_size).to(self.device)
        self.target_net = DQNNetwork(obs_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=lr, weight_decay=1e-5
        )
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=100, gamma=0.9
        )
        self.buffer = ReplayBuffer(buffer_size)

    # ── Action selection ─────────────────────────────────────────────────────

    def select_action(self, state: torch.Tensor, task_ids: List[str]) -> Tuple[int, Action]:
        """Epsilon-greedy selection used during training."""
        if random.random() < self.epsilon:
            action_idx = random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                q = self.policy_net(state.unsqueeze(0).to(self.device))
                action_idx = int(q.argmax().item())
        return action_idx, decode_action(action_idx, task_ids)

    def greedy_action(self, state: torch.Tensor, task_ids: List[str]) -> Tuple[int, Action]:
        """Pure greedy selection for inference (no exploration)."""
        self.policy_net.eval()
        with torch.no_grad():
            q = self.policy_net(state.unsqueeze(0).to(self.device))
            action_idx = int(q.argmax().item())
        self.policy_net.train()
        return action_idx, decode_action(action_idx, task_ids)

    # ── Learning ─────────────────────────────────────────────────────────────

    def store(self, state, action_idx, reward, next_state, done):
        self.buffer.push(state, action_idx, reward, next_state, float(done))

    def update(self) -> Optional[float]:
        """Sample a minibatch and perform a Double DQN gradient update."""
        if len(self.buffer) < self.batch_size:
            return None

        transitions = self.buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        states      = torch.stack(batch.state).to(self.device)
        actions     = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        rewards     = torch.FloatTensor(batch.reward).to(self.device)
        next_states = torch.stack(batch.next_state).to(self.device)
        dones       = torch.FloatTensor(batch.done).to(self.device)

        # Q(s, a) from policy net
        q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        # Target: action chosen by policy net, value from target net (Double DQN)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q       = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            targets      = rewards + self.gamma * next_q * (1.0 - dones)

        loss = nn.SmoothL1Loss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def step_lr_scheduler(self):
        self.lr_scheduler.step()

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(
            {
                "policy_net":   self.policy_net.state_dict(),
                "target_net":   self.target_net.state_dict(),
                "optimizer":    self.optimizer.state_dict(),
                "epsilon":      self.epsilon,
                "update_count": self.update_count,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt.get("epsilon", self.epsilon_end)
        self.update_count = ckpt.get("update_count", 0)
