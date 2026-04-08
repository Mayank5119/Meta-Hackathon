"""
PyTorch DQN Agent for the Construction Superintendent OpenEnv.
Handles observation flattening, Q-network definition, and action selection.
"""

import math
import random
from collections import deque
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from env.models import Action, ActionType, Observation

# =============================================================================
# Neural Network architecture
# =============================================================================

class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# =============================================================================
# Replay Memory
# =============================================================================

class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# =============================================================================
# DQN Agent
# =============================================================================

class ConstructionDQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 1000,
        memory_capacity: int = 10000,
        batch_size: int = 64,
        target_update: int = 10,
        device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = torch.device(device)
        self.steps_done = 0

        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(memory_capacity)

    def select_action(self, state: np.ndarray, valid_actions_mask: np.ndarray = None) -> int:
        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no.grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                
                if valid_actions_mask is not None:
                    mask_tensor = torch.FloatTensor(valid_actions_mask).unsqueeze(0).to(self.device)
                    # Set invalid actions to a very low negative number
                    q_values = q_values + (mask_tensor - 1.0) * 1e9
                    
                return q_values.max(1)[1].item()
        else:
            if valid_actions_mask is not None:
                valid_indices = np.where(valid_actions_mask == 1)[0]
                return np.random.choice(valid_indices)
            return random.randrange(self.action_dim)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

        state_batch = torch.FloatTensor(np.array(batch_state)).to(self.device)
        action_batch = torch.LongTensor(batch_action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch_reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch_next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch_done).to(self.device)

        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states
        with torch.no.grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0]
            
        # Compute the expected Q values
        expected_state_action_values = reward_batch + (self.gamma * next_state_values * (1 - done_batch))

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def save(self, filepath: str):
        torch.save(self.policy_net.state_dict(), filepath)
        
    def load(self, filepath: str):
        self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())