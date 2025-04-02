import random
from collections import deque, namedtuple
from typing import Any, Deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from trade_rl.agents.base import TradingAgent
from trade_rl.env import TradingEnvironment

# TODO: Move to config
LR = 2.5e-4
BUFFER_SIZE = 10_000
GAMMA = 0.99
BATCH_SIZE = 128

# TODO: Linear schedule
EPS = 0.5

# TODO: Consider adding target net for stability


class DQNAgent(TradingAgent):
    def __init__(self, env: TradingEnvironment) -> None:
        super().__init__(env)

        obs_dim = int(env.observation_space.shape[0])  # type: ignore
        action_dim = int(env.action_space.n)  # type: ignore
        self.actions = env.action_space
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.qnet = DQN(obs_dim=obs_dim, action_dim=action_dim).to(self.device)
        self.opt = torch.optim.Adam(self.qnet.parameters(), lr=LR)
        self.memory = ReplayBuffer(BUFFER_SIZE)

    def get_action(self, obs: Any) -> int:
        if random.random() < EPS:
            return self.actions.sample()
        obs_ = torch.Tensor(obs).to(self.device)
        with torch.no_grad():
            return self.qnet(obs_).argmax().cpu().item()

    def update(
        self, obs: Any, action: int, reward: float, terminated: bool, next_obs: Any
    ) -> None:
        _tensorify = lambda x, dtype: torch.tensor(x, dtype=dtype, device=self.device)
        obs_ = _tensorify(obs[None, :], torch.float32)
        next_obs_ = _tensorify(next_obs[None, :], torch.float32)
        action_ = _tensorify([action], torch.int64)
        reward_ = _tensorify([reward], torch.float32)
        terminated_ = _tensorify([terminated], torch.float32)
        self.memory.push(obs_, action_, next_obs_, reward_, terminated_)

        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        obs_ = torch.cat(batch.obs)
        next_obs_ = torch.cat(batch.next_obs)
        action_ = torch.cat(batch.action)
        reward_ = torch.cat(batch.reward)
        done_ = torch.cat(batch.done)

        q = self.qnet(obs_).gather(1, action_.unsqueeze(1)).squeeze()
        with torch.no_grad():
            target_q = reward_ + (1 - done_) * GAMMA * self.qnet(next_obs_).max(1)[0]
        loss = F.mse_loss(q, target_q)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


class DQN(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


Transition = namedtuple('Transition', ('obs', 'action', 'next_obs', 'reward', 'done'))


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.memory: Deque[Transition] = deque([], maxlen=capacity)

    def push(self, *args: Any) -> None:
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> Any:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)
