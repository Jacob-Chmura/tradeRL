from collections import deque
from typing import Any, Deque, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from trade_rl.agents.base import TradingAgent
from trade_rl.env import TradingEnvironment
from trade_rl.util.args import ReinforceArgs

# TODO: Linear Schedule
# TODO: Consider GAE or critic for stability


class ReinforceAgent(TradingAgent):
    def __init__(self, env: TradingEnvironment, args: ReinforceArgs) -> None:
        super().__init__(env)

        obs_dim = int(env.observation_space.shape[0])  # type: ignore
        action_dim = int(env.action_space.n)  # type: ignore
        self.actions = env.action_space
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy = Policy(obs_dim, action_dim).to(self.device)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=args.lr)
        self.temp = args.temp
        self.batch_size = args.batch_size
        self.gamma = args.gamma

        self.rewards: List[float] = []
        self.log_probs: List[torch.Tensor] = []

    def get_action(self, obs: Any) -> int:
        obs_ = torch.Tensor(obs).to(self.device)
        action_logits = self.policy(obs_)
        probs = F.softmax(action_logits / self.temp, dim=-1)
        m = Categorical(probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        return int(action)

    def update(
        self, obs: Any, action: int, reward: float, terminated: bool, next_obs: Any
    ) -> None:
        self.rewards.append(reward)
        if len(self.rewards) % self.batch_size == 0:
            loss = self._get_loss()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            self._update_temperature()
            self.rewards, self.log_probs = [], []

    def _get_loss(self) -> torch.Tensor:
        R = 0.0
        policy_loss = []
        returns_: Deque[float] = deque()
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns_.appendleft(R)

        returns = torch.tensor(list(returns_))
        returns = (returns - returns.mean()) / (returns.std() + 1e-12)

        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)

        loss = torch.stack(policy_loss).sum()
        return loss

    def _update_temperature(self) -> None: ...


class Policy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(obs_dim, 16)
        self.layer2 = nn.Linear(16, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        action_logits = self.layer2(x)
        return action_logits
