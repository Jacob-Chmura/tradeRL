import pathlib
import random
from collections import deque, namedtuple
from typing import Any, Deque, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from trade_rl.agents.base import TradingAgent
from trade_rl.env import TradingEnvironment
from trade_rl.util.args import DQNArgs

# TODO: Consider adding target net for stability


class DQNAgent(TradingAgent):
    def __init__(self, env: TradingEnvironment, args: DQNArgs) -> None:
        super().__init__(env)

        obs_dim = int(env.observation_space.shape[0])  # type: ignore
        action_dim = int(env.action_space.n)  # type: ignore
        self.actions = env.action_space
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.qnet = DQN(obs_dim=obs_dim, action_dim=action_dim).to(self.device)
        self.opt = torch.optim.Adam(self.qnet.parameters(), lr=args.lr)
        self.memory = ReplayBuffer(args.buffer_size)

        self.eps = self.eps_start = args.eps_start
        self.eps_end = args.eps_end
        self.batch_size = args.batch_size
        self.gamma = args.gamma

    def get_action(self, obs: Any) -> int:
        if random.random() < self.eps:
            return self.actions.sample()
        obs_ = torch.Tensor(obs).to(self.device)
        with torch.no_grad():
            return self.qnet(obs_).argmax().cpu().item()

    def update(
        self,
        obs: Any,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: Any,
        info: Dict[str, Any],
    ) -> None:
        self.eps = self.linear_schedule(
            self.eps_start, self.eps_end, info['global_step']
        )

        _tensorify = lambda x, dtype: torch.tensor(x, dtype=dtype, device=self.device)
        obs_ = _tensorify(obs[None, :], torch.float32)
        next_obs_ = _tensorify(next_obs[None, :], torch.float32)
        action_ = _tensorify([action], torch.int64)
        reward_ = _tensorify([reward], torch.float32)
        terminated_ = _tensorify([terminated], torch.float32)
        self.memory.push(obs_, action_, next_obs_, reward_, terminated_)

        if len(self.memory) >= self.batch_size:
            loss = self._get_loss()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

    def save_model(self, path: str | pathlib.Path) -> None:
        path = pathlib.Path(path) / f'{self.__class__.__name__}_dqn.pt'
        self.logger.info(f'Saving model to: {path}')
        torch.save(self.qnet.state_dict(), path)
        self.logger.info(f'Saved model to: {path}')

    def load_model(self, path: str | pathlib.Path) -> None:
        path = pathlib.Path(path) / f'{self.__class__.__name__}_dqn.pt'
        self.logger.info(f'Loading model from: {path}')
        self.qnet.load_state_dict(torch.load(path, weights_only=True, map_location=torch.device('cpu')))
        self.qnet.eval()
        self.logger.info(f'Loaded model from: {path}')

        self.eps = 0
        self.logger.info(f'Agent Exploration eps = {self.eps}')

    def _get_loss(self) -> torch.Tensor:
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        obs_ = torch.cat(batch.obs)
        next_obs_ = torch.cat(batch.next_obs)
        action_ = torch.cat(batch.action)
        reward_ = torch.cat(batch.reward)
        done_ = torch.cat(batch.done)

        q = self.qnet(obs_).gather(1, action_.unsqueeze(1)).squeeze()
        with torch.no_grad():
            target = reward_ + (1 - done_) * self.gamma * self.qnet(next_obs_).max(1)[0]
        loss = F.mse_loss(q, target)
        return loss


class DQN(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
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
