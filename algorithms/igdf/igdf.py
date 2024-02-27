# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import d4rl
import gym
import h5py
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import tqdm
from tqdm import trange
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR

from contrastiveoi import ContrastiveInfo

TensorBatch = List[torch.Tensor]


EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


@dataclass
class TrainConfig:
    # Wandb logging
    project: str = "DAR"
    group: str = "InfoMAX-IW-1.0"
    name: str = "IQL"
    # Experiment
    device: str = "cuda:0"
    env: str = "hopper-medium-expert-v2"  # OpenAI gym environment name
    env_num: int = 46  # body-xxx data(41) or joint-xxx data(46) or broken-xxx data(51) or others 56
    seed: int = 42  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(1e6)  # Max time steps to run environment
    checkpoints_path: Optional[str] = None  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    data_path: str = '/mnt/data/optimal/shijiyuan/kang/DA/datasets'
    # IQL
    buffer_size: int = 2_500_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    beta: float = 3.0  # Inverse temperature. Small beta -> BC, big beta -> maximizing Q
    iql_tau: float = 0.7  # Coefficient for asymmetric loss
    iql_deterministic: bool = False  # Use deterministic actor
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    vf_lr: float = 3e-4  # V function learning rate
    qf_lr: float = 3e-4  # Critic learning rate
    actor_lr: float = 3e-4  # Actor learning rate
    actor_dropout: Optional[float] = None  # Adroit uses dropout for policy network
    # Info
    repr_dim: int = 64
    ensemble_size: int = 1
    info_lr: float = 3e-4
    info_batch_size: int = 128
    num_update: int = 7000
    repr_norm: bool = False
    repr_norm_temp: bool = False
    ortho_init: bool = False
    output_gain: Optional[float] = None
    xi: float = 0.5
    alpha: float = 1.0

    def __post_init__(self):
        self.name = f"{self.name}-{str(uuid.uuid4())[:8]}-{self.seed}-{self.xi}-{self.alpha}"
        self.group = f"{self.group}-{self.env}-{self.env_num}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cuda:0",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        if self._pointer >= self._buffer_size:
            self._pointer = 0

        self._states[self._pointer] = self._to_tensor(state)
        self._actions[self._pointer] = self._to_tensor(action)
        self._rewards[self._pointer] = self._to_tensor(np.array([reward], dtype=np.float32))
        self._next_states[self._pointer] = self._to_tensor(next_state)
        self._dones[self._pointer] = self._to_tensor(np.array([done], dtype=np.float32))

        self._pointer += 1
        if self._size < self._buffer_size:
            self._size += 1

def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cuda:0"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()


class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cuda:0"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(self(state) * self.max_action, -self.max_action, self.max_action)
            .cpu()
            .data.numpy()
            .flatten()
        )


class TwinQ(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


class ImplicitQLearning:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        iql_tau: float = 0.7,
        beta: float = 3.0,
        max_steps: int = 1000000,
        discount: float = 0.99,
        tau: float = 0.005,
        device: str = "cuda:0",
    ):
        self.max_action = max_action
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau

        self.total_it = 0
        self.device = device

    def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
        # Update value function
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(
        self,
        next_v: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        log_dict: Dict,
        mask,
    ):
        qtarget = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        # q_loss = sum(F.mse_loss(q, qtarget) for q in qs) / len(qs)
        q_loss  = sum(
            (mask * (q - qtarget)**2).mean() for q in qs
        ) / len(qs)
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(
        self,
        adv: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict,
    ):
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.actor(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape missmatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

    def train(self, batch: TensorBatch, mask: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = batch

        log_dict = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)
        # Update value function
        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update Q function
        self._update_q(next_v, observations, actions, rewards, dones, log_dict, mask)
        # Update actor
        self._update_policy(adv, observations, actions, log_dict)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])

        self.total_it = state_dict["total_it"]

def merge_batch(source_batch, target_batch) -> TensorBatch:
    merged_states  = torch.concat((source_batch[0], target_batch[0]), axis = 0)
    merged_actions = torch.concat((source_batch[1], target_batch[1]), axis = 0)
    merged_rewards = torch.concat((source_batch[2], target_batch[2]), axis = 0)
    merged_next_states = torch.concat((source_batch[3], target_batch[3]), axis = 0)
    merged_dones = torch.concat((source_batch[4], target_batch[4]), axis = 0)

    return [merged_states, merged_actions, merged_rewards, merged_next_states, merged_dones]
            
def get_data(data_path):
    with h5py.File(data_path, 'r') as dataset:
        N = dataset['rewards'].shape[0]
        state_ = []
        action_ =[]
        reward_ = []
        next_state_ = []
        done_ = []

        state_ = np.array(dataset['observations'])
        action_ = np.array(dataset['actions'])
        reward_ = np.array(np.squeeze(dataset['rewards']))
        next_state_ = np.array(dataset['next_observations'])
        done_ = np.array(dataset['terminals'])

    return {
        'observations': state_,
        'actions': action_,
        'next_observations': next_state_,
        'rewards': reward_,
        'terminals': done_,
    }

def get_target_data1(data_path):
    with h5py.File(data_path, 'r') as dataset:
        N = int(1e5)
        state_ = []
        action_ =[]
        reward_ = []
        next_state_ = []
        done_ = []

        state_ = np.array(dataset['observations'])
        action_ = np.array(dataset['actions'])
        reward_ = np.array(np.squeeze(dataset['rewards']))
        next_state_ = np.array(dataset['next_observations'])
        done_ = np.array(dataset['terminals'])

    return {
        'observations': state_[:N],
        'actions': action_[:N],
        'next_observations': next_state_[:N],
        'rewards': reward_[:N],
        'terminals': done_[:N],
    }

def get_target_data2(data_path):
    with h5py.File(data_path, 'r') as dataset:
        N = int(1e5)
        state_ = []
        action_ =[]
        reward_ = []
        next_state_ = []
        done_ = []

        state_ = np.array(dataset['observations'])
        action_ = np.array(dataset['actions'])
        reward_ = np.array(np.squeeze(dataset['rewards']))
        next_state_ = np.array(dataset['next_observations'])
        done_ = np.array(dataset['terminals'])

    return {
        'observations': np.concatenate((state_[:N],state_[-N:])),
        'actions': np.concatenate((action_[:N],action_[-N:])),
        'next_observations': np.concatenate((next_state_[:N],next_state_[-N:])),
        'rewards': np.concatenate((reward_[:N],reward_[-N:])),
        'terminals': np.concatenate((done_[:N],done_[-N:])),
    }

def merge_data(source_data, target_data):
    merged_states  = np.concatenate((source_data['observations'], target_data['observations']), axis = 0)
    merged_actions = np.concatenate((source_data['actions'], target_data['actions']), axis = 0)
    merged_rewards = np.concatenate((source_data['next_observations'], target_data['next_observations']), axis = 0)
    merged_next_states = np.concatenate((source_data['rewards'], target_data['rewards']), axis = 0)
    source_data['terminals'] = np.squeeze(source_data['terminals'])
    merged_dones = np.concatenate((source_data['terminals'], target_data['terminals']), axis = 0)

    return {
        'observations': merged_states,
        'actions': merged_actions,
        'next_observations': merged_rewards,
        'rewards': merged_next_states,
        'terminals': merged_dones,
    }

@pyrallis.wrap()
def train(config: TrainConfig):
    wandb_init(asdict(config))
    env = gym.make(config.env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    split_env = config.env.split('-')
    if len(split_env) == 3:
        source_data_path = f"{config.data_path}/source_{split_env[0]}/{config.env_num}/{split_env[1]}.hdf5"
        target_data_path = f"{config.data_path}/target_{split_env[0]}/{split_env[0]}_{split_env[1]}_v2.hdf5"
        source_data = get_data(source_data_path)
        target_data = get_target_data1(target_data_path)
    if len(split_env) == 4 and split_env[2] == "expert":
        source_data_path = f"{config.data_path}/source_{split_env[0]}/{config.env_num}/{split_env[1]}_{split_env[2]}.hdf5"
        target_data_path = f"{config.data_path}/target_{split_env[0]}/{split_env[0]}_{split_env[1]}_{split_env[2]}_v2.hdf5"
        source_data = get_data(source_data_path)
        target_data = get_target_data2(target_data_path)
    if len(split_env) == 4 and split_env[2] == "replay":
        source_data_path = f"{config.data_path}/source_{split_env[0]}/{config.env_num}/{split_env[1]}_{split_env[2]}.hdf5"
        target_data_path = f"{config.data_path}/target_{split_env[0]}/{split_env[0]}_{split_env[1]}_{split_env[2]}_v2.hdf5"
        source_data = get_data(source_data_path)
        target_data = get_target_data1(target_data_path)
    dataset = merge_data(source_data, target_data)
    print("-----load over-----")

    if config.normalize_reward:
        modify_reward(dataset, config.env)

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    target_data["observations"] = normalize_states(
        target_data["observations"], state_mean, state_std
    )
    target_data["next_observations"] = normalize_states(
        target_data["next_observations"], state_mean, state_std
    )
    source_data["observations"] = normalize_states(
        source_data["observations"], state_mean, state_std
    )
    source_data["next_observations"] = normalize_states(
        source_data["next_observations"], state_mean, state_std
    )

    env = wrap_env(env, state_mean=state_mean, state_std=state_std)

    target_buffer = ReplayBuffer(
            state_dim,
            action_dim,
            config.buffer_size,
            config.device,
        )

    source_buffer = ReplayBuffer(
            state_dim,
            action_dim,
            config.buffer_size,
            config.device,
        )
    target_buffer.load_d4rl_dataset(target_data)
    source_buffer.load_d4rl_dataset(source_data)

    info = ContrastiveInfo(
        state_dim,
        action_dim,
        config.repr_dim,
        config.ensemble_size,
        config.repr_norm,
        config.repr_norm_temp,
        config.ortho_init,
        config.output_gain,
    ).to(config.device)

    info_optimizer = torch.optim.Adam(info.parameters(), lr=config.info_lr)

    total_steps = 0
    for train_step in trange(config.num_update, desc="Training"):
        # tar_s, tar_ss= [128, state_dim]
        # tar_a = [128, action_dim]
        # tar_sa = [128, action_dim + state_dim]
        tar_s, tar_a, _, tar_ss, _ = target_buffer.sample(config.info_batch_size) 
        _, _, _, src_ss, _ = source_buffer.sample(config.info_batch_size - 1) # src_ss = [127, state_dim]
        
        tar_s = tar_s.unsqueeze(1) # [128, 1, state_dim]
        tar_a = tar_a.unsqueeze(1) # [128, 1, action_dim]
        tar_ss = tar_ss.unsqueeze(1) # [128, 1, state_dim]
        src_ss = src_ss.unsqueeze(0) # [1, 127, state_dim]
        src_ss = src_ss.expand(config.info_batch_size, -1, -1) # [128, 127, state_dim]
        ss = torch.concat((tar_ss, src_ss), dim = 1) # [128, 128, state_dim]

        # print(info(tar_s, tar_a, ss, return_repr = True)[2].shape)

        logits = info(tar_s, tar_a, ss) # [128, 1, 128]
        logits = logits.squeeze(1)
        matrix = torch.zeros((config.info_batch_size, config.info_batch_size), dtype = torch.float32, device = config.device)
        matrix[:, 0] = 1
        # correct = (torch.argmax(logits, dim=1) == torch.argmax(matrix, dim=1))
        # binary_accuracy      = torch.mean(((logits > 0) == matrix).type(torch.float32)).item()
        # categorical_accuracy = torch.mean((correct).type(torch.float32)).item()
        info_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, matrix)
        info_loss = torch.mean(info_loss)
        info_optimizer.zero_grad()
        info_loss.backward()
        info_optimizer.step()

        total_steps += 1
        wandb.log(
            {"info_loss": info_loss,
            #  "binary_accuracy": binary_accuracy,
            #  "categorical_accuracy": categorical_accuracy,
            },
            step = total_steps
        )

    # env = wrap_env(env, state_mean=state_mean, state_std=state_std)

    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    q_network = TwinQ(state_dim, action_dim).to(config.device)
    v_network = ValueFunction(state_dim).to(config.device)
    actor = (
        DeterministicPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
        if config.iql_deterministic
        else GaussianPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
    ).to(config.device)
    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # IQL
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "max_steps": config.max_timesteps,
    }

    print("---------------------------------------")
    print(f"Training IQL, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = ImplicitQLearning(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    evaluations = []
    for t in range(int(config.max_timesteps)):
        target_batch = target_buffer.sample(config.batch_size // 2)
        src_s, src_a, src_r, src_ss, done = source_buffer.sample(config.batch_size // 2)
        if config.repr_norm:
            logits = info(src_s, src_a, src_ss)
            diagonal_elements = torch.diag(logits).reshape(-1, 1)
            src_info = diagonal_elements
        else:
            logits, srcsa_repr, srcss_repr = info(src_s, src_a, src_ss, return_repr = True)
            srcsa_repr = torch.linalg.norm(srcsa_repr, dim=-1, keepdim=True)  # [128, 1]
            srcss_repr = torch.linalg.norm(srcss_repr, dim=-1, keepdim=True)  # [128, 1]
            diagonal_elements = torch.diag(logits).reshape(-1, 1)
            src_info = diagonal_elements / (srcsa_repr * srcss_repr) # [128, 1]
        sorted_indices = torch.argsort(src_info[:, 0])
        if config.xi == 0.25:
            sorted_num = -32
        if config.xi == 0.5:
            sorted_num = -64
        if config.xi == 0.75:
            sorted_num = -96
        if config.xi == 1.0:
            sorted_num = -128
        top_half_indices = sorted_indices[sorted_num:]
        src_s = src_s[top_half_indices]
        src_a = src_a[top_half_indices]
        src_ss = src_ss[top_half_indices]
        src_r = src_r[top_half_indices]
        done = done[top_half_indices]
        info_temp = torch.exp(src_info[top_half_indices] * config.alpha)
        if config.xi == 0.25:
            mask = torch.ones((128+32, 1)).to(config.device)
            mask[:32] = info_temp
        if config.xi == 0.5:
            mask = torch.ones((128+64, 1)).to(config.device)
            mask[:64] = info_temp
        if config.xi == 0.75:
            mask = torch.ones((128+96, 1)).to(config.device)
            mask[:96] = info_temp
        if config.xi == 1.0:
            mask = torch.ones((128+128, 1)).to(config.device)
            mask[:128] = info_temp
        source_batch = [src_s, src_a, src_r, src_ss, done]
        batch = merge_batch(source_batch, target_batch)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch, mask)
        wandb.log(log_dict, step=trainer.total_it)
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_scores = eval_actor(
                env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
            )
            eval_score = eval_scores.mean()
            normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
            evaluations.append(normalized_eval_score)
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}"
            )
            print("---------------------------------------")
            if config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )
            wandb.log(
                {"d4rl_normalized_score": normalized_eval_score}, step=trainer.total_it
            )
    wandb.finish()


if __name__ == "__main__":
    train()
