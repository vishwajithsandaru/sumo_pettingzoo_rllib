import sumo_rl
from torchrl.envs.libs.pettingzoo import PettingZooWrapper

import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

from torch import multiprocessing

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import check_env_specs

from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal

from torchrl.objectives import ClipPPOLoss, ValueEstimators

torch.manual_seed(0)
from matplotlib import pyplot as plt
from tqdm import tqdm

import ppo


is_fork = multiprocessing.get_start_method(allow_none=True) == 'fork'
device = (
    torch.device(0) 
    if torch.cuda.is_available() and not is_fork
    else torch.device('cpu')
)
vmas_device = device

frames_per_batch = 1000
n_iters = 100
total_frames = frames_per_batch * n_iters

num_epochs = 30
minibatch_size = 100
lr = 1e-3
max_grad_norm = 1.0

clip_episolon = 0.2
gamma = 0.9
lmbda = 0.9
entropy_eps = 1e-4

#Environment

max_steps = 100

num_vmas_envs = (frames_per_batch // max_steps)

env = PettingZooWrapper(sumo_rl.parallel_env(
    net_file='net/grid4x4/grid4x4.net.xml',
    route_file='net/grid4x4/grid4x4_1.rou.xml',
    use_gui=False,
    num_seconds=40
))

print(env.num_agents)

# ppo.train_ppo(env)