import tensordict
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sumo_rl
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from tensordict import TensorDict

from policy_network import PolicyNetwork
from ppo3 import PPO

env = PettingZooWrapper(sumo_rl.parallel_env(
    net_file='net/grid4x4/grid4x4.net.xml',
    route_file='net/grid4x4/grid4x4_1.rou.xml',
    use_gui=False,
    num_seconds=2000,
))

num_agents = env.num_agents

input_size = env.observation_spaces['A0'].shape[0]


output_size = env.action_spaces['A0'].n

print(env.action_spec)

policy_network = PolicyNetwork(input_size=input_size, output_size=4)
optimizer = optim.Adam(policy_network.parameters(), lr=0.001)

# ppo_agents = [PPO(policy_network, optimizer) for _ in range(num_agents)]
ppo_agents = {key: PPO(obs_dim=input_size, action_dim=output_size) for key in env.observation_spaces.keys()}

num_episodes = 100
max_steps_per_episode = 500

for episode in range(num_episodes):
    obs = env.reset()  # Reset environment
    # print('Observation: ', obs)
    for step in range(max_steps_per_episode):
        log_probs = []
        for agent_key in ppo_agents.keys():
            agent_obs = obs[agent_key]
            agent_actions = ppo_agents[agent_key].get_action(agent_obs['observation'])
            obs[agent_key]['action'] = torch.Tensor([agent_actions]).to(torch.int64)
        new_td= env.step(obs)
        next_obs = new_td['next']
        done = new_td['done']

        for agent_key in ppo_agents.keys():
            ppo_agents[agent_key].update(obs[agent_key]['observation'], obs[agent_key]['action'], next_obs[agent_key]['reward'], next_obs[agent_key]['observation'], done)
            obs = next_obs


        
        # print(next)
        # for key in ppo_agents.keys():
        #     print(key)
        #     ppo_agents[key].step(obs[key]['observation'], obs[key]['action'], next_obs[key]['reward'], next_obs[key]['observation'], log_probs[key], done)  # Update the policy
        # obs = next_obs

