import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=64):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)

# Proximal Policy Optimization (PPO) Algorithm
class PPO:
    def __init__(self, obs_dim, action_dim, clip_ratio=0.2, gamma=0.99, lr=3e-4, epochs=10, hidden_size=64):

        self.actor_critic = ActorCritic(obs_dim, action_dim, hidden_size)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.epochs = epochs

    def get_action(self, state):
        with torch.no_grad():
            action_probs, _ = self.actor_critic(torch.tensor(state).float())
            dist = Categorical(action_probs)
            action = dist.sample()
        return action.item()

    # def update(self, states, actions, rewards, next_states, dones):
    #     states = torch.tensor(states).float()
    #     actions = torch.tensor(actions).long().unsqueeze(1)
    #     rewards = torch.tensor(rewards).float().unsqueeze(1)
    #     next_states = torch.tensor(next_states).float()
    #     dones = torch.tensor(dones).float().unsqueeze(1)

    #     old_action_probs, old_values = self.actor_critic(states)
    #     old_action_probs = old_action_probs.gather(1, actions)

    #     for _ in range(self.epochs):
    #         action_probs, values = self.actor_critic(states)
    #         dist = Categorical(action_probs)
    #         entropy = dist.entropy().mean()

    #         action_values = rewards + self.gamma * (1 - dones) * values.detach()
    #         advantages = action_values - old_values.detach()

    #         ratio = (action_probs.gather(1, actions) / old_action_probs).clamp(0, 1)
    #         surr1 = ratio * advantages
    #         surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages

    #         actor_loss = -torch.min(surr1, surr2).mean()
    #         critic_loss = nn.MSELoss()(values, action_values)

    #         loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

    #         self.optimizer.zero_grad()
    #         loss.backward(retain_graph=True)
    #         clip_grad_norm_(self.actor_critic.parameters(), 0.5)
    #         self.optimizer.step()

    def update(self, states, actions, rewards, next_states, dones):
        
        states = torch.tensor(states).float()
        actions = torch.tensor(actions).long().unsqueeze(1)
        rewards = torch.tensor(rewards).float().unsqueeze(1)
        next_states = torch.tensor(next_states).float()
        dones = torch.tensor(dones).float().unsqueeze(1)

        old_action_probs, old_values = self.actor_critic(states)
        old_action_probs = old_action_probs.gather(1, actions)

        for _ in range(self.epochs):
            torch.autograd.set_detect_anomaly(True)
            action_probs, values = self.actor_critic(states)
            dist = Categorical(action_probs)
            entropy = dist.entropy().mean()

            action_values = rewards + self.gamma * (1 - dones) * values.clone().detach()  # Avoid inplace detach
            advantages = action_values - old_values.clone().detach()  # Avoid inplace detach

            ratio = (action_probs.gather(1, actions) / old_action_probs).clamp(0, 1)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values, action_values)

            loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            clip_grad_norm_(self.actor_critic.parameters(), 0.5)
            self.optimizer.step()