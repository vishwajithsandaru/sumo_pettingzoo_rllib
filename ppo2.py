import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, input_size, output_size):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_actor = nn.Linear(32, output_size)
        self.fc_critic = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc_actor(x)
        value = self.fc_critic(x)
        return logits, value

class PPO:
    def __init__(self, policy_network, optimizer, clip_ratio=0.2, gamma=0.99, gae_lambda=0.95, value_loss_coef=0.5, entropy_coef=0.01):
        self.policy_network = policy_network
        self.optimizer = optimizer
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

    def act(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs)
            logits = self.policy_network.forward(obs)
            # print(logits)
            action_probs = F.softmax(logits, dim=1)
            # print(action_probs)
            action_distribution = torch.distributions.Categorical(action_probs)
            # print(action_distribution)
            action = action_distribution.sample()
            # print(action)
            log_prob = action_distribution.log_prob(action)
        return action.numpy(), log_prob.numpy()

    def compute_returns(self, rewards, dones, values):
        returns = np.zeros_like(rewards, dtype=np.float32)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_advantage = 0
        last_return = 0

        for t in reversed(range(len(rewards))):
            if dones[t]:
                returns[t] = rewards[t] + self.gamma * last_return
                td_error = returns[t] - values[t]
            else:
                returns[t] = rewards[t] + self.gamma * values[t+1]
                td_error = rewards[t] + self.gamma * values[t+1] - values[t]
            
            advantages[t] = last_advantage = td_error + self.gamma * self.gae_lambda * last_advantage
        
        return returns, advantages

    def update_policy(self, obs, actions, log_probs, returns, advantages):
        obs = torch.FloatTensor(obs)
        actions = torch.LongTensor(actions)
        log_probs = torch.FloatTensor(log_probs)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)

        logits, values = self.policy_network(obs)

        new_action_probs = F.softmax(logits, dim=-1)
        new_action_distribution = torch.distributions.Categorical(new_action_probs)
        new_log_probs = new_action_distribution.log_prob(actions)

        ratio = torch.exp(new_log_probs - log_probs)
        clipped_ratio = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        value_loss = F.mse_loss(values.squeeze(), returns)
        
        entropy_loss = -(new_action_probs * new_action_probs.log()).sum(-1).mean()

        total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def train(self, obs, actions, log_probs, rewards, dones):
        returns, advantages = self.compute_returns(rewards, dones, values)
        self.update_policy(obs, actions, log_probs, returns, advantages)