import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)
    
class PPO:
    def __init__(self, input_size, output_size):
        self.policy = Policy(input_size, output_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.001)
        self.value_network = nn.Linear(input_size, 1)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=0.001)

    def update_policy(self, states, actions, advantages, returns, old_log_probs, clip_param=0.2):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e+8)
        values = self.value_network(states)
        critic_loss = F.mse_loss(values, returns.unsqueeze(1))
        
        new_log_probs = torch.log(self.policy(states).gather(1, actions.unsqueeze(1)))
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        self.value_optimizer.zero_grad()
        critic_loss.backward()
        self.value_optimizer.step()
        
def train_ppo(env, num_episodes=1000, max_steps=1000):
    input_size = env.observation_spec.shape[0]
    output_size = env.action_space.n
    agent = PPO(input_size, output_size)

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        print(f"Episode {episode+1}: Total Reward = {episode_reward}")
        for step in range(max_steps):
            state = torch.FloatTensor(state)
            action_probs = agent.policy(state)
            action = torch.multinomial(action_probs, 1).item()
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            agent.update_policy(
                torch.FloatTensor(state),
                torch.LongTensor([action]),
                torch.FloatTensor([reward]),
                torch.FloatTensor([episode_reward]),
                torch.log(action_probs[action])
            )
            
            state = next_state
            
            if done:
                break


        print(f"Episode {episode+1}: Total Reward = {episode_reward}")

