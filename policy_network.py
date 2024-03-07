import torch.nn as nn
import torch

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        # Define your network architecture here
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return x