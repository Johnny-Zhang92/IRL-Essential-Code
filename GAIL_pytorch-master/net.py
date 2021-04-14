import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


class disc_policy_net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(disc_policy_net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.output_dim)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, 1)

    def act(self, input):
        probs = self.forward(input)
        dist = Categorical(probs)
        action = dist.sample()
        action = action.detach().item()
        return action


class cont_policy_net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(cont_policy_net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.output_dim)

    def forward(self, input):
        x = torch.tanh(self.fc1(input))
        x = torch.tanh(self.fc2(x))
        mu = self.fc3(x)
        return mu

    def act(self, input):
        mu = self.forward(input)
        sigma = torch.ones_like(mu)
        dist = Normal(mu, sigma)
        action = dist.sample().detach().item()
        return action

    def get_distribution(self, input):
        mu = self.forward(input)
        sigma = torch.ones_like(mu)
        dist = Normal(mu, sigma)
        return dist


class value_net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(value_net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.output_dim)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class discriminator(nn.Module):
    def __init__(self, input_dim):
        super(discriminator, self).__init__()
        self.input_dim = input_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)
