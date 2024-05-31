"""
The networks for ReLara algorithm.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicActor(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.fc1 = nn.Linear(np.array(observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(action_space.shape))
        # action rescaling
        self.register_buffer("action_scale",
                             torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias",
                             torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32))
        self.log_std_max = 2
        self.log_std_min = -5

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class BasicQNetwork(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.fc1 = nn.Linear(np.array(observation_space.shape).prod() + np.prod(action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, activation=nn.ReLU()):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.activation = activation

    def forward(self, x):
        residual = x
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        x += residual
        x = self.activation(x)
        return x


class QNetworkResidual(nn.Module):
    def __init__(self, observation_space, action_space, block_num=3):
        super().__init__()
        self.fc1 = nn.Linear(np.array(observation_space.shape).prod() + np.prod(action_space.shape), 256)
        self.hidden_blocks = nn.ModuleList([ResidualBlock(256, 256) for _ in range(block_num)])
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = self.fc1(x)
        for block in self.hidden_blocks:
            x = block(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ActorResidual(nn.Module):

    def __init__(self, observation_space, action_space, block_num=3):
        super().__init__()
        self.fc1 = nn.Linear(np.array(observation_space.shape).prod(), 256)
        self.hidden_blocks = nn.ModuleList([ResidualBlock(256, 256) for _ in range(block_num)])
        self.fc2 = nn.Linear(256, 128)
        self.fc_mean = nn.Linear(128, np.prod(action_space.shape))
        self.fc_logstd = nn.Linear(128, np.prod(action_space.shape))
        # action rescaling
        self.register_buffer("action_scale",
                             torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias",
                             torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32))
        self.log_std_max = 2
        self.log_std_min = -5

    def forward(self, x):
        x = self.fc1(x)
        for block in self.hidden_blocks:
            x = block(x)
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
