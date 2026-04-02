import os
import torch as T
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

class CriticNetwork(nn.Module):

    def __init__(self, input_dims, n_actions, fc1_dims=256, fc2_dims=128, name='critic', checkpoint_dir='tmp/td3', learning_rate=10e-3):
        super(CriticNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = 