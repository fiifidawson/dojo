import os
import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from network import ActorNetwork, CriticNetwork


class Agent():

    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99, update_actor_interval = 2,
                 warmup=1000, n_actions=2, max_size=1000000, layer1_size=256, layer2_size=128,
                 batch_size=100, noise=0.1):
        pass