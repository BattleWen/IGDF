import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class Discriminator(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_dim, disc_obs_index=None):
        super(Discriminator, self).__init__()
        
        self.disc_obs_index = disc_obs_index
        if disc_obs_index is not None:
            num_inputs = len(disc_obs_index)

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear31 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear32 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear33 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, num_outputs)

        self.apply(weights_init_)

    def forward(self, state):
        if self.disc_obs_index is not None:
            state = state[:, self.disc_obs_index]
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        # x = F.relu(self.linear33(F.relu(self.linear32(F.relu(self.linear31(x))))))
        x = self.linear4(x)
        x = 2 * F.tanh(x) # TBD

        return x # regression label, unnormalized
    

 