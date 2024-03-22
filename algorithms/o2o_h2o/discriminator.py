import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop
import numpy as np

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
        self.linear4 = nn.Linear(hidden_dim, num_outputs)

        self.apply(weights_init_)

    def forward(self, state):
        if self.disc_obs_index is not None:
            state = state[:, self.disc_obs_index]
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        x = 2 * F.tanh(x) # TBD
        return x # regression label, unnormalized
    


class DeltaCla(object):
    def __init__(self, state_dim, action_dim, device, hidden_size, lr):

        self.device = torch.device("cuda" if cuda else "cpu")
        self.cla_sas = Discriminator(state_dim*2+action_dim, 2, hidden_size).to(device=self.device)
        self.cla_sa = Discriminator(state_dim+action_dim, 2, hidden_size).to(device=self.device)
        self.cla_sas_optim = RMSprop(self.cla_sas.parameters(), lr=lr)
        self.cla_sa_optim = RMSprop(self.cla_sa.parameters(), lr=lr)

    def delta_weight(self, s, a, ss):
        s = torch.FloatTensor(s).to(self.device)
        a = torch.FloatTensor(a).to(self.device)
        ss = torch.FloatTensor(ss).to(self.device)
        sas = torch.cat([s, a, ss], 1) 
        sa = torch.cat([s, a], 1)

        prob_sas    = F.softmax(self.cla_sas(sas)+self.cla_sa(sa), dim=1)
        prob_sa     = F.softmax(self.cla_sa(sa), dim=1)

        delata_sas  = np.log(prob_sas[:, 1]) - np.log(prob_sas[:, 0])
        delata_sa   = np.log(prob_sa[:, 1]) - np.log(prob_sa[:, 0])
        delta       = delata_sas - delata_sa

        weight      = np.exp(delta).detach()
        weight_clip = torch.clamp(weight, 1e-4, 1.)
        return weight_clip

    def update_param_cla(self, memorymu, memoryt, batch_size):
        losses = 0
        losssas = 0
        losssa = 0
        for index in [0, 1]:
            if index == 0 : memory = memorymu
            if index == 1 : memory = memoryt
            # _, state_batch, action_batch, _, next_state_batch, _, _ = memory.sample(batch_size=batch_size)
            state_batch, action_batch, _, next_state_batch, _ = memory.sample(batch_size=batch_size)

            # state_batch = state_batch + np.random.rand(batch_size, state_batch.shape[1]) * 0.001
            # action_batch = action_batch + np.random.rand(batch_size, action_batch.shape[1]) * 0.001 
            # next_state_batch = next_state_batch + np.random.rand(batch_size, next_state_batch.shape[1]) * 0.001
            
            state_batch = torch.FloatTensor(state_batch).to(self.device)
            next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
            action_batch = torch.FloatTensor(action_batch).to(self.device)
            
            state_action_state_batch = torch.cat([state_batch, action_batch, next_state_batch], 1) 
            state_action_batch = torch.cat([state_batch, action_batch], 1)
            # sort_index.shape = (batch_size, )
            sort_index = torch.tensor([index]*batch_size).to(self.device)
            # cla_sas_loss = F.cross_entropy(self.cla_sas(state_action_state_batch), sort_index)
            cla_sas_loss = F.cross_entropy(self.cla_sas(state_action_state_batch)+\
                                            self.cla_sa(state_action_batch).detach(), sort_index) 
            cla_sa_loss = F.cross_entropy(self.cla_sa(state_action_batch), sort_index)

            # cla_sas_loss = - torch.mean(F.softmax(self.cla_sas(state_action_state_batch), 1)[:, index])
            # cla_sa_loss = - torch.mean(F.softmax(self.cla_sa(state_action_batch), 1)[:, index])
            
            self.cla_sas_optim.zero_grad()
            self.cla_sa_optim.zero_grad()
            cla_sas_loss.backward()
            cla_sa_loss.backward()
            self.cla_sas_optim.step()
            self.cla_sa_optim.step()
            losssas = losssas + cla_sas_loss.item()
            losssa = losssa + cla_sa_loss.item()
        losses = losssas + losssa
        return losses, losssas, losssa
    
    def change_device2cpu(self):
        # self.policy.cpu()
        # self.critic.cpu()
        # self.critic_target.cpu()
        # self.disc.cpu()
        self.cla_sas.cpu()
        self.cla_sa.cpu()
        
    def change_device2device(self):
        # self.policy.to(device=self.device)
        # self.critic.to(device=self.device)
        # self.critic_target.to(device=self.device)
        # self.disc.to(device=self.device)
        self.cla_sas.to(device=self.device)
        self.cla_sa.to(device=self.device)
    

        
def count_vars(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

# from utils.mpi_tools import mpi_avg
#from utils.mpi_tools import proc_id

# def average_param(param):
#     for p in param:
# #        print(proc_id(), p.data.shape)
#         p.data.copy_(torch.Tensor(mpi_avg(p.data.numpy())))