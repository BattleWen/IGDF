import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD, RMSprop
from utils_ import soft_update, hard_update
from model import Discriminator
import random
import numpy as np


class DeltaCla(object):
    def __init__(self, state_dim, action_dim, args):

        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.cla_sas = Discriminator(state_dim*2+action_dim, 2, args.hidden_size).to(device=self.device)
        self.cla_sa = Discriminator(state_dim+action_dim, 2, args.hidden_size).to(device=self.device)
        self.cla_sas_optim = RMSprop(self.cla_sas.parameters(), lr=args.lr)
        self.cla_sa_optim = RMSprop(self.cla_sa.parameters(), lr=args.lr)

    def delta_dynamic(self, s, a, ss):
        # s = torch.FloatTensor(s).to(self.device).unsqueeze(0)
        # a = torch.FloatTensor(a).to(self.device).unsqueeze(0)
        # ss = torch.FloatTensor(ss).to(self.device).unsqueeze(0)
        s = torch.FloatTensor(s).to(self.device)
        a = torch.FloatTensor(a).to(self.device)
        ss = torch.FloatTensor(ss).to(self.device)
        sas = torch.cat([s, a, ss], 1) 
        sa = torch.cat([s, a], 1)

        # print("sas_size:",sas.size())
        
        # prob_sas = F.softmax(self.cla_sas(sas), dim=1).detach().cpu().numpy()
        # print(self.cla_sas(sas))
        # print(self.cla_sa(sa))
        # input()
        prob_sas = F.softmax(self.cla_sas(sas)+self.cla_sa(sa), dim=1).detach().cpu().numpy()
        # print(prob_sas)
        delata_sas = np.log(prob_sas[:, 1]) - np.log(prob_sas[:, 0])

        # delata_sas = np.log(max(prob_sas[:,1], 1E-10)) - np.log(max(prob_sas[:,0], 1E-10))

        prob_sa = F.softmax(self.cla_sa(sa), dim=1).detach().cpu().numpy() 
        # delata_sa = np.log(max(prob_sa[0,1], 1E-10)) - np.log(max(prob_sa[0,0], 1E-10))
        # delata_sa = np.log(max(prob_sa[:, 1], 1E-10)) - np.log(max(prob_sa[:, 0], 1E-10))
        delata_sa = np.log(prob_sa[:, 1]) - np.log(prob_sa[:, 0])
        # print(delata_sas)
        # print(delata_sa)
        delta = delata_sas - delata_sa
        # print("delta:",delta)
        return np.clip(delta, -10, 10)
        # return delta 
        # if delta > -5:
        #     return 0
        # else:
        #     return -10 

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
    

    def load_deltar(self, fpath, itr='last'):
        if itr=='last':
            saves = [int(x.split('.')[1]) for x in os.listdir(fpath) if 'sa_' in x]
            itr = '%d'%max(saves) if len(saves) > 0 else ''
        else:
            itr = '%d'%itr
        print('loaded: ', os.path.join(fpath, 'sa_'+itr+'.pt'))
        sa = torch.load(os.path.join(fpath, 'sa_'+itr+'.pt'))
        self.cla_sa.load_state_dict(sa)
        print('loaded: ', os.path.join(fpath, 'sas_'+itr+'.pt'))
        sas = torch.load(os.path.join(fpath, 'sas_'+itr+'.pt'))
        self.cla_sas.load_state_dict(sas)
        self.cla_sa.eval()
        self.cla_sas.eval()
        return itr
    

    
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