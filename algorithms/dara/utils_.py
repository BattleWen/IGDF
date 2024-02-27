import math
import torch

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
        
        
def predefineddeltar(state, envs, envt):
    if envt == 'Map-v0':
        return 0
    if envt == 'Mapa-v0':
        barw = 0.03
        barh = 0.50 
        if (state[0]>0.5-barw and state[0]<0.5+barw and state[1]>=-1 and state[1]<barh):
            return -23
        else:
            return 0
    if envt == 'Mapb-v0':
        barw = 0.03
        barh = 0.70 
        if (state[0]>0.5-barw and state[0]<0.5+barw and state[1]>=-1 and state[1]<barh):
            return -23
        else:
            return 0
    if envt == 'Mapc-v0':
        barw = 0.03
        barh = 0.25 
        inbarleft = (state[0]>0.5-barw and state[0]<0.5+barw and state[1]>0.5-barh and state[1]<0.5+barw)
        inbarup = (state[1]>0.5-barw and state[1]<0.5+barw and state[0]>0.5-barh and state[0]<0.5+barw)
        if inbarleft or inbarup: 
            return -23
        else:
            return 0
    return None 
        

import os.path as osp
import joblib
import numpy as np

def load_rep(fpath):
    rep = joblib.load(osp.join(fpath, 'reps.pkl'))
    return rep


def get_SREPS(args):
#    return joblib.load(osp.join(args.ctdir, 'reps-synthetic.pkl'))['images']
    if args.skill_rep == 0:
        SREPS = None
    elif args.skill_rep == 1:
        SREPS = None
    elif args.skill_rep == 2:
        SREPS = load_rep(args.ctdir)['states'][:, -1]
    elif args.skill_rep == 3:
        SREPS = load_rep(args.ctdir)['states']
    elif args.skill_rep == 4:
        SREPS = load_rep(args.ctdir)['images'][:, -1]
    elif args.skill_rep == 5:
        SREPS = load_rep(args.ctdir)['images']
    elif args.skill_rep == 6:
        SREPS = None
    return SREPS

    
def get_srep(SREPS, cz_onehot, skill_rep, index, t): 
#    return SREPS[index][t]/255.0
    if skill_rep == 1:
        return cz_onehot
    elif skill_rep in [2,4]:
        return SREPS[index]/255.0
    elif skill_rep in [3,5]:
        return SREPS[index][t]/255.0
    elif skill_rep == 6:
        pass
    return None


def get_transitionse(args, t, T):
    cs, ce = np.zeros(args.max_con), np.zeros(args.max_con)
    cs[args.trans_con[0]] = 1
    ce[args.trans_con[1]] = 1
    return [cs, ce, t/T]
    
def joint_store_sreps(args):
    if args.skill_rep == 0:
        SREPS = None
    elif args.skill_rep == 1:
        SREPS = None
    elif args.skill_rep == 2:
        SREPS = load_rep(args.ctdir)['states'][:, -1]
    elif args.skill_rep == 3:
        SREPS = load_rep(args.ctdir)['states']
    elif args.skill_rep == 4:
        SREPS = load_rep(args.ctdir)['images'][:, -1]
    elif args.skill_rep == 5:
        SREPS = load_rep(args.ctdir)['images']
    elif args.skill_rep == 6:
        SREPS = None
    return SREPS

    
def joint_get_sreps(SREPS, cz_onehot, skill_rep, index, t): 
#    return SREPS[index][t]/255.0
    if skill_rep == 1:
        return cz_onehot
    elif skill_rep in [2,4]:
        return SREPS[index]/255.0
    elif skill_rep in [3,5]:
        return SREPS[index][t]/255.0
    elif skill_rep == 6:
        pass
    return None
    

