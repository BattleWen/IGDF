# LRoS implementation 
import joblib
import numpy as np
import torch
import gym
import time
# from utils.mpi_tools import mpi_fork, proc_id, mpi_statistics_scalar, num_procs
# from utils.mpi_torch import average_gradients, sync_all_params
# from utils.logx import EpochLogger
from utils_ import joint_store_sreps, joint_get_sreps
import os.path as osp
import random

from sac import DeltaCla, count_vars
from replay_memory import ReplayMemory
import h5py
import wandb
import os

def get_source_data(data_path):
    data = h5py.File(data_path, 'r')
    N = len(data['observations']) # the number of trajectory N
    source_data = ReplayMemory(capacity=N)

    source_data.push(state=np.array(data['observations']),
                         action=np.array(data['actions']),
                         reward=np.array(np.squeeze(data['rewards'])),
                         next_state=np.array(data['next_observations']),
                         done=np.array(data['terminals']))
    return source_data, np.shape(data['observations'])[1], np.shape(data['actions'])[1]


def get_target_data(data_path,isMediumExpert):
    data = h5py.File(data_path, 'r')
    N = int(1e5)
    # N = int(1e3)
    target_data = ReplayMemory(capacity=N)

    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    for i in range(N - 1):
        obs = data['observations'][i].astype(np.float32)
        new_obs = data['observations'][i+1].astype(np.float32)
        action = data['actions'][i].astype(np.float32)
        reward = data['rewards'][i].astype(np.float32)
        done_bool = bool(data['terminals'][i])

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)

        if i%1000==0:
            print(i)

        if done_bool:
            continue

    if isMediumExpert:
        print("---------------------------")
        print("isMediumExpert.")
        print("----------------------------")
        for i in range(N - 1):
            obs = data['observations'][-N+i].astype(np.float32)
            new_obs = data['observations'][-N+i+1].astype(np.float32)
            action = data['actions'][-N+i].astype(np.float32)
            reward = data['rewards'][-N+i].astype(np.float32)
            done_bool = bool(data['terminals'][-N+i])

            obs_.append(obs)
            next_obs_.append(new_obs)
            action_.append(action)
            reward_.append(reward)
            done_.append(done_bool)

            if i%1000==0:
                print(i)

            if done_bool:
                continue

    obs_  = np.array(obs_)
    next_obs_ =  np.array(next_obs_)
    action_ = np.array(action_)
    reward_ = np.array(reward_)
    done_ = np.array(done_)

    target_data.push(state=obs_,
                         action=action_,
                         reward=reward_,
                         next_state=next_obs_,
                         done=done_)
    return target_data


def DAR(args, save_freq=10):
    wandb.init(
        config=vars(args),
        project='Classifier_train',
        group='DARA',
        name=f"DARA-{args.env_num}-{args.envt}-{args.seed}",
        save_code=True,
    )
    wandb.run.save()

    print("Start...")
    # source_data, state_dim, action_dim = get_source_data(data_path=r"dataset/source/{}/{}.hdf5".format(args.env_num,args.envs))
    # source_data, state_dim, action_dim = get_source_data(
    #     data_path=r"../DA/datasets/source_halfcheetah/{}/{}.hdf5".format(args.env_num, args.envs))
    source_data, state_dim, action_dim = get_source_data(
        data_path=r"../DA/datasets/source_walker2d/{}/buffer.hdf5".format(args.env_num))
    # source_data, state_dim, action_dim = get_source_data(
    #     data_path=r"dataset/source_walker2d/{}/{}.hdf5".format(args.env_num, args.envs))

    print("Get source data.")
    # # target_data = get_target_data(data_path=r"dataset/target/{}.hdf5".format(args.envt),isMediumExpert=args.isMediumExpert)
    # target_data = get_target_data(data_path=r"../DA/datasets/target_halfcheetah/{}_v2.hdf5".format(args.envt),
    #                               isMediumExpert=args.isMediumExpert)
    target_data = get_target_data(data_path=r"../DA/datasets/target_walker2d/{}_v2.hdf5".format(args.envt),
                                  isMediumExpert=args.isMediumExpert)
    # target_data = get_target_data(data_path=r"dataset/target_walker2d/{}.hdf5".format(args.envt),
    #                               isMediumExpert=args.isMediumExpert)
    # print("Get source data and target data.")

    agent = DeltaCla(state_dim, action_dim, args)

    start_time = time.time()
    total_t = 0
    for epoch in range(args.epochs):
        for e in range(args.updates_per_step_cla):
            losses, losssas, losssa = agent.update_param_cla(source_data, target_data, args.batch_size)
            total_t += 1
            wandb.log({
                "losses": losses,
                "losssas": losssas,
                "losssa": losssa,
                "Epoch": epoch,
                "Time": time.time() - start_time,
            }, step = total_t)
        if (epoch % save_freq == 0) or (epoch == args.epochs - 1):
            torch.save(agent.cla_sa.state_dict(), os.path.join(args.fpath, f"sa_{epoch}.pt"))
            torch.save(agent.cla_sas.state_dict(), os.path.join(args.fpath, f"sas_{epoch}.pt"))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()  #
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--envs', type=str, default='medium')
    parser.add_argument('--envt', type=str, default='halfcheetah_medium')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--updates_per_step_cla', type=int, default=50)

    parser.add_argument('--env_num', type=int, default=0)
    parser.add_argument('--isMediumExpert', type=bool, default=False)

    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')

    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    parser.add_argument('--cpu', type=int, default=1)

    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--checkpoints', type=str, default='./checkpoints')
    parser.add_argument('--fpath', type=str, default=None)

    args = parser.parse_args()

    # args.exp_name = ''+ args.envs + '_' + args.envt
    args.fpath = f'{args.checkpoints}/{args.env_num}-{args.envt}-{args.seed}/'
    if not os.path.exists(args.fpath):
        os.makedirs(args.fpath)
    
    DAR(args)

    #python3 train3.py --seed=42 --env_num=41 --envs=random --envt=walker2d_random --cuda

    # data = [
    #     {
    #         'observations': [state_0, state_1, state_2, ..., state_N-1, ..., state_N],
    #         'actions': [action_0, action_1, action_2, ..., action_N-1, ..., action_N],
    #         'rewards': [reward_0, reward_1, reward_2, ..., reward_N-1, ..., reward_N],
    #         'next_observations': [next_state_0, next_state_1, next_state_2, ..., next_state_N-1, next_state_N]
    #     },
    # ]
    # state_1 ==  next_state_0 

    # \delta_r = W(p(target|s,a,s')||q(source|s,a,s')) - W(p(target|s,a)||q(source|s,a)) 

