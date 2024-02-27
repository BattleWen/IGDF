# LRoS implementation 
import joblib
import numpy as np
import torch
import gym
import time
from utils.mpi_tools import mpi_fork, proc_id, mpi_statistics_scalar, num_procs
# from utils.mpi_torch import average_gradients, sync_all_params
from utils.logx import EpochLogger
from utils_ import joint_store_sreps, joint_get_sreps
import os.path as osp
import random
import visdom
import os

from sac import DeltaCla, count_vars
from replay_memory import ReplayMemory
import h5py


def get_source_data(data_path):
    data = h5py.File(data_path, 'r')
    N = len(data['observations'])
    source_data = ReplayMemory(capacity=N)

    source_data.push(state=np.squeeze(np.array(data['observations'])),
                         action=np.squeeze(np.array(data['actions'])),
                         reward=np.squeeze(np.array(np.squeeze(data['rewards']))),
                         next_state=np.squeeze(np.array(data['next_observations'])),
                         done=np.squeeze(np.array(data['terminals'])))
    return source_data, np.shape(np.squeeze(data['observations']))[1], np.shape(np.squeeze(data['actions']))[1]

# def get_target_data(data_path):
#     data = h5py.File(data_path, 'r')
#     N = int(1e3)
#     target_data = ReplayMemory(capacity=N)
#
#     obs_ = []
#     next_obs_ = []
#     action_ = []
#     reward_ = []
#     done_ = []
#
#     for i in range(N - 1):
#         obs = data['observations'][i].astype(np.float32)
#         new_obs = data['observations'][i+1].astype(np.float32)
#         action = data['actions'][i].astype(np.float32)
#         reward = data['rewards'][i].astype(np.float32)
#         done_bool = bool(data['terminals'][i])
#
#         obs_.append(obs)
#         next_obs_.append(new_obs)
#         action_.append(action)
#         reward_.append(reward)
#         done_.append(done_bool)
#         if i%1000==0:
#             print(i)
#
#         if done_bool:
#             continue
#     obs_  = np.array(obs_)
#     next_obs_ =  np.array(next_obs_)
#     action_ = np.array(action_)
#     reward_ = np.array(reward_)
#     done_ = np.array(done_)
#
#     target_data.push(state=obs_,
#                          action=action_,
#                          reward=reward_,
#                          next_state=next_obs_,
#                          done=done_)
#     return target_data

def DAR(vis, args, logger_kwargs=dict(), save_freq=1000):
    logger = EpochLogger(**logger_kwargs)

    vas = vars(args)
    logger.save_config(locals())
    logger.add_vis(vis)

    print("Start...")
    source_data, state_dim, action_dim = get_source_data(
        data_path=r"dataset/source_dog/{}.hdf5".format(args.envs))

    print("Get source data.")
    # target_data = get_target_data(data_path=r"dataset/target/{}.hdf5".format(args.envt))
    # print("Get source data and target data.")

    agent = DeltaCla(state_dim, action_dim, args)

    # agent.load_deltar(fpath=os.path.abspath(__file__)[:-7]+"log/"+args.exp_name+"/"+args.sub_exp_name+ str(args.seed),itr=10)
    agent.load_deltar(
        fpath= "log/" + args.exp_name + "/" + args.sub_exp_name +"-"+ str(args.seed),
        itr=args.itr)

    num = int(1e5)
    i=0
    reward_new = []
    delta_coe = 0.01
    while i<len(source_data.state_buffer):
        state_batch = source_data.state_buffer[i:num+i]
        action_batch = source_data.action_buffer[i:num+i]
        next_state_batch = source_data.next_state_buffer[i:num+i]

        delta = agent.delta_dynamic(state_batch, action_batch, next_state_batch)
        # print(source_data.reward_buffer[i:num+i] + 0.1 * delta)

        # for j in range(100):
        #     print(source_data.reward_buffer[i:num + i][j], delta_coe*delta[j])
        # input()

        reward_new.extend(source_data.reward_buffer[i:num+i] + delta_coe * delta)

        # print(np.max(source_data.reward_buffer), np.min(source_data.reward_buffer))
        # print(reward_new)
        i+=num

    print(np.shape( np.array(source_data.state_buffer)))
    # reward_new = np.reshape(np.array(reward_new),newshape=(-1,1))
    print(np.shape(np.array(reward_new)))
    results = {
        'observations': source_data.state_buffer,
        'next_observations': source_data.next_state_buffer,
        'actions': source_data.action_buffer,
        'rewards': reward_new,
        'terminals': source_data.done_buffer,
    }

    os.makedirs("dataset/source_dog/dar_{}/".format(delta_coe), exist_ok=True)
    hfile = h5py.File(r"dataset/source_dog/dar_{}/{}_dar.hdf5".format( delta_coe, args.envs), 'w')

    for k in results:
        hfile.create_dataset(k, data=results[k], compression='gzip')
    hfile.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()  #
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--envs', type=str, default='random_05')
    parser.add_argument('--envt', type=str, default='walker2d_random')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--updates_per_step_cla', type=int, default=1000)

    parser.add_argument('--itr', type=int, default=0)
    # parser.add_argument('--env_num', type=int, default=0)

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
    parser.add_argument('--sub_exp_name', type=str, default=None)
    parser.add_argument('--ctdir', type=str, default=None)

    args = parser.parse_args()

    args.exp_name = '' + args.envs + '_' + args.envt

    PREF = 'delta_' + args.envs + '_' + args.envt
    vis = visdom.Visdom(env=PREF + str(args.seed))
    # vis = visdom.Visdom(server='http://localhost', port=8097)
    args.sub_exp_name = PREF
    mpi_fork(args.cpu)

    from utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.sub_exp_name, args.seed)
    DAR(vis, args, logger_kwargs=logger_kwargs)

    """
    python3 test_dog.py --seed=0 --envs=medium_expert --envt=dog_medium_expert --cuda --itr=10
    """
