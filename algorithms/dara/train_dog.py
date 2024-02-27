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


def get_target_data(data_path):
    data = h5py.File(data_path, 'r')
    N = len(data['observations'])
    target_data = ReplayMemory(capacity=N)

    obs_  = np.squeeze(np.array(data['observations']))
    next_obs_ =  np.squeeze(np.array(data['next_observations']))
    action_ = np.squeeze(np.array(data['actions']))

    reward_ = np.array(np.zeros(N))
    done_ = np.array(np.zeros(N))

    target_data.push(state=obs_,
                         action=action_,
                         reward=reward_,
                         next_state=next_obs_,
                         done=done_)
    return target_data


def DAR(vis, args, logger_kwargs=dict(), save_freq=10):
    logger = EpochLogger(**logger_kwargs)

    vas = vars(args)
    logger.save_config(locals())
    logger.add_vis(vis)

    print("Start...")

    source_data, state_dim, action_dim = get_source_data(
        data_path=r"dataset/source_dog/{}.hdf5".format(args.envs))

    print("Get source data.")
    print("obs dim:",state_dim," action dim:",action_dim)
    # target_data = get_target_data(data_path=r"dataset/target/{}.hdf5".format(args.envt),isMediumExpert=args.isMediumExpert)
    # target_data = get_target_data(data_path=r"dataset/target_cheetah/{}.hdf5".format(args.envt),
    #                               isMediumExpert=args.isMediumExpert)
    target_data = get_target_data(data_path=r"dataset/target_dog/{}.h5".format(args.envt),
                                 )
    print("Get source data and target data.")

    agent = DeltaCla(state_dim, action_dim, args)

    start_time = time.time()
    total_t = 0
    upnum = 0
    for epoch in range(args.epochs):
        for e in range(args.updates_per_step_cla):
            cal_loss = agent.update_param_cla(source_data, target_data, args.batch_size)
            logger.store(Delta=cal_loss)
        if proc_id() == 0:
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('Delta', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()
        if proc_id() == 0 and (epoch % save_freq == 0) or (epoch == args.epochs - 1):
            agent.change_device2cpu()
            logger.save_state({'envs': args.envs, 'envt': args.envt},
                              [{'sas': agent.cla_sas, 'sa': agent.cla_sa},
                               # agent.policy,
                               # agent.critic,
                               # agent.disc
                               ],
                              epoch)
            agent.change_device2device()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()  #
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--envs', type=str, default='random')
    parser.add_argument('--envt', type=str, default='walker2d_random')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--updates_per_step_cla', type=int, default=50)

    # parser.add_argument('--env_num', type=int, default=0)
    # parser.add_argument('--isMediumExpert', type=bool, default=False)

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

    args.exp_name = ''+ args.envs + '_' + args.envt

    PREF ='delta_' + args.envs + '_' + args.envt
    vis = visdom.Visdom(env=PREF + str(args.seed))
    # vis = visdom.Visdom(server='http://localhost', port=8097)
    args.sub_exp_name = PREF
    mpi_fork(args.cpu)

    from utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.sub_exp_name, args.seed)
    DAR(vis, args, logger_kwargs=logger_kwargs)


    #python3 train_dog.py --seed=0 --envs=medium --envt=dog_medium_expert --cuda
