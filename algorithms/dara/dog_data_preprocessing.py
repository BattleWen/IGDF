import numpy as np
import h5py
import os

# def get_file():
#     # 定义路径
#     path='dataset/target_dog/result'
#     # 设置空列表
#     file_list=[]
#     # 使用os.walk获取文件路径及文件
#     for home, dirs, files in os.walk(path):
#         # 遍历文件名
#         for filename in files:
#             # 将文件路径包含文件一同加入列表
#             file_list.append(os.path.join(home,filename))
#     # 赋值
#     return file_list
#
# file_list = get_file()
#
# N=300
# obs_ = []
# next_obs_ = []
# action_ = []
#
# for file in file_list:
#     data = h5py.File(file, 'r')
#     for i in range(N - 1):
#         obs = data['obs'][i].astype(np.float32)
#         new_obs = data['obs'][i + 1].astype(np.float32)
#         action = data['action'][i].astype(np.float32)
#         # reward = data['rewards'][i].astype(np.float32)
#         # done_bool = bool(data['terminals'][i])
#
#         obs_.append(obs)
#         next_obs_.append(new_obs)
#         action_.append(action)
#         # reward_.append(reward)
#         # done_.append(done_bool)
#
#
# obs_ = np.array(obs_)
# next_obs_ = np.array(next_obs_)
# action_ = np.array(action_)
# # reward_ = np.array(reward_)
# # done_ = np.array(done_)
#
# file_name="dataset/target_dog/dog_medium_expert.h5"
#
# # # 创建文件并写入数据
# if not os.path.exists(file_name):
#     f = h5py.File(file_name, 'w')
# else:
#     f = h5py.File(file_name, 'a')
# # f.create_dataset('obs_{}'.format(ind), data=obs_list)
# # f.create_dataset('action_{}'.format(ind), data=action_list)
# f.create_dataset('observations', data=obs_)
# f.create_dataset('actions', data=action_)
# f.create_dataset('next_observations', data=next_obs_)
# f.close()

data_path = r"dataset/target_dog/dog_medium_expert.h5"
data = h5py.File(data_path, 'r')
print(len(data["observations"]))

