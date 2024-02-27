import numpy as np
import h5py

# data_path = r"dataset/source/41/dar_0.1/medium_replay_dar.hdf5"
data_path = r"dataset/source_hopper/41/dar_0.1/medium_replay_dar.hdf5"
data = h5py.File(data_path, 'r')

print(len(data["observations"]))