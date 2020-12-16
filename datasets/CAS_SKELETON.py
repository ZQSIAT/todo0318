# -*- coding: utf-8 -*-
"""
This python code is designed for reading CAS_MHAD skeleton data.
"""

from multiprocessing import Process
import os
import pandas
import torch.utils.data as data
import numpy as np
import scipy.io as scio
import random
import math
from transforms.temporal_transforms import sparse_sampling_frames_from_segments_dual,\
    variant_sparse_sampling_frames_from_segments_dual

def mat_load(path):
    data = scio.loadmat(path)
    return data
    pass

def get_path_list_from_file_list(file_format, file_list_file):
    file_dir_list = []
    label_list = []
    with open(file_list_file, "r") as flp:

        pass
    pass

if __name__ == '__main__':
    a = "/home/lhj/zqs/dataloader_test/CAS.mat"
    b = mat_load(a)
    print(b.keys())

    pass