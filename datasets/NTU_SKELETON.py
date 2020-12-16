# -*- coding: utf-8 -*-
"""
This python code is designed for reading NTU_RGBD skeleton data.
"""

from multiprocessing import Process
import os
import pandas as pd
import torch.utils.data as data
import numpy as np
from numpy import linalg
import scipy.io as scio
import random
import math
import itertools
from sklearn import preprocessing

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from transforms.temporal_transforms import sparse_sampling_frames_from_segments_dual, variant_sparse_sampling_frames_from_segments_dual

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

def mat_load(path):
    data = scio.loadmat(path)
    return data
    pass
def draw_skeleton_joint_20(frame_idx, skeleton_date):
    dst_array = skeleton_date
    seq_len = dst_array.shape[1] // 3

    axis_x = dst_array[frame_idx, list(map(lambda m: 3 * m, range(seq_len)))]  # 16 x 20
    axis_y = dst_array[frame_idx, list(map(lambda m: 3 * m + 1, range(seq_len)))]
    axis_z = dst_array[frame_idx, list(map(lambda m: 3 * m + 2, range(seq_len)))]

    convert_axis_x = np.zeros((1, seq_len))
    convert_axis_y = np.zeros((1, seq_len))
    convert_axis_z = np.zeros((1, seq_len))

    for si in range(seq_len):
        convert_axis_x[si] = axis_x[si] - axis_x[0]
        convert_axis_y[si] = axis_y[si] - axis_y[0]
        convert_axis_z[si] = axis_z[si] - axis_z[0]
        pass

    # dst_array[:, list(map(lambda m: 3 * m, range(seq_len)))] = convert_axis_x
    # dst_array[:, list(map(lambda m: 3 * m + 1, range(seq_len)))] = convert_axis_y
    # dst_array[:, list(map(lambda m: 3 * m + 2, range(seq_len)))] = convert_axis_z
    x = convert_axis_x
    y = convert_axis_y
    z = convert_axis_z
    fig = plt.figure()
    draw_bone = Axes3D(fig)

    # x = np.array(skeleton_date[frame_idx, list(map(lambda m: 3 * m, range(20)))])
    # y = np.array(skeleton_date[frame_idx, list(map(lambda m: 3 * m + 1, range(20)))])
    # z = np.array(skeleton_date[frame_idx, list(map(lambda m: 3 * m + 2, range(20)))])
    # print(x[0])
    # print(list(map(lambda m: 3 * m, range(20))))
    bone_list = [[0, 1], [1, 2], [2, 3], [2, 4], [2, 8], [4, 5], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11], [0, 12], [0, 16], [16, 17], [17, 18], [18, 19], [12, 13], [13, 14], [14, 15]]
    draw_bone.scatter(x, y, z, c='r')
    for i, ICount in enumerate(bone_list):
        # draw_bone.plot(list(map(lambda n: x[n], ICount)), list(map(lambda n: y[n], ICount)), marker='o')
        draw_bone.plot(list(map(lambda n: x[n], ICount)), list(map(lambda n: y[n], ICount)), list(map(lambda n: z[n], ICount)), marker='o')
        pass
    # 设置坐标轴范围
    draw_bone.set_xlim(-1, 2)
    draw_bone.set_ylim(-1, 2)
    draw_bone.set_zlim(-1, 3)
    # 设置坐标轴名称
    draw_bone.set_xlabel("X")
    draw_bone.set_ylabel("Y")
    draw_bone.set_zlabel("Z")

    plt.show()
    pass
def draw_3_skeleton_joint_20(frame_idx, skeleton_date_1, skeleton_date_2, skeleton_date_3):
    fig = plt.figure()
    draw_bone = Axes3D(fig)
    x = np.array(skeleton_date_1[frame_idx, list(map(lambda m: 3 * m, range(20)))])
    y = np.array(skeleton_date_1[frame_idx, list(map(lambda m: 3 * m + 1, range(20)))])
    z = np.array(skeleton_date_1[frame_idx, list(map(lambda m: 3 * m + 2, range(20)))])
    x1 = np.array(skeleton_date_2[frame_idx, list(map(lambda m: 3 * m, range(20)))])
    y1 = np.array(skeleton_date_2[frame_idx, list(map(lambda m: 3 * m + 1, range(20)))])
    z1 = np.array(skeleton_date_2[frame_idx, list(map(lambda m: 3 * m + 2, range(20)))])
    x2 = np.array(skeleton_date_3[frame_idx, list(map(lambda m: 3 * m, range(20)))])
    y2 = np.array(skeleton_date_3[frame_idx, list(map(lambda m: 3 * m + 1, range(20)))])
    z2 = np.array(skeleton_date_3[frame_idx, list(map(lambda m: 3 * m + 2, range(20)))])
    # print(x[0])
    # print(list(map(lambda m: 3 * m, range(20))))
    bone_list = [[0, 1], [1, 2], [2, 3], [2, 4], [2, 8], [4, 5], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11], [0, 12], [0, 16], [16, 17], [17, 18], [18, 19], [12, 13], [13, 14], [14, 15]]
    draw_bone.scatter(x, y, z, c='r')
    draw_bone.scatter(x1, y1, z1, c='g')
    draw_bone.scatter(x2, y2, z2, c='r')
    for i, ICount in enumerate(bone_list):
        # draw_bone.plot(list(map(lambda n: x[n], ICount)), list(map(lambda n: y[n], ICount)), marker='o')
        draw_bone.plot(list(map(lambda n: x[n], ICount)), list(map(lambda n: y[n], ICount)), list(map(lambda n: z[n], ICount)), color='green', linewidth=4, marker='o')
        draw_bone.plot(list(map(lambda n: x1[n], ICount)), list(map(lambda n: y1[n], ICount)), list(map(lambda n: z1[n], ICount)), marker='o')
        draw_bone.plot(list(map(lambda n: x2[n], ICount)), list(map(lambda n: y2[n], ICount)), list(map(lambda n: z2[n], ICount)), color='blue', linewidth=4,marker='o')

        pass
    # 设置坐标轴范围
    draw_bone.set_xlim(-1, 2)
    draw_bone.set_ylim(-1, 2)
    draw_bone.set_zlim(-1, 3)
    # 设置坐标轴名称
    draw_bone.set_xlabel("X")
    draw_bone.set_ylabel("Y")
    draw_bone.set_zlabel("Z")

    plt.show()
    pass

def draw_cos_surface_skeleton_joint_20(frame_idx, skeleton_date):
    fig = plt.figure()
    draw_bone = Axes3D(fig)

    pass
def create_combination(n, m): # Generate all combinations of n numbers from m numbers
    n_in_m_combinations = []
    n_in_m_combinations = list(itertools.combinations(range(m),n))
    return n_in_m_combinations
    pass

def map_25_to_20(skeleton_data, keys = "kb"):# Map 25 nodes to 20
    kb_data_25 = skeleton_data[keys]
    kb_data_20 = np.zeros((kb_data_25.shape[0], 60))
    for i in [0, 1, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19]:
        kb_data_20[:, [3 * i, 3 * i + 1, 3 * i + 2]] = kb_data_25[:, [3 * i, 3 * i + 1, 3 * i + 2]]
        pass
    kb_data_20[:, [6, 7, 8]] = np.array([(kb_data_25[:, 6] + kb_data_25[:, 60]) / 2,
                                         (kb_data_25[:, 7] + kb_data_25[:, 61]) / 2,
                                         (kb_data_25[:, 8] + kb_data_25[:, 62]) / 2]).T
    kb_data_20[:, [21, 22, 23]] = np.array([(kb_data_25[:, 21] + kb_data_25[:, 63] + kb_data_25[:, 66]) / 3,
                                            (kb_data_25[:, 22] + kb_data_25[:, 64] + kb_data_25[:, 67]) / 3,
                                            (kb_data_25[:, 23] + kb_data_25[:, 65] + kb_data_25[:, 68]) / 3]).T
    kb_data_20[:, [33, 34, 35]] = np.array([(kb_data_25[:, 33] + kb_data_25[:, 69] + kb_data_25[:, 72]) / 3,
                                            (kb_data_25[:, 34] + kb_data_25[:, 70] + kb_data_25[:, 73]) / 3,
                                            (kb_data_25[:, 35] + kb_data_25[:, 71] + kb_data_25[:, 74]) / 3]).T

    return kb_data_20
    pass

def get_path_list_from_file_list(file_dirs, file_list_file):
    file_dir_list = []
    label_list = []
    frame_list = []
    with open(file_list_file, "r") as flp:
        for line in flp.readlines():
            flp_line = line.strip("\n").split("\t")
            target_file_path = file_dirs + "/" + flp_line[0]
            target_label = int(flp_line[1])
            target_length = int(flp_line[2])

            file_dir_list.append(target_file_path)
            label_list.append(target_label)
            frame_list.append(target_length)

            pass
        flp.close()
        pass
    return file_dir_list, label_list, frame_list
    pass

class NTURGBD_SKELETON(data.Dataset):
    def __init__(self, index_parm, spatial_transform=None, temporal_transform=None):
        self.modality = index_parm["modality"]
        self.is_skeleton_transform_velocity = index_parm["is_skeleton_transform_velocity"]
        self.subset_type = index_parm["subset_type"]
        self.temporal_parm = index_parm["temporal_param"]
        self.spatial_parm = index_parm["spatial_param"]

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform

        self.file_dirs = index_parm["file_dirs"]
        self.skeleton_format = index_parm["file_format"] + ".mat"

        logger = index_parm["logger"]

        self.file_list, self.label_list, self.frame_list = get_path_list_from_file_list(self.file_dirs, index_parm["file_subset_list"])

        self.combinations_3_20 = create_combination(3, 20)
        self.combinations_2_20 = create_combination(2, 20)
        pass
    pass

    def __getitem__(self, index):
        target_file_dir = self.file_list[index]
        target_label = self.label_list[index]
        target_length = self.frame_list[index]

        target_data = self._loading_data(target_file_dir, target_length)

        return target_data, target_label, target_length
        pass

    def __len__(self):
        return len(self.file_list)
        pass

    def _loading_data(self, file_path, file_count):
        if self.modality == "Skeleton":
            temporal_param = self.temporal_parm

            if temporal_param is None:
                file_idx = None
                pass
            else:
                if "dynamic_snapshot_sampling" in temporal_param.keys():
                    scales = temporal_param["dynamic_snapshot_sampling"]["scales"]
                    segments = temporal_param["dynamic_snapshot_sampling"]["segments"]
                    sampling_type = temporal_param["dynamic_snapshot_sampling"]["sampling_type"]

                    file_count_scales = math.ceil(random.choice(scales) * file_count)# pick one scales[0.5 0.8 1] in random
                    file_idx = sparse_sampling_frames_from_segments_dual(file_count_scales, segments, sampling_type)
                    pass
                elif "variant_snapshot_sampling" in temporal_param.keys():
                    raise NotImplementedError
                    pass
                elif "snapshot_sampling" in temporal_param.keys():
                    segments = temporal_param["snapshot_sampling"]["segments"]
                    sampling_type = temporal_param["snapshot_sampling"]["sampling_type"]

                    file_idx = sparse_sampling_frames_from_segments_dual(file_count, segments, sampling_type)
                    pass
                else:
                    file_idx = None
                    pass

            norm_val = False
            if self.is_skeleton_transform_velocity:
                data_processed = read_ntu_skeleton_angular_velocity(file_path, file_count, skeleton_loader=mat_load, file_idx=file_idx, combinations_3_20=self.combinations_3_20)

                pass
            else:
                # data_processed = read_skeleton_momentum(file_path, file_count, skeleton_loader=mat_load, file_idx=file_idx)
                # data_processed = read_ntu_skeleton3p_cos_3x16x1140x1(file_path, file_count, skeleton_loader=mat_load, file_idx=file_idx, combinations_3_20=self.combinations_3_20)
                data_processed = fusion_momentum_and_pojm3d(file_path, file_count, skeleton_loader=mat_load, file_idx=file_idx, combinations_3_20=self.combinations_3_20)
                pass

            if self.temporal_transform is not None:
                data_processed = self.temporal_transform(data_processed)
                pass
            if self.spatial_transform is not None:
                data_processed = self.spatial_transform(data_processed)
                pass
            pass
        else:
            raise ValueError("Error !! Unknown modality: {:s}.".format(self.modality))
            pass
        return data_processed
        pass

def read_ntu_skeleton_cos_old_slow(skeleton_file_dir, skeleton_count, skeleton_loader=None, file_idx=None, combinations_3_20 = None):

    if file_idx is not None:
        frame_list = file_idx
    else:
        frame_list = list(range(skeleton_count))

    assert os.path.exists(skeleton_file_dir), "{:s} does not exist!".format(skeleton_file_dir)
    skeleton_data = skeleton_loader(skeleton_file_dir)
    # Map 25 nodes to 20
    if (("kb" in skeleton_data.keys()) and ("kb2" not in skeleton_data.keys())):# single people action
        kb_data_20 = map_25_to_20(skeleton_data, "kb")
        pass
    elif("kb2" in skeleton_data.keys() and ("kb" in skeleton_data.keys())):# Double action, there is a virtual person
        kb1_data_20 = map_25_to_20(skeleton_data, "kb")
        # draw_skeleton_joint_20(5, kb_data_20)
        kb2_data_20 = map_25_to_20(skeleton_data, "kb2")
        # draw_skeleton_joint_20(5, kb2_data_20)
        kb_data_20 = np.array((kb1_data_20 + kb2_data_20)/2)
        # draw_3_skeleton_joint_20(10,kb1_data_20,kb_data_20, kb2_data_20)
        pass
    else:
        print("{:s} was broken!".format(skeleton_file_dir))
        raise AssertionError
        pass
    # draw the skeleton
    # draw_skeleton_joint_20(5,kb_data_20)
    # plt.show()
    # raise RuntimeError
    skel_array = np.zeros((len(file_idx), 3, 30, 38))#F x C x H x W
    for j, JCount in enumerate(file_idx):
        temp_skeleton_array = []
        for i, ICount in enumerate(combinations_3_20):
            vector_a = np.array([kb_data_20[JCount, 3 * ICount[0]] -kb_data_20[JCount, 3 * ICount[1]],\
                        kb_data_20[JCount, 3 * ICount[0] + 1] - kb_data_20[JCount, 3 * ICount[1] + 1],\
                        kb_data_20[JCount, 3 * ICount[0] + 2] - kb_data_20[JCount, 3 * ICount[1] + 2]])
            vector_b = np.array([kb_data_20[JCount, 3 * ICount[0]] -kb_data_20[JCount, 3 * ICount[2]],\
                        kb_data_20[JCount, 3 * ICount[0] + 1] - kb_data_20[JCount, 3 * ICount[2] + 1],\
                        kb_data_20[JCount, 3 * ICount[0] + 2] - kb_data_20[JCount, 3 * ICount[2] + 2]])
            vector_c = np.array([kb_data_20[JCount, 3 * ICount[1]] -kb_data_20[JCount, 3 * ICount[2]],\
                        kb_data_20[JCount, 3 * ICount[1] + 1] - kb_data_20[JCount, 3 * ICount[2] + 1],\
                        kb_data_20[JCount, 3 * ICount[1] + 2] - kb_data_20[JCount, 3 * ICount[2] + 2]])
            if((linalg.norm(vector_a) * linalg.norm(vector_b)) == 0):
                cos_theta_1 = 0
                pass
            else:
                cos_theta_1 = np.dot(vector_a, vector_b) / (linalg.norm(vector_a) * linalg.norm(vector_b))
                pass
            if((linalg.norm(vector_a) * linalg.norm(vector_c)) == 0):
                cos_theta_2 = 0
                pass
            else:
                cos_theta_2 = np.dot(vector_a, vector_c) / (linalg.norm(vector_a) * linalg.norm(vector_c))
                pass
            if((linalg.norm(vector_b) * linalg.norm(vector_c)) == 0):
                cos_theta_3 = 0
                pass
            else:
                cos_theta_3 = np.dot(vector_b, vector_c) / (linalg.norm(vector_b) * linalg.norm(vector_c))
                pass

            temp_skeleton_array.append([cos_theta_1, cos_theta_2, cos_theta_3])
            pass
        skeleton_temp_array = np.concatenate([np.expand_dims(x, 0) for x in temp_skeleton_array], axis=0)
        skeleton_temp_array = skeleton_temp_array.T.reshape(3,30,38)
        skel_array[j, :, :, :] = skeleton_temp_array

        pass
    # print(skel_array.shape)
    # print(skel_array[0,:,:,:])
    # print("\n")
    # print(skel_array[1,:,:,:])
    # raise RuntimeError
    return skel_array # F x C x H x W
    pass

def read_ntu_skeleton_copy(skeleton_file_dir, skeleton_count, skeleton_loader=None, file_idx=None, combinations_3_20 = None):
    assert os.path.exists(skeleton_file_dir), "{:s} does not exist!".format(skeleton_file_dir)
    skeleton_data = skeleton_loader(skeleton_file_dir)
    if (("kb" in skeleton_data.keys()) and ("kb2" not in skeleton_data.keys())):# single people action
        kb_data_20 = skeleton_data["kb"]

        pass
    elif("kb2" in skeleton_data.keys() and ("kb" in skeleton_data.keys())):# Double action, there is a virtual person
        kb1_data_20 = skeleton_data["kb"]
        # draw_skeleton_joint_20(5, kb_data_20)
        kb2_data_20 = skeleton_data["kb2"]
        # draw_skeleton_joint_20(5, kb2_data_20)
        kb_data_20 = np.array((kb1_data_20 + kb2_data_20)/2)
        # draw_3_skeleton_joint_20(10,kb1_data_20,kb_data_20, kb2_data_20)

        pass
    else:
        print("{:s} was broken!".format(skeleton_file_dir))
        raise AssertionError
        pass

    dst_array = np.array(kb_data_20[file_idx, :])  # 16 * 60(75)
    # dst_array = (dst_array - 0.844379) / 1.552702 # ntu_grb+d
    dst_array = (dst_array - 0.809465) / 1.529892 # cas_mhad
    # dst_array = (dst_array - 0.831969) / 1.475146  # utd_mhad
    # dst_array = (dst_array - 0.707839) / 1.156303  # utd_mvhad

    seq_len = dst_array.shape[1] // 3   # 20(25)
    skel_array = np.zeros(len(file_idx), 3, seq_len, seq_len)  # 16 x 3 x 20 x 20

    axis_x = dst_array[:, list(map(lambda m: 3 * m, range(seq_len)))]  # 16 x 20
    axis_y = dst_array[:, list(map(lambda m: 3 * m + 1, range(seq_len)))]  # 16 x 20
    axis_z = dst_array[:, list(map(lambda m: 3 * m + 2, range(seq_len)))]  # 16 x 20

    array_axis_x = np.zeros((len(file_idx), seq_len, seq_len))
    array_axis_y = np.zeros((len(file_idx), seq_len, seq_len))
    array_axis_z = np.zeros((len(file_idx), seq_len, seq_len))

    for si in range(len(file_idx)):
        array_axis_x[si, :, :] = preprocessing.normalize(np.tile(axis_x[si, :], (seq_len, 1)))
        array_axis_y[si, :, :] = preprocessing.normalize(np.tile(axis_y[si, :], (seq_len, 1)))
        array_axis_z[si, :, :] = preprocessing.normalize(np.tile(axis_z[si, :], (seq_len, 1)))
        pass

    # print(array_axis_x[0,:,:])
    # print(array_axis_x.shape)
    # raise RuntimeError

    skel_array[:, 0, :, :] = array_axis_x
    skel_array[:, 1, :, :] = array_axis_y
    skel_array[:, 2, :, :] = array_axis_z

    return skel_array # F 16 x C 3 x H 25 x W 25
    pass

def read_ntu_skeleton_outer(skeleton_file_dir, skeleton_count, skeleton_loader=None, file_idx=None, combinations_3_20 = None):
    assert os.path.exists(skeleton_file_dir), "{:s} does not exist!".format(skeleton_file_dir)
    skeleton_data = skeleton_loader(skeleton_file_dir)
    if (("kb" in skeleton_data.keys()) and ("kb2" not in skeleton_data.keys())):# single people action
        kb_data_20 = skeleton_data["kb"]

        pass
    elif("kb2" in skeleton_data.keys() and ("kb" in skeleton_data.keys())):# Double action, there is a virtual person
        kb1_data_20 = skeleton_data["kb"]
        # draw_skeleton_joint_20(5, kb_data_20)
        kb2_data_20 = skeleton_data["kb2"]
        # draw_skeleton_joint_20(5, kb2_data_20)
        kb_data_20 = np.array((kb1_data_20 + kb2_data_20)/2)
        # draw_3_skeleton_joint_20(10,kb1_data_20,kb_data_20, kb2_data_20)

        pass
    else:
        print("{:s} was broken!".format(skeleton_file_dir))
        raise AssertionError
        pass

    dst_array = np.array(kb_data_20[file_idx, :])  # 16 * 75
    # dst_array = (dst_array - 0.844379) / 1.552702 # ntu_grb+d
    # dst_array = (dst_array - 0.809465) / 1.529892 # cas_mhad
    dst_array = (dst_array - 0.831969) / 1.475146  # utd_mhad
    # dst_array = (dst_array - 0.707839) / 1.156303  # utd_mvhad

    seq_len = dst_array.shape[1] // 3
    skel_array = np.zeros((len(file_idx), 3, seq_len, seq_len))

    axis_x = dst_array[:, list(map(lambda m: 3 * m, range(seq_len)))]
    axis_y = dst_array[:, list(map(lambda m: 3 * m + 1, range(seq_len)))]
    axis_z = dst_array[:, list(map(lambda m: 3 * m + 2, range(seq_len)))]

    array_axis_x = np.zeros((len(file_idx), seq_len, seq_len))
    array_axis_y = np.zeros((len(file_idx), seq_len, seq_len))
    array_axis_z = np.zeros((len(file_idx), seq_len, seq_len))

    for si in range(len(file_idx)):
        array_axis_x[si, :, :] = preprocessing.normalize(np.outer(axis_x[si, :], axis_x[si, :].T))
        array_axis_y[si, :, :] = preprocessing.normalize(np.outer(axis_y[si, :], axis_y[si, :].T))
        array_axis_z[si, :, :] = preprocessing.normalize(np.outer(axis_z[si, :], axis_z[si, :].T))
        pass

    # print(array_axis_x[0,:,:])
    # print(array_axis_x.shape)
    # raise RuntimeError

    skel_array[:, 0, :, :] = array_axis_x
    skel_array[:, 1, :, :] = array_axis_y
    skel_array[:, 2, :, :] = array_axis_z

    return skel_array # F 16 x C 3 x H 25 x W 25
    pass

def read_ntu_skeleton_cuboid(skeleton_file_dir, skeleton_count, skeleton_loader=None, file_idx=None, combinations_3_20 = None):
    assert os.path.exists(skeleton_file_dir), "{:s} does not exist!".format(skeleton_file_dir)
    skeleton_data = skeleton_loader(skeleton_file_dir)
    if (("kb" in skeleton_data.keys()) and ("kb2" not in skeleton_data.keys())):# single people action
        kb_data_20 = skeleton_data["kb"]
        pass
    elif("kb2" in skeleton_data.keys() and ("kb" in skeleton_data.keys())):# Double action, there is a virtual person
        kb1_data_20 = skeleton_data["kb"]
        # draw_skeleton_joint_20(5, kb_data_20)
        kb2_data_20 = skeleton_data["kb2"]
        # draw_skeleton_joint_20(5, kb2_data_20)
        kb_data_20 = np.array((kb1_data_20 + kb2_data_20)/2)
        # draw_3_skeleton_joint_20(10,kb1_data_20,kb_data_20, kb2_data_20)
        pass
    else:
        print("{:s} was broken!".format(skeleton_file_dir))
        raise AssertionError
        pass

    dst_array = np.array(kb_data_20[file_idx, :])  # 16 * 75
    # dst_array = (dst_array - 0.844379) / 1.552702 # ntu_grb+d
    # dst_array = (dst_array - 0.809465) / 1.529892 # cas_mhad
    # dst_array = (dst_array - 0.831969) / 1.475146  # utd_mhad
    dst_array = (dst_array - 0.707839) / 1.156303  # utd_mvhad

    seq_len = dst_array.shape[1] // 3
    skel_array = np.zeros((len(file_idx), 3, seq_len, seq_len))

    axis_x = dst_array[:, list(map(lambda m: 3 * m, range(seq_len)))]
    axis_y = dst_array[:, list(map(lambda m: 3 * m + 1, range(seq_len)))]
    axis_z = dst_array[:, list(map(lambda m: 3 * m + 2, range(seq_len)))]

    array_axis_x = np.zeros((len(file_idx), seq_len, seq_len))
    array_axis_y = np.zeros((len(file_idx), seq_len, seq_len))
    array_axis_z = np.zeros((len(file_idx), seq_len, seq_len))

    for si in range(seq_len):
        array_axis_x[:, si, :] = np.abs(axis_x[:, np.tile(si, seq_len)] - axis_x[:, :])
        array_axis_y[:, si, :] = np.abs(axis_y[:, np.tile(si, seq_len)] - axis_y[:, :])
        array_axis_z[:, si, :] = np.abs(axis_z[:, np.tile(si, seq_len)] - axis_z[:, :])

        pass
    # print(array_axis_x)
    # print(array_axis_x.shape)
    # raise RuntimeError
    # print(array_axis_x[0,:,:])
    # print(array_axis_x.shape)
    # raise RuntimeError

    skel_array[:, 0, :, :] = array_axis_x
    skel_array[:, 1, :, :] = array_axis_y
    skel_array[:, 2, :, :] = array_axis_z

    return skel_array # F 16 x C 3 x H 25 x W 25
    pass

def read_ntu_skeleton_hoj3d(skeleton_file_dir, skeleton_count, skeleton_loader=None, file_idx=None, combinations_3_20 = None):

    assert os.path.exists(skeleton_file_dir), "{:s} does not exist!".format(skeleton_file_dir)
    skeleton_data = skeleton_loader(skeleton_file_dir)
    kb_data_20 = np.array(skeleton_data["kb"])
    # print(kb_data_20)
    # print(kb_data_20.shape)
    # raise RuntimeError

    dst_array = np.array(kb_data_20[:,:,file_idx])  # 7 x 12 x 16

    seq_len = len(file_idx)
    skel_array = np.zeros((len(file_idx), 3, 7, 12))


    array_axis_x = np.zeros((len(file_idx), 7, 12))
    array_axis_y = np.zeros((len(file_idx), 7, 12))
    array_axis_z = np.zeros((len(file_idx), 7, 12))

    for si in range(seq_len):
        array_axis_x[si, :, :] = dst_array[:,:,si]
        array_axis_y[si, :, :] = dst_array[:,:,si]
        array_axis_z[si, :, :] = dst_array[:,:,si]

        pass
    # print(array_axis_x)
    # print(array_axis_x.shape)
    # raise RuntimeError


    skel_array[:, 0, :, :] = array_axis_x # 16 x 3 x 7 x 12
    skel_array[:, 1, :, :] = array_axis_y
    skel_array[:, 2, :, :] = array_axis_z

    # print(skel_array[0, 0, :, :])
    # print(skel_array.shape)
    # raise RuntimeError
    return skel_array # F 16 x C 3 x H 7 x W 12
    pass

def read_ntu_skeleton_3p_cos(skeleton_file_dir, skeleton_count, skeleton_loader=None, file_idx=None, combinations_3_20 = None):

    # if file_idx is not None:
    #     frame_list = file_idx
    # else:
    #     frame_list = list(range(skeleton_count))

    assert os.path.exists(skeleton_file_dir), "{:s} does not exist!".format(skeleton_file_dir)
    skeleton_data = skeleton_loader(skeleton_file_dir)
    # Map 25 nodes to 20
    if (("kb" in skeleton_data.keys()) and ("kb2" not in skeleton_data.keys())):# single people action
        kb_data_20 = map_25_to_20(skeleton_data, "kb")
        pass
    elif("kb2" in skeleton_data.keys() and ("kb" in skeleton_data.keys())):# Double action, there is a virtual person
        kb1_data_20 = map_25_to_20(skeleton_data, "kb")
        # draw_skeleton_joint_20(5, kb_data_20)
        kb2_data_20 = map_25_to_20(skeleton_data, "kb2")
        # draw_skeleton_joint_20(5, kb2_data_20)
        kb_data_20 = np.array((kb1_data_20 + kb2_data_20)/2)
        # draw_3_skeleton_joint_20(10,kb1_data_20,kb_data_20, kb2_data_20)
        pass
    else:
        print("{:s} was broken!".format(skeleton_file_dir))
        raise AssertionError
        pass

    # kb_data_20 = skeleton_data["kb"]
    # draw the skeleton
    # draw_skeleton_joint_20(5,kb_data_20)
    # plt.show()
    # raise RuntimeError
    skel_array = np.zeros((len(file_idx), 3, 30, 38))#F x C x H x W
    vector_a = np.zeros((len(combinations_3_20), len(file_idx), 3))
    vector_b = np.zeros((len(combinations_3_20), len(file_idx), 3))
    vector_c = np.zeros((len(combinations_3_20), len(file_idx), 3))
    dst_array = np.array(kb_data_20[file_idx,:])

    # Subtract the data mean
    # dst_array = (dst_array - 0.844379) / 1.552702 # ntu_grb+d
    # dst_array = (dst_array - 0.809465) / 1.529892 # cas_mhad
    # dst_array = (dst_array - 0.831969) / 1.475146  # utd_mhad
    dst_array = (dst_array - 0.707839) / 1.156303  # utd_mvhad

    # print(dst_array[:,[0,1,2]]-dst_array[:,[3,4,5]])
    # print((dst_array[:,[0,1,2]]-dst_array[:,[3,4,5]]).shape)
    # raise RuntimeError
    copy_dst_array = np.tile(dst_array,(len(combinations_3_20),1))
    copy_dst_array = copy_dst_array.reshape([len(combinations_3_20), len(file_idx), 60])  # 1140*16*60
    # print(np.array(copy_dst_array[2,:,[0,1,2]]-copy_dst_array[0,:,[3,4,5]]).T)
    # print(np.array(copy_dst_array[2,:,[0,1,2]]-copy_dst_array[0,:,[3,4,5]]).T.shape)
    # raise RuntimeError
    for i,ICount in enumerate(combinations_3_20):
        vector_a[i,:,:] = np.array(copy_dst_array[i, :, [3*ICount[0],3*ICount[0]+1,3*ICount[0]+2]] - copy_dst_array[i, :, [3*ICount[1],3*ICount[1]+1,3*ICount[1]+2]]).T
        vector_b[i,:,:] = np.array(copy_dst_array[i, :, [3*ICount[0],3*ICount[0]+1,3*ICount[0]+2]] - copy_dst_array[i, :, [3*ICount[2],3*ICount[2]+1,3*ICount[2]+2]]).T
        vector_c[i,:,:] = np.array(copy_dst_array[i, :, [3*ICount[1],3*ICount[1]+1,3*ICount[1]+2]] - copy_dst_array[i, :, [3*ICount[2],3*ICount[2]+1,3*ICount[2]+2]]).T
        # raise RuntimeError
        pass
    # get the norm of the vector
    norm_vector_ab = linalg.norm(vector_a, axis=2) * linalg.norm(vector_b, axis=2)
    norm_vector_ac = linalg.norm(vector_a, axis=2) * linalg.norm(vector_c, axis=2)
    norm_vector_bc = linalg.norm(vector_b, axis=2) * linalg.norm(vector_c, axis=2)
    # Get matrix dot product
    product_vector_ab = ((vector_a[:, :, 0] * vector_b[:, :, 0]) + (vector_a[:, :, 1] * vector_b[:, :, 1]) + (vector_a[:, :, 2] * vector_b[:, :, 2]))
    product_vector_ac = ((vector_a[:, :, 0] * vector_c[:, :, 0]) + (vector_a[:, :, 1] * vector_c[:, :, 1]) + (vector_a[:, :, 2] * vector_c[:, :, 2]))
    product_vector_bc = ((vector_b[:, :, 0] * vector_c[:, :, 0]) + (vector_b[:, :, 1] * vector_c[:, :, 1]) + (vector_b[:, :, 2] * vector_c[:, :, 2]))

    # Prevent the denominator from being 0
    norm_vector_ab[norm_vector_ab == 0] = 1
    norm_vector_ac[norm_vector_ac == 0] = 1
    norm_vector_bc[norm_vector_bc == 0] = 1
    product_vector_ab[norm_vector_ab == 0] = 0
    product_vector_ac[norm_vector_ac == 0] = 0
    product_vector_bc[norm_vector_bc == 0] = 0

    # Get the cos theta of the vector
    skel_array[:, 0, :, :] = (product_vector_ab / norm_vector_ab).T.reshape(len(file_idx), 30, 38)
    skel_array[:, 1, :, :] = (product_vector_ac / norm_vector_ac).T.reshape(len(file_idx), 30, 38)
    skel_array[:, 2, :, :] = (product_vector_bc / norm_vector_bc).T.reshape(len(file_idx), 30, 38)
    # skel_array[:, 0, :, :] = preprocessing.normalize((product_vector_ab / norm_vector_ab).T, norm='l1').reshape(len(file_idx), 30, 38)
    # skel_array[:, 1, :, :] = preprocessing.normalize((product_vector_ac / norm_vector_ac).T, norm='l1').reshape(len(file_idx), 30, 38)
    # skel_array[:, 2, :, :] = preprocessing.normalize((product_vector_bc / norm_vector_bc).T, norm='l1').reshape(len(file_idx), 30, 38)

    # temp_skel_array = np.array([(product_vector_ab / norm_vector_ab), (product_vector_ac / norm_vector_ac), (product_vector_bc / norm_vector_bc)]).reshape([len(file_idx),len(combinations_3_20), 3])
    # skel_array = temp_skel_array.reshape([len(file_idx), 3, 30, 38])
    # print(skel_array)
    # print(skel_array.shape)
    return skel_array # F x C x H x W
    pass

def read_ntu_skeleton_3p_cos_3x38x30x30(skeleton_file_dir, skeleton_count, skeleton_loader=None, file_idx=None, combinations_3_20 = None):

    # if file_idx is not None:
    #     frame_list = file_idx
    # else:
    #     frame_list = list(range(skeleton_count))

    assert os.path.exists(skeleton_file_dir), "{:s} does not exist!".format(skeleton_file_dir)
    skeleton_data = skeleton_loader(skeleton_file_dir)
    # Map 25 nodes to 20
    if (("kb" in skeleton_data.keys()) and ("kb2" not in skeleton_data.keys())):# single people action
        kb_data_20 = map_25_to_20(skeleton_data, "kb")
        pass
    elif("kb2" in skeleton_data.keys() and ("kb" in skeleton_data.keys())):# Double action, there is a virtual person
        kb1_data_20 = map_25_to_20(skeleton_data, "kb")
        # draw_skeleton_joint_20(5, kb_data_20)
        kb2_data_20 = map_25_to_20(skeleton_data, "kb2")
        # draw_skeleton_joint_20(5, kb2_data_20)
        kb_data_20 = np.array((kb1_data_20 + kb2_data_20)/2)
        # draw_3_skeleton_joint_20(10,kb1_data_20,kb_data_20, kb2_data_20)
        pass
    else:
        print("{:s} was broken!".format(skeleton_file_dir))
        raise AssertionError
        pass

    # kb_data_20 = skeleton_data["kb"]
    # draw the skeleton
    # draw_skeleton_joint_20(5,kb_data_20)
    # plt.show()
    # raise RuntimeError
    skel_array = np.zeros((38, 3, 30, len(file_idx)))#F x C x H x W
    vector_a = np.zeros((len(combinations_3_20), len(file_idx), 3))
    vector_b = np.zeros((len(combinations_3_20), len(file_idx), 3))
    vector_c = np.zeros((len(combinations_3_20), len(file_idx), 3))
    dst_array = np.array(kb_data_20[file_idx,:])

    # Subtract the data mean
    # dst_array = (dst_array - 0.844379) / 1.552702 # ntu_grb+d
    # dst_array = (dst_array - 0.809465) / 1.529892 # cas_mhad
    # dst_array = (dst_array - 0.831969) / 1.475146  # utd_mhad
    dst_array = (dst_array - 0.707839) / 1.156303  # utd_mvhad

    # print(dst_array[:,[0,1,2]]-dst_array[:,[3,4,5]])
    # print((dst_array[:,[0,1,2]]-dst_array[:,[3,4,5]]).shape)
    # raise RuntimeError
    copy_dst_array = np.tile(dst_array,(len(combinations_3_20),1))
    copy_dst_array = copy_dst_array.reshape([len(combinations_3_20), len(file_idx), 60])  # 1140*16*60
    # print(np.array(copy_dst_array[2,:,[0,1,2]]-copy_dst_array[0,:,[3,4,5]]).T)
    # print(np.array(copy_dst_array[2,:,[0,1,2]]-copy_dst_array[0,:,[3,4,5]]).T.shape)
    # raise RuntimeError
    for i,ICount in enumerate(combinations_3_20):
        vector_a[i,:,:] = np.array(copy_dst_array[i, :, [3*ICount[0],3*ICount[0]+1,3*ICount[0]+2]] - copy_dst_array[i, :, [3*ICount[1],3*ICount[1]+1,3*ICount[1]+2]]).T
        vector_b[i,:,:] = np.array(copy_dst_array[i, :, [3*ICount[0],3*ICount[0]+1,3*ICount[0]+2]] - copy_dst_array[i, :, [3*ICount[2],3*ICount[2]+1,3*ICount[2]+2]]).T
        vector_c[i,:,:] = np.array(copy_dst_array[i, :, [3*ICount[1],3*ICount[1]+1,3*ICount[1]+2]] - copy_dst_array[i, :, [3*ICount[2],3*ICount[2]+1,3*ICount[2]+2]]).T
        # raise RuntimeError
        pass
    # get the norm of the vector
    norm_vector_ab = linalg.norm(vector_a, axis=2) * linalg.norm(vector_b, axis=2)
    norm_vector_ac = linalg.norm(vector_a, axis=2) * linalg.norm(vector_c, axis=2)
    norm_vector_bc = linalg.norm(vector_b, axis=2) * linalg.norm(vector_c, axis=2)
    # Get matrix dot product
    product_vector_ab = ((vector_a[:, :, 0] * vector_b[:, :, 0]) + (vector_a[:, :, 1] * vector_b[:, :, 1]) + (vector_a[:, :, 2] * vector_b[:, :, 2]))
    product_vector_ac = ((vector_a[:, :, 0] * vector_c[:, :, 0]) + (vector_a[:, :, 1] * vector_c[:, :, 1]) + (vector_a[:, :, 2] * vector_c[:, :, 2]))
    product_vector_bc = ((vector_b[:, :, 0] * vector_c[:, :, 0]) + (vector_b[:, :, 1] * vector_c[:, :, 1]) + (vector_b[:, :, 2] * vector_c[:, :, 2]))

    # Prevent the denominator from being 0
    norm_vector_ab[norm_vector_ab == 0] = 1
    norm_vector_ac[norm_vector_ac == 0] = 1
    norm_vector_bc[norm_vector_bc == 0] = 1
    product_vector_ab[norm_vector_ab == 0] = 0
    product_vector_ac[norm_vector_ac == 0] = 0
    product_vector_bc[norm_vector_bc == 0] = 0

    # Get the cos theta of the vector
    skel_array[:, 0, :, :] = (product_vector_ab / norm_vector_ab).reshape(38, 30, len(file_idx))
    skel_array[:, 1, :, :] = (product_vector_ac / norm_vector_ac).reshape(38, 30, len(file_idx))
    skel_array[:, 2, :, :] = (product_vector_bc / norm_vector_bc).reshape(38, 30, len(file_idx))
    # skel_array[:, 0, :, :] = preprocessing.normalize((product_vector_ab / norm_vector_ab).T, norm='l1').reshape(len(file_idx), 30, 38)
    # skel_array[:, 1, :, :] = preprocessing.normalize((product_vector_ac / norm_vector_ac).T, norm='l1').reshape(len(file_idx), 30, 38)
    # skel_array[:, 2, :, :] = preprocessing.normalize((product_vector_bc / norm_vector_bc).T, norm='l1').reshape(len(file_idx), 30, 38)

    # temp_skel_array = np.array([(product_vector_ab / norm_vector_ab), (product_vector_ac / norm_vector_ac), (product_vector_bc / norm_vector_bc)]).reshape([len(file_idx),len(combinations_3_20), 3])
    # skel_array = temp_skel_array.reshape([len(file_idx), 3, 30, 38])
    # print(skel_array)
    # print(skel_array.shape)
    return skel_array # F x C x H x W
    pass


# @staticmethod
def read_ntu_skeleton3p_cos_3x16x1140x1(skeleton_file_dir, skeleton_count, skeleton_loader=None, file_idx=None, combinations_3_20 = None):

    # if file_idx is not None:
    #     frame_list = file_idx
    # else:
    #     frame_list = list(range(skeleton_count))

    assert os.path.exists(skeleton_file_dir), "{:s} does not exist!".format(skeleton_file_dir)
    skeleton_data = skeleton_loader(skeleton_file_dir)
    # Map 25 nodes to 20
    if (("kb" in skeleton_data.keys()) and ("kb2" not in skeleton_data.keys())):# single people action
        kb_data_20 = map_25_to_20(skeleton_data, "kb")
        pass
    elif("kb2" in skeleton_data.keys() and ("kb" in skeleton_data.keys())):# Double action, there is a virtual person
        # print(skeleton_data["kb2"].shape)
        # print(skeleton_data["kb"].shape)
        # raise RuntimeError
        if skeleton_data["kb2"].shape[0] == skeleton_data["kb"].shape[0]:
            kb1_data_20 = map_25_to_20(skeleton_data, "kb")
            # draw_skeleton_joint_20(5, kb_data_20)
            kb2_data_20 = map_25_to_20(skeleton_data, "kb2")

            # draw_skeleton_joint_20(5, kb2_data_20)
            kb_data_20 = np.array((kb1_data_20 + kb2_data_20) / 2)
            # draw_3_skeleton_joint_20(10,kb1_data_20,kb_data_20, kb2_data_20)
            pass
        else:
            kb_data_20 = map_25_to_20(skeleton_data, "kb")
            pass
        pass
    else:
        print("{:s} was broken!".format(skeleton_file_dir))
        raise AssertionError
        pass

    # kb_data_20 = skeleton_data["kb"]
    # draw the skeleton
    # draw_skeleton_joint_20(5,kb_data_20)
    # plt.show()
    # raise RuntimeError
    # print(file_idx)
    # raise RuntimeError
    skel_array = np.zeros((len(file_idx), 3, len(combinations_3_20), 1))  #F x C x H x W
    vector_a = np.zeros((len(combinations_3_20), len(file_idx), 3))
    vector_b = np.zeros((len(combinations_3_20), len(file_idx), 3))
    vector_c = np.zeros((len(combinations_3_20), len(file_idx), 3))
    dst_array = np.array(kb_data_20[file_idx, :])

    # Subtract the data mean
    # dst_array = (dst_array - 0.844379) / 1.552702 # ntu_grb+d
    dst_array = (dst_array - 0.809465) / 1.529892 # cas_mhad
    # dst_array = (dst_array - 0.831969) / 1.475146  # utd_mhad
    # dst_array = (dst_array - 0.707839) / 1.156303  # utd_mvhad

    # print(dst_array[:,[0,1,2]]-dst_array[:,[3,4,5]])
    # print((dst_array[:,[0,1,2]]-dst_array[:,[3,4,5]]).shape)
    # raise RuntimeError
    copy_dst_array = np.tile(dst_array, (len(combinations_3_20), 1))
    copy_dst_array = copy_dst_array.reshape([len(combinations_3_20), len(file_idx), 60])  # 1140*16*60
    # print(np.array(copy_dst_array[2,:,[0,1,2]]-copy_dst_array[0,:,[3,4,5]]).T)
    # print(np.array(copy_dst_array[2,:,[0,1,2]]-copy_dst_array[0,:,[3,4,5]]).T.shape)
    # raise RuntimeError
    for i,ICount in enumerate(combinations_3_20):
        vector_a[i,:,:] = np.array(copy_dst_array[i, :, [3*ICount[0],3*ICount[0]+1,3*ICount[0]+2]] - copy_dst_array[i, :, [3*ICount[1],3*ICount[1]+1,3*ICount[1]+2]]).T
        vector_b[i,:,:] = np.array(copy_dst_array[i, :, [3*ICount[0],3*ICount[0]+1,3*ICount[0]+2]] - copy_dst_array[i, :, [3*ICount[2],3*ICount[2]+1,3*ICount[2]+2]]).T
        vector_c[i,:,:] = np.array(copy_dst_array[i, :, [3*ICount[1],3*ICount[1]+1,3*ICount[1]+2]] - copy_dst_array[i, :, [3*ICount[2],3*ICount[2]+1,3*ICount[2]+2]]).T
        # raise RuntimeError
        pass
    # get the norm of the vector
    norm_vector_ab = linalg.norm(vector_a, axis=2) * linalg.norm(vector_b, axis=2)
    norm_vector_ac = linalg.norm(vector_a, axis=2) * linalg.norm(vector_c, axis=2)
    norm_vector_bc = linalg.norm(vector_b, axis=2) * linalg.norm(vector_c, axis=2)
    # Get matrix dot product
    product_vector_ab = ((vector_a[:, :, 0] * vector_b[:, :, 0]) + (vector_a[:, :, 1] * vector_b[:, :, 1]) + (vector_a[:, :, 2] * vector_b[:, :, 2]))
    product_vector_ac = ((vector_a[:, :, 0] * vector_c[:, :, 0]) + (vector_a[:, :, 1] * vector_c[:, :, 1]) + (vector_a[:, :, 2] * vector_c[:, :, 2]))
    product_vector_bc = ((vector_b[:, :, 0] * vector_c[:, :, 0]) + (vector_b[:, :, 1] * vector_c[:, :, 1]) + (vector_b[:, :, 2] * vector_c[:, :, 2]))

    # Prevent the denominator from being 0
    norm_vector_ab[norm_vector_ab == 0] = 1
    norm_vector_ac[norm_vector_ac == 0] = 1
    norm_vector_bc[norm_vector_bc == 0] = 1
    product_vector_ab[norm_vector_ab == 0] = 0
    product_vector_ac[norm_vector_ac == 0] = 0
    product_vector_bc[norm_vector_bc == 0] = 0

    # Get the cos theta of the vector
    # skel_array[:, 0, :, :] = (product_vector_ab / norm_vector_ab).T.reshape(len(file_idx), len(combinations_3_20), 1)# 16 x 1140 x 1
    # skel_array[:, 1, :, :] = (product_vector_ac / norm_vector_ac).T.reshape(len(file_idx), len(combinations_3_20), 1)
    # skel_array[:, 2, :, :] = (product_vector_bc / norm_vector_bc).T.reshape(len(file_idx), len(combinations_3_20), 1)
    skel_array[:, 0, :, :] = preprocessing.normalize((product_vector_ab / norm_vector_ab).T, norm='l2').reshape(len(file_idx), len(combinations_3_20), 1)
    skel_array[:, 1, :, :] = preprocessing.normalize((product_vector_ac / norm_vector_ac).T, norm='l2').reshape(len(file_idx), len(combinations_3_20), 1)
    skel_array[:, 2, :, :] = preprocessing.normalize((product_vector_bc / norm_vector_bc).T, norm='l2').reshape(len(file_idx), len(combinations_3_20), 1)

    # temp_skel_array = np.array([(product_vector_ab / norm_vector_ab), (product_vector_ac / norm_vector_ac), (product_vector_bc / norm_vector_bc)]).reshape([len(file_idx),len(combinations_3_20), 3])
    # skel_array = temp_skel_array.reshape([len(file_idx), 3, 30, 38])
    # print(skel_array)
    # print(skel_array.shape)
    # raise RuntimeError
    return skel_array  # F x C x H x W
    pass
# --todo--
def read_ntu_skeleton_convert_axis_2p_cos(skeleton_file_dir, skeleton_count, skeleton_loader=None, file_idx=None, combinations_2_20 = None):

    # if file_idx is not None:
    #     frame_list = file_idx
    # else:
    #     frame_list = list(range(skeleton_count))

    assert os.path.exists(skeleton_file_dir), "{:s} does not exist!".format(skeleton_file_dir)
    skeleton_data = skeleton_loader(skeleton_file_dir)
    # Map 25 nodes to 20
    if (("kb" in skeleton_data.keys()) and ("kb2" not in skeleton_data.keys())):# single people action
        kb_data_20 = map_25_to_20(skeleton_data, "kb")
        pass
    elif("kb2" in skeleton_data.keys() and ("kb" in skeleton_data.keys())):# Double action, there is a virtual person
        kb1_data_20 = map_25_to_20(skeleton_data, "kb")
        # draw_skeleton_joint_20(5, kb_data_20)
        kb2_data_20 = map_25_to_20(skeleton_data, "kb2")
        # draw_skeleton_joint_20(5, kb2_data_20)
        kb_data_20 = np.array((kb1_data_20 + kb2_data_20)/2)
        # draw_3_skeleton_joint_20(10,kb1_data_20,kb_data_20, kb2_data_20)
        pass
    else:
        print("{:s} was broken!".format(skeleton_file_dir))
        raise AssertionError
        pass
    # kb_data_20 = skeleton_data["kb"]


    dst_array = np.array(kb_data_20[file_idx, :])

    # Subtract the data mean
    # dst_array = (dst_array - 0.844379) / 1.552702 # ntu_grb+d
    # dst_array = (dst_array - 0.809465) / 1.529892 # cas_mhad
    # dst_array = (dst_array - 0.831969) / 1.475146  # utd_mhad
    dst_array = (dst_array - 0.707839) / 1.156303  # utd_mvhad

    seq_len = dst_array.shape[1] // 3

    axis_x = dst_array[:, list(map(lambda m: 3 * m, range(seq_len)))] # 16 x 20
    axis_y = dst_array[:, list(map(lambda m: 3 * m + 1, range(seq_len)))]
    axis_z = dst_array[:, list(map(lambda m: 3 * m + 2, range(seq_len)))]

    convert_axis_x = np.zeros((len(file_idx), seq_len))
    convert_axis_y = np.zeros((len(file_idx), seq_len))
    convert_axis_z = np.zeros((len(file_idx), seq_len))

    for si in range(seq_len):
        convert_axis_x[:, si] = axis_x[:, si] - axis_x[:, 0]
        convert_axis_y[:, si] = axis_y[:, si] - axis_y[:, 0]
        convert_axis_z[:, si] = axis_z[:, si] - axis_z[:, 0]
        pass

    dst_array[:, list(map(lambda m: 3 * m, range(seq_len)))] = convert_axis_x
    dst_array[:, list(map(lambda m: 3 * m + 1, range(seq_len)))] = convert_axis_y
    dst_array[:, list(map(lambda m: 3 * m + 2, range(seq_len)))] = convert_axis_z

    skel_array = np.zeros((len(file_idx), 3, len(combinations_2_20), 1))  # F x C x H x W

    vector_a = np.zeros((len(combinations_2_20), len(file_idx), 3))
    vector_b = np.zeros((len(combinations_2_20), len(file_idx), 3))

    # print(dst_array[:,[0,1,2]]-dst_array[:,[3,4,5]])
    # print((dst_array[:,[0,1,2]]-dst_array[:,[3,4,5]]).shape)
    # raise RuntimeError
    copy_dst_array = np.tile(dst_array,(len(combinations_2_20),1))
    copy_dst_array = copy_dst_array.reshape([len(combinations_2_20), len(file_idx), 60])  # 190*16*60
    # print(np.array(copy_dst_array[2,:,[0,1,2]]-copy_dst_array[0,:,[3,4,5]]).T)
    # print(np.array(copy_dst_array[2,:,[0,1,2]]-copy_dst_array[0,:,[3,4,5]]).T.shape)
    # raise RuntimeError
    for i,ICount in enumerate(combinations_2_20):
        vector_a[i,:,:] = np.array(copy_dst_array[i, :, [3*ICount[0],3*ICount[0]+1,3*ICount[0]+2]]).T
        vector_b[i,:,:] = np.array(copy_dst_array[i, :, [3*ICount[1],3*ICount[1]+1,3*ICount[1]+2]]).T
        # print(vector_a[i,:,:].shape)
        # print(np.array(copy_dst_array[i, :, [3*ICount[0],3*ICount[0]+1,3*ICount[0]+2]]).T.shape)
        # raise RuntimeError
        pass
    # get the norm of the vector
    norm_vector_ab = linalg.norm(vector_a, axis=2) * linalg.norm(vector_b, axis=2)
    # Get matrix dot product
    product_vector_ab = ((vector_a[:, :, 0] * vector_b[:, :, 0]) + (vector_a[:, :, 1] * vector_b[:, :, 1]) + (vector_a[:, :, 2] * vector_b[:, :, 2]))

    # Prevent the denominator from being 0
    norm_vector_ab[norm_vector_ab == 0] = 1
    product_vector_ab[norm_vector_ab == 0] = 0

    # Get the cos theta of the vector
    skel_array[:, 0, :, :] = (product_vector_ab / norm_vector_ab).T.reshape(len(file_idx), len(combinations_2_20), 1)  # 16 x 190 x 1
    skel_array[:, 1, :, :] = (product_vector_ab / norm_vector_ab).T.reshape(len(file_idx), len(combinations_2_20), 1)
    skel_array[:, 2, :, :] = (product_vector_ab / norm_vector_ab).T.reshape(len(file_idx), len(combinations_2_20), 1)
    # temp_skel_array = np.array([(product_vector_ab / norm_vector_ab), (product_vector_ac / norm_vector_ac), (product_vector_bc / norm_vector_bc)]).reshape([len(file_idx),len(combinations_3_20), 3])
    # skel_array = temp_skel_array.reshape([len(file_idx), 3, 30, 38])
    # print(skel_array)
    # print(skel_array.shape)
    return skel_array # F x C x H x W
    pass

def read_ntu_skeleton_convert_axis_normal_vector_1140x1(skeleton_file_dir, skeleton_count, skeleton_loader=None, file_idx=None, combinations_3_20 = None):

    # if file_idx is not None:
    #     frame_list = file_idx
    # else:
    #     frame_list = list(range(skeleton_count))

    assert os.path.exists(skeleton_file_dir), "{:s} does not exist!".format(skeleton_file_dir)
    skeleton_data = skeleton_loader(skeleton_file_dir)

    # todo
    # Map 25 nodes to 20
    if (("kb" in skeleton_data.keys()) and ("kb2" not in skeleton_data.keys())):# single people action
        kb_data_20 = map_25_to_20(skeleton_data, "kb")
        pass
    elif("kb2" in skeleton_data.keys() and ("kb" in skeleton_data.keys())):# Double action, there is a virtual person
        kb1_data_20 = map_25_to_20(skeleton_data, "kb")
        # draw_skeleton_joint_20(5, kb_data_20)
        kb2_data_20 = map_25_to_20(skeleton_data, "kb2")
        # draw_skeleton_joint_20(5, kb2_data_20)
        kb_data_20 = np.array((kb1_data_20 + kb2_data_20)/2)
        # draw_3_skeleton_joint_20(10,kb1_data_20,kb_data_20, kb2_data_20)
        pass
    else:
        print("{:s} was broken!".format(skeleton_file_dir))
        raise AssertionError
        pass

    # todo
    # kb_data_20 = skeleton_data["kb"]

    # draw the skeleton
    # draw_skeleton_joint_20(5,kb_data_20)
    # plt.show()
    # raise RuntimeError
    skel_array = np.zeros((len(file_idx), 3, len(combinations_3_20), 1))#F x C x H x W
    vector_a = np.zeros((len(combinations_3_20), len(file_idx), 3))
    vector_b = np.zeros((len(combinations_3_20), len(file_idx), 3))
    vector_c = np.zeros((len(combinations_3_20), len(file_idx), 3))
    dst_array = np.array(kb_data_20[file_idx,:])

    # dst_array = (dst_array - 0.844379) / 1.552702 # ntu_grb+d
    dst_array = (dst_array - 0.809465) / 1.529892 # cas_mhad
    # dst_array = (dst_array - 0.831969) / 1.475146  # utd_mhad
    # dst_array = (dst_array - 0.707839) / 1.156303  # utd_mvhad

    seq_len = dst_array.shape[1] // 3

    axis_x = dst_array[:, list(map(lambda m: 3 * m, range(seq_len)))]  # 16 x 20
    axis_y = dst_array[:, list(map(lambda m: 3 * m + 1, range(seq_len)))]
    axis_z = dst_array[:, list(map(lambda m: 3 * m + 2, range(seq_len)))]

    convert_axis_x = np.zeros((len(file_idx), seq_len))
    convert_axis_y = np.zeros((len(file_idx), seq_len))
    convert_axis_z = np.zeros((len(file_idx), seq_len))

    for si in range(seq_len):
        convert_axis_x[:, si] = axis_x[:, si] - axis_x[:, 0]
        convert_axis_y[:, si] = axis_y[:, si] - axis_y[:, 0]
        convert_axis_z[:, si] = axis_z[:, si] - axis_z[:, 0]
        pass

    dst_array[:, list(map(lambda m: 3 * m, range(seq_len)))] = convert_axis_x
    dst_array[:, list(map(lambda m: 3 * m + 1, range(seq_len)))] = convert_axis_y
    dst_array[:, list(map(lambda m: 3 * m + 2, range(seq_len)))] = convert_axis_z
    # print(dst_array[:,[0,1,2]]-dst_array[:,[3,4,5]])
    # print((dst_array[:,[0,1,2]]-dst_array[:,[3,4,5]]).shape)
    # raise RuntimeError
    copy_dst_array = np.tile(dst_array,(len(combinations_3_20),1))
    copy_dst_array = copy_dst_array.reshape([len(combinations_3_20), len(file_idx), 60])  # 1140*16*60
    # print(np.array(copy_dst_array[2,:,[0,1,2]]-copy_dst_array[0,:,[3,4,5]]).T)
    # print(np.array(copy_dst_array[2,:,[0,1,2]]-copy_dst_array[0,:,[3,4,5]]).T.shape)
    # raise RuntimeError
    for i,ICount in enumerate(combinations_3_20):
        vector_a[i,:,:] = np.array(copy_dst_array[i, :, [3*ICount[0],3*ICount[0]+1,3*ICount[0]+2]] - copy_dst_array[i, :, [3*ICount[1],3*ICount[1]+1,3*ICount[1]+2]]).T
        vector_b[i,:,:] = np.array(copy_dst_array[i, :, [3*ICount[2],3*ICount[2]+1,3*ICount[2]+2]] - copy_dst_array[i, :, [3*ICount[1],3*ICount[1]+1,3*ICount[1]+2]]).T
        # raise RuntimeError
        pass

    # get the cross product is a vector normal to the plane
    cross_product = np.cross(vector_a,vector_b,axis=2)

    skel_array[:, 0, :, :] = preprocessing.normalize(cross_product[:, :, 0], norm='l2').T.reshape(len(file_idx), len(combinations_3_20), 1)
    skel_array[:, 1, :, :] = preprocessing.normalize(cross_product[:, :, 1], norm='l2').T.reshape(len(file_idx), len(combinations_3_20), 1)
    skel_array[:, 2, :, :] = preprocessing.normalize(cross_product[:, :, 2], norm='l2').T.reshape(len(file_idx), len(combinations_3_20), 1)
    # print(skel_array[0,:,:,:])
    # print(cross_product[:, :, 0].T.shape)
    # raise RuntimeError

    return skel_array # F x C x H x W
    pass

def read_ntu_skeleton_angular_velocity(skeleton_file_dir, skeleton_count, skeleton_loader=None, file_idx=None, combinations_3_20 = None):

    assert os.path.exists(skeleton_file_dir), "{:s} does not exist!".format(skeleton_file_dir)
    skeleton_data = skeleton_loader(skeleton_file_dir)
    # Map 25 nodes to 20
    if (("kb" in skeleton_data.keys()) and ("kb2" not in skeleton_data.keys())):# single people action
        kb_data_20 = map_25_to_20(skeleton_data, "kb")
        pass
    elif("kb2" in skeleton_data.keys() and ("kb" in skeleton_data.keys())):# Double action, there is a virtual person
        kb1_data_20 = map_25_to_20(skeleton_data, "kb")
        # draw_skeleton_joint_20(5, kb_data_20)
        kb2_data_20 = map_25_to_20(skeleton_data, "kb2")
        # draw_skeleton_joint_20(5, kb2_data_20)
        kb_data_20 = np.array((kb1_data_20 + kb2_data_20)/2)
        # draw_3_skeleton_joint_20(10,kb1_data_20,kb_data_20, kb2_data_20)
        pass
    else:
        print("{:s} was broken!".format(skeleton_file_dir))
        raise AssertionError
        pass

    # kb_data_20 = skeleton_data["kb"]
    # draw the skeleton
    # draw_skeleton_joint_20(5,kb_data_20)
    # plt.show()
    # raise RuntimeError
    skel_array = np.zeros((len(file_idx)//2, 3, 30, 38))  # F x C x H x W
    # print(file_idx,  skel_array.shape)
    # raise RuntimeError
    vector_a = np.zeros((len(combinations_3_20), len(file_idx), 3))
    vector_b = np.zeros((len(combinations_3_20), len(file_idx), 3))
    vector_c = np.zeros((len(combinations_3_20), len(file_idx), 3))
    dst_array = np.array(kb_data_20[file_idx, :])
    # Subtract the data mean

    # dst_array = (dst_array - 0.844379) / 1.552702 # ntu_grb+d
    dst_array = (dst_array - 0.809465) / 1.529892 # cas_mhad
    # dst_array = (dst_array - 0.831969) / 1.475146  # utd_mhad

    # print(dst_array[:,[0,1,2]]-dst_array[:,[3,4,5]])
    # print((dst_array[:,[0,1,2]]-dst_array[:,[3,4,5]]).shape)
    # raise RuntimeError
    copy_dst_array = np.tile(dst_array, (len(combinations_3_20), 1))
    copy_dst_array = copy_dst_array.reshape([len(combinations_3_20), len(file_idx), 60])  # 1140*16*60
    # print(np.array(copy_dst_array[2,:,[0,1,2]]-copy_dst_array[0,:,[3,4,5]]).T)
    # print(np.array(copy_dst_array[2,:,[0,1,2]]-copy_dst_array[0,:,[3,4,5]]).T.shape)
    # raise RuntimeError
    for i, ICount in enumerate(combinations_3_20):
        vector_a[i, :, :] = np.array(copy_dst_array[i, :, [3 * ICount[0], 3 * ICount[0] + 1, 3 * ICount[0] + 2]] - copy_dst_array[i, :, [3 * ICount[1], 3 * ICount[1] + 1, 3 * ICount[1] + 2]]).T
        vector_b[i, :, :] = np.array(copy_dst_array[i, :, [3 * ICount[0], 3 * ICount[0] + 1, 3 * ICount[0] + 2]] - copy_dst_array[i, :, [3 * ICount[2], 3 * ICount[2] + 1, 3 * ICount[2] + 2]]).T
        vector_c[i, :, :] = np.array(copy_dst_array[i, :, [3 * ICount[1], 3 * ICount[1] + 1, 3 * ICount[1] + 2]] - copy_dst_array[i, :, [3 * ICount[2], 3 * ICount[2] + 1, 3 * ICount[2] + 2]]).T
        # raise RuntimeError
        pass
    # get the norm of the vector
    norm_vector_ab = linalg.norm(vector_a, axis=2) * linalg.norm(vector_b, axis=2)
    norm_vector_ac = linalg.norm(vector_a, axis=2) * linalg.norm(vector_c, axis=2)
    norm_vector_bc = linalg.norm(vector_b, axis=2) * linalg.norm(vector_c, axis=2)
    # Get matrix dot product
    product_vector_ab = ((vector_a[:, :, 0] * vector_b[:, :, 0]) + (vector_a[:, :, 1] * vector_b[:, :, 1]) + (vector_a[:, :, 2] * vector_b[:, :, 2]))
    product_vector_ac = ((vector_a[:, :, 0] * vector_c[:, :, 0]) + (vector_a[:, :, 1] * vector_c[:, :, 1]) + (vector_a[:, :, 2] * vector_c[:, :, 2]))
    product_vector_bc = ((vector_b[:, :, 0] * vector_c[:, :, 0]) + (vector_b[:, :, 1] * vector_c[:, :, 1]) + (vector_b[:, :, 2] * vector_c[:, :, 2]))

    # Prevent the denominator from being 0
    norm_vector_ab[norm_vector_ab == 0] = 1
    norm_vector_ac[norm_vector_ac == 0] = 1
    norm_vector_bc[norm_vector_bc == 0] = 1
    product_vector_ab[norm_vector_ab == 0] = 0
    product_vector_ac[norm_vector_ac == 0] = 0
    product_vector_bc[norm_vector_bc == 0] = 0

    # Get the cos theta of the vector
    # skel_array[:, 0, :, :] = (product_vector_ab / norm_vector_ab).T.reshape(len(file_idx), 30, 38)
    # skel_array[:, 1, :, :] = (product_vector_ac / norm_vector_ac).T.reshape(len(file_idx), 30, 38)
    # skel_array[:, 2, :, :] = (product_vector_bc / norm_vector_bc).T.reshape(len(file_idx), 30, 38)
    # skel_array[:, 0, :, :] = preprocessing.normalize((product_vector_ab / norm_vector_ab).T, norm='l1').reshape(len(file_idx), 30, 38)
    # skel_array[:, 1, :, :] = preprocessing.normalize((product_vector_ac / norm_vector_ac).T, norm='l1').reshape(len(file_idx), 30, 38)
    # skel_array[:, 2, :, :] = preprocessing.normalize((product_vector_bc / norm_vector_bc).T, norm='l1').reshape(len(file_idx), 30, 38)

    # print(skel_array)
    # print(skel_array.shape)

    cos_ab_theta = (product_vector_ab / norm_vector_ab) # len(combinations_3_20) * len(file_idx) 1140 x 32
    cos_ac_theta = (product_vector_ac / norm_vector_ac) # len(combinations_3_20) * len(file_idx)
    cos_bc_theta = (product_vector_bc / norm_vector_bc) # len(combinations_3_20) * len(file_idx)

    # print(cos_ab_theta)
    # print(cos_ab_theta.shape)

    seqs_len = (len(file_idx) // 2) # 16

    d_cos_ab = np.zeros((len(combinations_3_20), seqs_len))
    d_cos_ac = np.zeros((len(combinations_3_20), seqs_len))
    d_cos_bc = np.zeros((len(combinations_3_20), seqs_len))

    for si in range(seqs_len):
        seq_ab_b = np.asarray(cos_ab_theta[:, si * 2]).astype(np.float)
        seq_ab_n = np.asarray(cos_ab_theta[:, si * 2 + 1]).astype(np.float)

        seq_ac_b = np.asarray(cos_ac_theta[:, si * 2]).astype(np.float)
        seq_ac_n = np.asarray(cos_ac_theta[:, si * 2 + 1]).astype(np.float)

        seq_bc_b = np.asarray(cos_bc_theta[:, si * 2]).astype(np.float)
        seq_bc_n = np.asarray(cos_bc_theta[:, si * 2 + 1]).astype(np.float)
        # [dy, dx] = np.gradient(img_c)

        d_cos_ab[:, si] = (seq_ab_n - seq_ab_b) / 2.0
        d_cos_ac[:, si] = (seq_ac_n - seq_ac_b) / 2.0
        d_cos_bc[:, si] = (seq_bc_n - seq_bc_b) / 2.0

        pass

    # print(d_cos_ab)
    # print(d_cos_ab.shape)
    # raise RuntimeError

    skel_array[:, 0, :, :] = preprocessing.normalize(d_cos_ab).T.reshape(seqs_len, 30, 38)
    skel_array[:, 1, :, :] = preprocessing.normalize(d_cos_ac).T.reshape(seqs_len, 30, 38)
    skel_array[:, 2, :, :] = preprocessing.normalize(d_cos_bc).T.reshape(seqs_len, 30, 38)

    return skel_array # F x C x H x W
    pass


# @staticmethod
def read_skeleton_momentum(skeleton_file_dir, skeleton_count, skeleton_loader=None, file_idx=None):

    spherical_coordinate_system = False
    assert os.path.exists(skeleton_file_dir), "{:s} does not exist!".format(skeleton_file_dir)
    skeleton_data = skeleton_loader(skeleton_file_dir)
    # Map 25 nodes to 20
    if (("kb" in skeleton_data.keys()) and ("kb2" not in skeleton_data.keys())):# single people action
        kb_data_20 = map_25_to_20(skeleton_data, "kb")
        pass
    elif("kb2" in skeleton_data.keys() and ("kb" in skeleton_data.keys())):# Double action, there is a virtual person
        kb1_data_20 = map_25_to_20(skeleton_data, "kb")
        # draw_skeleton_joint_20(5, kb_data_20)
        kb2_data_20 = map_25_to_20(skeleton_data, "kb2")
        # draw_skeleton_joint_20(5, kb2_data_20)
        kb_data_20 = np.array((kb1_data_20 + kb2_data_20)/2)
        # draw_3_skeleton_joint_20(10,kb1_data_20,kb_data_20, kb2_data_20)
        pass
    else:
        print("{:s} was broken!".format(skeleton_file_dir))
        raise AssertionError
        pass

    # kb_data_20 = skeleton_data["kb"]
    # draw the skeleton
    # draw_skeleton_joint_20(5,kb_data_20)
    # plt.show()
    # raise RuntimeError
    # print(kb_data_20.shape)


    skel_array = np.zeros((len(file_idx), 3, 20, 1))  # F x C x H x W
    dst_array_1 = np.array(kb_data_20[file_idx, :])
    dst_array_2 = np.array(kb_data_20[[fi+1 for fi in file_idx], :])
    dst_array_1 = (dst_array_1 - 0.809465) / 1.529892  # cas_mhad
    dst_array_2 = (dst_array_2 - 0.809465) / 1.529892  # cas_mhad
    dst_array = np.array(dst_array_2 - dst_array_1)   # 16 x 60
    # print("dst_array", dst_array_1, "\n\n", dst_array_2, "\n\n", dst_array)
    # dst_array = (dst_array - 0.844379) / 1.552702  # ntu_grb+d
    # dst_array = (dst_array - 0.809465) / 1.529892  # cas_mhad
    # dst_array = (dst_array - 0.831969) / 1.475146  # utd_mhad

    axis_x = dst_array[:, list(map(lambda m: 3 * m, range(20)))]  # 16 x 20
    axis_y = dst_array[:, list(map(lambda m: 3 * m + 1, range(20)))]  # 16 x 20
    axis_z = dst_array[:, list(map(lambda m: 3 * m + 2, range(20)))]  # 16 x 20

    if spherical_coordinate_system:

        r = np.zeros((16, 20))  # 16 x 20
        theta = np.zeros((16, 20))  # 16 x 20
        phi = np.zeros((16, 20))  # 16 x 20

        for i in range(20):
            r[:, i] = np.linalg.norm([axis_x[:, i], axis_y[:, i], axis_z[:, i]], axis=0)
            temp_z = axis_z[:, i]
            temp_r = r[:, i]
            temp_z[temp_r == 0] = 0
            temp_r[temp_r == 0] = 1
            theta[:, i] = np.arccos(temp_z / temp_r)
            temp_y = axis_y[:, i]
            temp_x = axis_x[:, i]
            temp_y[axis_x[:, i] == 0] = 0
            temp_x[axis_x[:, i] == 0] = 1
            phi[:, i] = np.arctan(temp_y / temp_x)
            # print(r.shape, theta.shape, phi.shape)
            # raise RuntimeError
            pass

        skel_array[:, 0, :, :] = preprocessing.normalize(r, norm='l2').reshape(len(file_idx), 20, 1)
        skel_array[:, 1, :, :] = preprocessing.normalize(theta, norm='l2').reshape(len(file_idx), 20, 1)
        skel_array[:, 2, :, :] = preprocessing.normalize(phi, norm='l2').reshape(len(file_idx), 20, 1)
        pass
    else:
        skel_array[:, 0, :, :] = preprocessing.normalize(axis_x, norm='l2').reshape(len(file_idx), 20, 1)
        skel_array[:, 1, :, :] = preprocessing.normalize(axis_y, norm='l2').reshape(len(file_idx), 20, 1)
        skel_array[:, 2, :, :] = preprocessing.normalize(axis_z, norm='l2').reshape(len(file_idx), 20, 1)
        pass
    # print("skel_array", skel_array.shape)
    return skel_array  # F x C x H x W
    pass


def fusion_momentum_and_pojm3d(skeleton_file_dir, skeleton_count, skeleton_loader=None, file_idx=None, combinations_3_20 = None):
    fusion_tensor = []
    cos_3xFx1140x1 = read_ntu_skeleton3p_cos_3x16x1140x1(skeleton_file_dir, skeleton_count, skeleton_loader, file_idx, combinations_3_20)
    fusion_tensor.append(cos_3xFx1140x1)
    momentum_3xFx20x1 = read_skeleton_momentum(skeleton_file_dir, skeleton_count, skeleton_loader, file_idx)
    fusion_tensor.append(momentum_3xFx20x1)
    return fusion_tensor
    pass

if __name__ == '__main__':

    combination = create_combination(3, 20)
    # hoj3d_file_dir = "/home/hly/zqs/datasets/hoj3d/cas_mhad/skeleton/S001C001T001A001.mat"
    skel_dirs = "/data/zqs/skeleton_datasets_36/datasets/ntu_rgb+d/skeleton/S006C001P015R002A054.mat"
    # os.environ["DISPLAY"] = "localhost:13.0"
    # skeleton_data = read_ntu_skeleton3p_cos_3x16x1140x1(skel_dirs, 71, skeleton_loader=mat_load, file_idx=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], combinations_3_20=combination)
    # import matplotlib

    # matplotlib.use("TkAgg")
    # skeleton_data = mat_load(skel_dirs)
    # kb_data_20 = map_25_to_20(skeleton_data, "kb")
    # frame_idx = 5
    # draw_skeleton_joint_20(frame_idx, kb_data_20)
    from configs.param_config import ConfigClass
    config = ConfigClass()

    from transforms.group_transforms import group_data_transforms

    index_param = {}
    index_param["modality"] = "Skeleton"
    index_param["is_skeleton_transform_velocity"] = False

    index_param["file_dirs"] = "/data/zqs/skeleton_datasets_36/datasets/cas_mhad/skeleton"
    index_param["file_format"] = "S{:03d}C{:03d}T{:03d}A{:03d}"
    # index_param["file_subset_list"] = "/home/zqs/workspace-xp/deployment/action-net-47/NTU_RGB+D_output/protocols/NTU_RGB+D_2_cross_view_train.txt"
    index_param["file_subset_list"] = "/data/zqs/skeleton_datasets_36/workspace/deployment/action-net-36/CAS_MHAD_output/protocols/CAS_MHAD_2_cross_view_train.txt"

    index_param["subset_type"] = "train"
    index_param["logger"] = None

    modality = "Skeleton"
    spatial_method = "spatial_group_crop_org"
    temporal_method = "dynamic_snapshot_sampling"

    loader_param = config.get_loader_param("CAS_MHAD", modality)

    spatial_param = loader_param["spatial_transform"][spatial_method]["train"]
    temporal_param = loader_param["temporal_transform"][temporal_method]["train"]

    index_param["temporal_param"] = temporal_param
    index_param["spatial_param"] = spatial_param
    # print(temporal_param)
    # print(index_param)
    # raise RuntimeError
    # spatial_transform, temporal_transform = group_data_transforms(None,
    #                                                               temporal_param,
    #                                                               modality)

    # spatial_transform = None

    data_loader = NTURGBD_SKELETON(index_param)

    # data_len = data_loader.__len__()
    # print(data_len)

    skeleton_data, label, length = data_loader.__getitem__(12)

    print(type(skeleton_data))
    print(skeleton_data[0].shape, skeleton_data[1].shape)

    # draw_cos_surface_skeleton_joint_20(5, skeleton_data[1,:,:,:].reshape(3,1140))
    pass