# -*- coding: utf-8 -*-
r"""This python file is designed to read the UOW Combined3D dataset,
including depth maps and skeleton data.
"""

import cv2
import numpy as np


def read_uow_depth_maps(depth_file):
    r""" Read the UOW binary file of depth maps, and then return depth maps.

    Param:
        depth_file (string): the path for a depth file (*.bin).
    Return:
        depth_map (np.uint16): in format of '[frames, height, width]'
    Note: Only the segmented date are stored in the binary file.
    """

    file_ = open(depth_file, 'rb')

    # get header info
    frames = np.fromstring(file_.read(4), dtype=np.int32)[0]
    cols = np.fromstring(file_.read(4), dtype=np.int32)[0]
    rows = np.fromstring(file_.read(4), dtype=np.int32)[0]

    # read the remaining data
    data = file_.read()

    # extract the frame data
    frame_data = np.fromstring(data, dtype=np.int16)

    print(frame_data.shape)
    # extract the depth maps and mask data
    depth_map = frame_data.astype(np.uint16).reshape([frames, rows, cols])

    return depth_map


def read_uow_skeleton_data(skeleton_file):
    r""" Extract the skeleton data form a txt file, and then return color images.
        Note:
            The skeleton data are separated into two files, eg.'*_mm.txt' &
            '*_scr.txt'.

    Param:
        skeleton_file (string): the path for a txt file (*.txt).
    Return:
        skeletons_data (np.float64): in format of '[frames, joints, axis]'

    """

    data_raw = np.fromfile(skeleton_file, sep='\n')

    n_frames = int(data_raw[0])
    n_joints = int(data_raw[1])
    assert n_joints == 20, "Error: joint count is %i not 20" % n_joints

    data = np.zeros([n_frames, n_joints*3])

    for i in range(0, n_frames):
        ind = i*(n_joints*3) + 2
        data[i, :] = data_raw[ind+1:ind+20*3+1]
    skeletons_data = data.reshape([n_frames, 20, 3])

    return skeletons_data


if __name__ == "__main__":
    from configs.param_config import ConfigClass
    config = ConfigClass()

    config.set_environ()

    from matplotlib import pyplot as plt
    from transforms.volume_transforms import volume_data_transforms

    index_param = {}
    index_param["modality"] = "Depth"
    index_param["is_gradient"] = False
    index_param["is_segmented"] = False
    index_param["file_format"] = "/data/xp_ji/datasets/UOW_Combined3D/Depth/<$file_format>_depth.bin"
    index_param["file_subset_list"] = "/home/xp_ji/workspace/deployment/action-net/MSRAction3D_output/protocols/MSRAction3D_1_cross_subjects_train.txt"

    modality = "Depth"
    spatial_method = "spatial_crop"
    temporal_method = "snapshot_sampling"

    loader_param = config.get_loader_param("MSRDailyAct3D", modality)

    spatial_param = loader_param["spatial_transform"][spatial_method]
    temporal_param = loader_param["temporal_transform"][temporal_method]

    spatial_transform, temporal_transform = volume_data_transforms(spatial_param,
                                                                   temporal_param,
                                                                   spatial_method,
                                                                   temporal_method,
                                                                   modality)

    data_loader = MSRAction3D(index_param, spatial_transform=spatial_transform,
                                temporal_transform=temporal_transform)

    print("length: {:d}".format(data_loader.__len__()))

    depth_maps, label, length = data_loader.__getitem__(1)

    print(depth_maps.shape)


    # depth_maps = read_msr_depth_maps("/data/xp_ji/datasets/MSRAction3D/a20_s10_e03_sdepth.bin")
    dmaps = depth_maps[1, 3, :, :]
    #
    # print(dmaps.shape)
    # print(dmaps.max())
    # print(dmaps.min())

    plt.imshow(dmaps)
    plt.colorbar()
    plt.show()


