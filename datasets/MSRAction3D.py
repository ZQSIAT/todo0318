# -*- coding: utf-8 -*-
r"""This python file is designed to read the MSRAction3D dataset,
including depth maps, color videos, and skeleton data.
"""
import cv2
import torch
import torch.utils.data as data
import numpy as np


def get_path_list_form_file_list(file_format, file_list_file):
    """
    get_path_list_form_file_list:
    :param file_format:
    :param file_list_file:
    :return:
    """

    file_list = []
    label_list = []
    frame_list = []

    with open(file_list_file, "r") as flp:
        for line in flp.readlines():
            flp_line = line.strip("\n").split("\t")

            # default format is "<$file_format>/t<$label>/t<[frames,height,width]>"
            target_file = file_format.replace("<$file_format>", flp_line[0])
            target_label = int(flp_line[1])
            target_length = int(flp_line[2])

            file_list.append(target_file)
            label_list.append(target_label)
            frame_list.append(target_length)

        flp.close()

    # file_path = list(map(lambda x: file_format.format(x), file_list))

    return file_list, label_list, frame_list


class MSRAction3D(data.Dataset):
    def __init__(self, index_param, spatial_transform=None, temporal_transform=None):
        self.modality = index_param["modality"]
        self.is_gradient = index_param["is_gradient"]
        self.is_segmented = index_param["is_segmented"]
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.file_list, self.label_list, self.frame_list = \
            get_path_list_form_file_list(index_param["file_format"], index_param["file_subset_list"])

    def __getitem__(self, index):
        target_file = self.file_list[index]
        target_label = self.label_list[index]
        target_length = self.frame_list[index]

        target_data = self._loading_data(target_file)

        return target_data, target_label, target_length

    def __len__(self):
        return len(self.file_list)

    def _loading_data(self, file_path):
        """
        _loading_data:
        :param file_path:
        :return:
        """
        return self._loading_transformed_data(file_path)

    def _loading_transformed_data(self, file_path):
        """
        _loading_transformed_data:
        :param file_path:
        :return: data_processed:
        """
        if self.modality == "Depth":
            data_processed = read_msr_depth_maps(file_path, is_segment=self.is_segmented)
            # if self.is_gradient:
            #     data_gradient = np.gradient(data_processed)
            #     data_processed = np.array(data_gradient)
            #     data_processed_norm = 1 / (np.sqrt(np.square(data_processed).sum(axis=0, keepdims=True) + 1))
            #     data_processed = data_processed * data_processed_norm.repeat(3, axis=0)
            # else:
            #     data_processed = data_processed[np.newaxis, :]

            data_processed = data_processed[np.newaxis, :]

            # transfer to GPU

            if self.temporal_transform is not None:
                data_processed = self.temporal_transform(data_processed)

            if self.spatial_transform is not None:
                data_processed = self.spatial_transform(data_processed)

        elif self.modality == "Skeleton":
            raise NotImplementedError("The part has not yet been implemented!")

        elif self.modality == "RGB":
            raise ValueError("This dataset does not contain RGB videos!")

        else:
            raise ValueError("Unknown modality: '{:s}'".format(self.modality))

        return data_processed


def read_msr_depth_maps(depth_file, header_only=False, is_segment=False, is_mask=False):
    r""" Read the MSR binary file of depth maps, and then return depth maps and
    mask maps.

    Param:
        depth_file (string): the path for a depth file (*.bin).
        is_segment (bool): set it 'True' to get human segmented maps.
        is_mask: set it 'True' to get mask maps
    Return:
        depth_map (np.uint32): in format of '[frames, height, width]'
        mask_map (np.uint8): in format of '[frames, height, width]'
        Note: mask_map is returned only the input parameter 'is_mask' is 'True'
    """

    file_ = open(depth_file, 'rb')

    # get header info
    frames = np.fromstring(file_.read(4), dtype=np.int32)[0]
    cols = np.fromstring(file_.read(4), dtype=np.int32)[0]
    rows = np.fromstring(file_.read(4), dtype=np.int32)[0]

    if header_only:
        return [frames, rows, cols]

    # read the remaining data
    sdata = file_.read()

    # depth maps
    #dt = np.dtype([('depth', np.int32, cols), ('mask', np.uint8, cols)])
    frame_data = np.fromstring(sdata, dtype=np.int32)

    # extract the depth maps and mask data
    depth_map = frame_data.reshape([frames, rows, cols])

    return depth_map


def read_msr_skeleton_data(skeleton_file, word_coordinate=True, screen_coordinate=False, resolution=[240, 320]):
    r""" Extract the skeleton data form a txt file, and then return the joint data.

    Param:
        skeleton_file (string): the path for a txt file (*.txt).
        word_coordinate: set it 'True' to acquire skeletons in world coordinate
        screen_coordinate: set it 'True' to acquire skeletons in screen coordinate
        resolution (list): the resolution utilized to rescale the screen coordinate.
            Default resolution is 240x320
    Return:
        skeletons_world (np.float64): in format of '[frames, joints, axis]'
        skeletons_screen (np.float64): in format of '[frames, joints, axis]'
    """
    raise NotImplementedError


if __name__ == '__main__':
    from configs.param_config import ConfigClass
    config = ConfigClass()

    config.set_environ()

    from matplotlib import pyplot as plt
    from transforms.volume_transforms import volume_data_transforms

    index_param = {}
    index_param["modality"] = "Depth"
    index_param["is_gradient"] = True
    index_param["is_segmented"] = False
    index_param["file_format"] = "/data/xp_ji/datasets/MSRAction3D/<$file_format>_sdepth.bin"
    index_param["file_subset_list"] = "/home/xp_ji/workspace/deployment/action-net/MSRAction3D_output/protocols/MSRAction3D_1_cross_subjects_train.txt"

    modality = "Depth"
    spatial_method = "spatial_crop_g3"
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


    # amax = 0
    # amin = 4095
    # data_feat = np.zeros(shape=(1, 1, 50000, 240, 320))
    # start_idx = 0
    #
    # for index in range(data_loader.__len__()):
    #     depth_maps, label, length = data_loader.__getitem__(index)
    #     print("{:d}-{:d}--{:d}".format(index, label, length))
    #     tmax = depth_maps.max()
    #     depth_maps[depth_maps==0]=tmax
    #     tmin = depth_maps.min()
    #     print("{:f}##{:f}".format(tmax, tmin))
    #     if tmax > amax:
    #         amax = tmax
    #     if tmin < amin:
    #         amin = tmin
    #
    #     # cur_frames = depth_maps.shape[1]
    #     #
    #     # data_feat[:, :, start_idx:start_idx+cur_frames, :, :] = depth_maps
    #     #
    #     # start_idx = start_idx + cur_frames
    #
    #
    # print(amin)
    # print(amax)

    # print(start_idx)
    # data_feat = data_feat[:, :, 0:start_idx, :, :]
    # print(data_feat.shape)
    #
    # mean = np.mean(data_feat, axis=(0, 2, 3, 4))
    # std = np.std(data_feat, axis=(0, 2, 3, 4))
    #
    #
    # print(mean)
    # print(std)








