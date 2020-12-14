# -*- coding: utf-8 -*-
r"""This python file is designed to read the MSRDailyAct3D dataset,
including depth maps, color videos, and skeleton data.
"""
import cv2
import torch
import torch.utils.data as data
import numpy as np
from transforms.temporal_transforms import sparse_sampling_frames_from_segments_dual
from utils.depth_utils import DepthMaps2VectorMaps


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


class MSRDailyAct3D(data.Dataset):
    def __init__(self, index_param, spatial_transform=None, temporal_transform=None):
        self.modality = index_param["modality"]
        self.is_gradient = index_param["is_gradient"]
        self.is_vector = index_param["is_vector"]
        self.is_segmented = index_param["is_segmented"]
        self.subset_type = index_param["subset_type"]
        self.temporal_param = index_param["temporal_param"]
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.file_list, self.label_list, self.frame_list = \
            get_path_list_form_file_list(index_param["file_format"], index_param["file_subset_list"])

        # logger = index_param["logger"]
        #
        # sample_data, _, _ = self.__getitem__(0)
        # if logger is not None:
        #     logger.info("=> Sample data shape is {:s}".format(str(sample_data.shape)))
        # else:
        #     print("=> Sample data shape is {:s}".format(str(sample_data.shape)))

    def __getitem__(self, index):
        target_file = self.file_list[index]
        target_label = self.label_list[index]
        target_length = self.frame_list[index]

        target_data = self._loading_data(target_file, target_length)

        return target_data, target_label, target_length

    def __len__(self):
        return len(self.file_list)

    def _loading_data(self, file_path, seq_lens):
        """
        _loading_data:
        :param file_path:
        :return:
        """
        return self._loading_transformed_data(file_path, seq_lens)

    def _loading_transformed_data(self, file_path, seq_lens):
        """
        _loading_transformed_data:
        :param file_path:
        :return: data_processed:
        """
        if self.modality == "Depth":
            temporal_param = self.temporal_param

            if temporal_param is not None:
                # using seqs idx to reduce the loading time
                if "snapshot_sampling" in temporal_param.keys():
                    segments = temporal_param["snapshot_sampling"]["segments"]
                    sampling_type = temporal_param["snapshot_sampling"]["sampling_type"]
                    seqs_idx = sparse_sampling_frames_from_segments_dual(seq_lens,
                                                                          segments,
                                                                          sampling_type)

                elif "adjoin_snapshot_sampling" in temporal_param.keys():
                    segments = temporal_param["adjoin_snapshot_sampling"]["segments"]
                    sampling_type = temporal_param["adjoin_snapshot_sampling"]["sampling_type"]
                    seqs_idx = sparse_sampling_frames_from_segments_dual(seq_lens,
                                                                          segments//2,
                                                                          sampling_type)
            else:
                seqs_idx = None

            if self.is_vector:
                intrinsic_param = [525.0 / 2.0, 525.0 / 2.0, 315.5 / 2.0, 239.5 / 2.0]
                z_range = [0.0, 255.0] # [800.0, 4000.0]
                map_size = [240, 320]

                vector_convert = DepthMaps2VectorMaps(intrinsic_param, z_range, map_size)
                norm_val = True

            else:
                vector_convert = None
                norm_val = True

            data_processed = read_msr_depth_maps(file_path, seqs_idx,
                                                 norm_val=norm_val,
                                                 vector_convert=vector_convert,
                                                 is_segment=self.is_segmented)

        elif self.modality == "Skeleton":
            raise NotImplementedError("The part has not yet been implemented!")

        elif self.modality == "RGB":
            data_processed = read_msr_color_images(file_path, resize=True, is_gray=self.is_gradient)

        else:
            raise ValueError("Unknown modality: '{:s}'".format(self.modality))

        if self.temporal_transform is not None:
            data_processed = self.temporal_transform(data_processed)

        if self.spatial_transform is not None:
            data_processed = self.spatial_transform(data_processed)

        return data_processed


def read_gradient_file(g3_file, dim=[3, -1, 240, 320]):
    depth_gradient = np.fromfile(g3_file, dtype=np.float32)

    depth_gradient = depth_gradient.reshape(dim)

    return depth_gradient


def read_msr_depth_maps(depth_file, seqs_idx, norm_val=False, vector_convert=None, header_only=False, is_segment=False, is_mask=False):
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

    # depth maps and mask images are stored together per row
    dt = np.dtype([('depth', np.int32, cols), ('mask', np.uint8, cols)])
    frame_data = np.fromstring(sdata, dtype=dt)

    # extract the depth maps and mask data
    depth_map = frame_data['depth'].reshape([frames, rows, cols])
    mask_map = frame_data['mask'].reshape([frames, rows, cols])

    if is_segment:
        mask_map_inv = mask_map == 0
        depth_map[mask_map_inv] = 0

    if seqs_idx is not None:
        depth_map = depth_map[seqs_idx, :]
        mask_map = mask_map[seqs_idx, :]

    depth_map = depth_map.astype(np.float)

    if norm_val:
        #depth_map = (depth_map - 800.0) / 3200.0  # valid 800-4000mm
        depth_map = depth_map / 4000.0  # valid 800-4000mm
        depth_map = np.clip(depth_map, 0.0, 1.0)
        depth_map = depth_map * 255.0

    if vector_convert is not None:
        depth_map = vector_convert(depth_map)

    else:
        depth_map = depth_map[:, np.newaxis]

    if is_mask:
        return depth_map, mask_map
    else:
        return depth_map


def read_msr_color_images(color_file, resize=True, is_gray=False):
    r""" Extract the color images form a video file, and then return color images.

    Param:
        color_file (string): the path for a video file (*.avi).
        resize (bool): set it 'True' to reduce the images size from 640x480
        to 320x240.
    Return:
        color_images (np.uint8): in format of '[frames, height, width, color-bits]'
        Note: channels of color-bits is '[B,G,R]'
    """
    video_capture = cv2.VideoCapture(color_file)
    n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    chx = 1 if is_gray else 3
    if resize:
        color_images = np.empty([n_frames, 240, 320, chx], dtype=np.uint8)
        rows, cols, channels = [240, 320, chx]
    else:
        color_images = np.empty([480, 640, chx, n_frames], dtype=np.uint8)
        rows, cols, channels = [480, 640, chx]

    for f in range(n_frames):
        valid, color = video_capture.read()
        if not valid:
            break
        if is_gray:
            color = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        if resize:
            color = cv2.resize(color, (cols, rows))
        if is_gray:
            color_images[f, :, :, 0] = color
        else:
            color_images[f, :, :, :] = color

    video_capture.release()
    color_images = color_images.transpose(3, 0, 1, 2)
    return color_images


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

    assert word_coordinate or screen_coordinate, "Error: requires at least world or image coordinates to be true"

    data_raw = np.fromfile(skeleton_file, sep='\n')

    n_frames = int(data_raw[0])
    n_joints = int(data_raw[1])
    assert n_joints == 20, "Error: joint count is %i not 20" % n_joints

    data = np.zeros([n_frames, n_joints*4*2])

    for i in range(0, n_frames):
        ind = i*(n_joints*2*4+1) + 2
        data[i, :] = data_raw[ind+1:ind+20*4*2+1]

    # Get rid of confidence variable (it's useless for this data)
    data = data.reshape([n_frames, 40, 4])
    data = data[:, :, :3]

    if word_coordinate:
        skeletons_world = data[:, ::2, :]

        # convert meters to millimeters
        skeletons_world *= 1000.

    if screen_coordinate:
        skeletons_screen = data[:, 1::2, :].astype(np.float)

        # Rescale normalized coordinate by the image size
        skeletons_screen *= np.array(resolution + [1])

        r""" Note: The depth values in the image coordinates doesn't make sense (~20,000!),
        so replace them with the values from the world coordinates.
        """
        skeletons_screen[:, :, 2] = skeletons_world[:, :, 2]
        skeletons_screen = skeletons_screen.astype(np.int16)

    if word_coordinate and screen_coordinate:
        return skeletons_world, skeletons_screen

    elif word_coordinate:
        return skeletons_world

    elif screen_coordinate:
        return skeletons_screen

    return -1


if __name__ == '__main__':
    from configs.param_config import ConfigClass
    config = ConfigClass()

    config.set_environ()

    from matplotlib import pyplot as plt
    from transforms.volume_transforms import volume_data_transforms

    modality = "Depth"
    is_gradient = True

    subset = "train"
    index_param = {}
    index_param["modality"] = modality
    index_param["is_gradient"] = is_gradient
    index_param["is_segmented"] = False
    index_param["file_format"] = "/data/xp_ji/datasets/MSRDailyAct3D/<$file_format>_depth.bin"
    index_param["file_subset_list"] = "/home/xp_ji/workspace/deployment/action-net/MSRDailyAct3D_output/protocols/MSRDailyAct3D_1_cross_subjects_train.txt"


    spatial_method = "spatial_crop_g3"
    temporal_method = "snapshot_sampling"

    loader_param = config.get_loader_param("MSRDailyAct3D", modality)

    spatial_param = loader_param["spatial_transform"][spatial_method]["train"]
    temporal_param = loader_param["temporal_transform"][temporal_method]

    spatial_transform, temporal_transform = volume_data_transforms(spatial_param,
                                                                   temporal_param,
                                                                   spatial_method,
                                                                   temporal_method,
                                                                   modality,
                                                                   is_gradient,
                                                                   )


    data_loader = MSRDailyAct3D(index_param, spatial_transform=spatial_transform,
                                temporal_transform=temporal_transform)



    print("length: {:d}".format(data_loader.__len__()))

    depth_maps, label, length = data_loader.__getitem__(1)

    print(depth_maps.shape)

    print(label)

    dmaps = depth_maps[2, 7, :, :]

    print(dmaps.shape)
    print(dmaps.max())
    print(dmaps.min())

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








