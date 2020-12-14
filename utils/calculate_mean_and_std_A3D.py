import torch
import numpy as np
from datasets.MSRDailyAct3D import MSRDailyAct3D
from datasets.MSRAction3D import MSRAction3D
from datasets.NTU_RGBD import NTURGBD
from transforms.volume_transforms import volume_data_transforms
import time


class ReserveMeanStd(object):
    """reserve the calculated mean and std for applying on test data
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.mean = None
        self.std = None

    def update(self, mean, std):
        self.mean = mean
        self.std = std


def calculate_mean_and_std_3c(index_param ,spatial_transform, temporal_transform):

    data_loader = NTURGBD(index_param, spatial_transform=spatial_transform,
                                temporal_transform=temporal_transform)

    amax = -1e5
    amin = 1e5
    data_sum = torch.Tensor([0., 0., 0.]).cuda()
    data_sum_sq = torch.Tensor([0., 0., 0.]).cuda()
    data_count = torch.Tensor([0., 0., 0.]).cuda()

    start_time = time.time()

    for index in range(data_loader.__len__()):

        rgb_videos, label, length = data_loader.__getitem__(index)
        if index % 10 == 0:
            print("=>[{:d}/{:d}]".format(index, data_loader.__len__()))
            print(rgb_videos.shape)


        data_x = rgb_videos.cuda()
        data_x = data_x.view(3, -1).cuda()
        # data_count += (data_x > 0).float().sum()#.shape[0]
        data_count += data_x.shape[1]
        data_sum += data_x.sum(dim=1)
        data_sum_sq += (data_x * data_x).sum(dim=1)

    data_mean = data_sum / data_count
    temp = data_sum_sq - (data_sum * data_sum / data_count)
    data_var = temp / (data_count - 1)
    data_std = torch.sqrt(data_var)

    print("data mean: {:s}".format(str(data_mean)))
    print("data std: {:s}".format(str(data_std)))

    end_time = time.time()

    print("Time elapsed: {:.2f} minutes".format((end_time - start_time) / 60.0))


def calculate_mean_and_std(index_param, spatial_transform, temporal_transform):
    data_loader = MSRDailyAct3D(index_param, spatial_transform=spatial_transform,
                                temporal_transform=temporal_transform)

    amax = -1e5
    amin = 1e5
    data_sum = torch.Tensor([0.]).cuda()
    data_sum_sq = torch.Tensor([0.]).cuda()
    data_count = torch.Tensor([0.]).cuda()

    start_time = time.time()

    for index in range(data_loader.__len__()):

        depth_maps, label, length = data_loader.__getitem__(index)


        data_x = depth_maps.cuda()
        data_x = data_x.view(-1, 1).cuda()

        # data_count += (data_x > 0).float().sum()#.shape[0]
        data_count += data_x.shape[0]
        data_sum += data_x.sum()
        data_sum_sq += (data_x * data_x).sum()

        tmax = depth_maps.max()
        depth_maps[depth_maps == 0] = tmax
        tmin = depth_maps.min()

        if tmax > amax:
            amax = tmax
        if tmin < amin:
            amin = tmin

        if index % 10 ==0:
            print("processing {:d}".format(index))
            print("{:d}-{:d}--{:d}".format(index, label, length))
            print("{:f}##{:f}".format(tmax, tmin))

    data_mean = data_sum / data_count
    temp = data_sum_sq - (data_sum * data_sum / data_count)
    data_var = temp / (data_count - 1)
    data_std = torch.sqrt(data_var)

    print("data mean: {:4f}".format(data_mean.item()))
    print("data std: {:4f}".format(data_std.item()))

    end_time = time.time()
    print("data max: {:.2f}".format(amax))
    print("data min: {:.2f}".format(amin))

    print("file count: {:d}".format(data_loader.__len__()))

    print("Time elapsed: {:.2f} minutes".format((end_time - start_time) / 60.0))

    # cur_frames = depth_maps.shape[1]
    #
    # data_feat[:, :, start_idx:start_idx+cur_frames, :, :] = depth_maps
    #
    # start_idx = start_idx + cur_frames





if __name__ == "__main__":
    #loader_param = {"data_name": "cifar-10", "subset": "test", "data_dir": "/data/xp_ji/datasets/cifar", "data_dosage": 0.5}

    from configs.param_config import ConfigClass

    config = ConfigClass()
    config.set_environ()

    subset = "train"
    modality = "Depth"
    index_param = {}
    data_name = "NTU_RGB+D"
    index_param["modality"] = modality
    index_param["dual_input"] = False
    index_param["subset_type"] = subset
    index_param["is_gradient"] = False
    index_param["is_segmented"] = False
    index_param["file_format"] = "/data/xp_ji/datasets/{:s}/<$file_format>_depth.bin".format(data_name)
    index_param[
        "file_subset_list"] = "/home/xp_ji/workspace/deployment/action-net/{data_name:s}_output/" \
                              "protocols/{data_name:s}_1_cross_subjects_{subset:s}.txt".format(data_name=data_name,
                                                                                               subset=subset)

    spatial_method = "spatial_crop"
    temporal_method = "snapshot_sampling"

    loader_param = config.get_loader_param(data_name, modality)

    spatial_param = loader_param["spatial_transform"][spatial_method][subset]
    temporal_param = loader_param["temporal_transform"][temporal_method][subset]

    print(spatial_param)

    spatial_param["normalization"] = 255.0
    spatial_param["standardization"]["mean"] = 0.0
    spatial_param["standardization"]["std"] = 1.0
    index_param["temporal_param"] = None

    spatial_transform, temporal_transform = volume_data_transforms(spatial_param,
                                                                   temporal_param,
                                                                   modality,
                                                                   False)

    calculate_mean_and_std(index_param, spatial_transform, temporal_transform=None)

    #print(mean, std)
