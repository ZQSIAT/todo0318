# -*- coding: utf-8 -*-
r"""This python file is designed to read the NTU RGB+D dataset,
including depth maps, color videos, and skeleton data.
"""
from multiprocessing import Process
import cv2
# cv2.setNumThreads(0)
# cv2.ocl.setUseOPenCL(False)
import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import random
from utils.depth_utils import add_gaussian_shifts
from transforms.temporal_transforms import sparse_sampling_frames_from_segments_dual
from utils.depth_utils import DepthMaps2VectorMaps


depth_format = "MDepth-{idx:08d}.png"


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # with open(path, 'rb') as f:
    #     img = Image.open(f)
    #     # img = img.convert('L')
    #     img_arr = np.array(img, dtype=np.float16)
    if not os.path.exists(path):
        print(path)
        import ipdb; ipdb.set_trace()
        raise RuntimeError
    img = Image.open(path)
    # img = img.convert('L')
    img_arr = np.array(img, dtype=np.float32)
    return img_arr

def cv2_loader(path):
    if not os.path.exists(path):
        print(path)
        raise RuntimeError
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return img

def get_path_list_form_file_list(file_format, file_list_file):
    """
    get_path_list_form_file_list:
    :param file_format:
    :param file_list_file:
    :return:
    """

    file_dir_list = []
    label_list = []
    frame_list = []

    with open(file_list_file, "r") as flp:
        for line in flp.readlines():
            flp_line = line.strip("\n").split("\t")

            # default format is "<$file_format>/t<$label>/t<[frames,height,width]>"
            target_file_path = file_format + "/" + flp_line[0]
            target_label = int(flp_line[1])
            target_length = int(flp_line[2])

            file_dir_list.append(target_file_path)
            label_list.append(target_label)
            frame_list.append(target_length)

        flp.close()

    # file_path = list(map(lambda x: file_format.format(x), file_list))

    return file_dir_list, label_list, frame_list


class NTURGBD(data.Dataset):
    def __init__(self, index_param, spatial_transform=None, temporal_transform=None):
        self.modality = index_param["modality"]
        self.is_gradient = index_param["is_gradient"]
        self.is_segmented = index_param["is_segmented"]
        # self.is_vector = index_param["is_vector"]
        self.data_dosage = index_param["data_dosage"]
        self.subset_type = index_param["subset_type"]
        self.temporal_param = index_param["temporal_param"]
        self.small_validation = index_param["small_validation"]
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        logger = index_param["logger"]
        file_list, label_list, frame_list = \
            get_path_list_form_file_list(index_param["file_format"], index_param["file_subset_list"])

        self.file_list, self.label_list, self.frame_list = \
            self._small_dosage(file_list, label_list, frame_list, logger)

        self.add_noise = index_param["add_noise"]

        # sample_data, _ ,_ = self.__getitem__(0)
        # if logger is not None:
        #     logger.info("=> Sample data shape is {:s}".format(str(sample_data.shape)))
        # else:
        #     print("=> Sample data shape is {:s}".format(str(sample_data.shape)))


    def _small_dosage(self, file_list, label_list, frame_list, logger):
        if self.data_dosage is not None:
            assert isinstance(self.data_dosage, float) and 0.0 < self.data_dosage <= 1.0, "Invalid input data!"
            if self.data_dosage < 1.0:
                if self.small_validation or self.subset_type=="train":
                    data_len = len(file_list)
                    select_idx = random.sample(range(data_len), int(data_len * self.data_dosage))
                    file_list_s = [file_list[i] for i in select_idx]
                    label_list_s = [label_list[i] for i in select_idx]
                    frame_list_s = [frame_list[i] for i in select_idx]

                    if logger is not None:
                        logger.info("=> Using {:2.0f}% [{:d}/{:d}] samples "
                                    "for model {:s}.".format(self.data_dosage * 100,
                                                             len(label_list_s),
                                                             data_len,
                                                             self.subset_type))
                    else:
                        print("=> Using {:2.0f}% [{:d}/{:d}] samples "
                                    "for model {:s}.".format(self.data_dosage * 100,
                                                             len(label_list_s),
                                                             data_len,
                                                             self.subset_type))
                    return file_list_s, label_list_s, frame_list_s

        return file_list, label_list, frame_list

    def __getitem__(self, index):
        target_file_dir = self.file_list[index]
        target_label = self.label_list[index]
        target_length = self.frame_list[index]
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        target_data = self._loading_data(target_file_dir, target_length)

        return target_data, target_label, target_length

    def __len__(self):
        return len(self.file_list)

    def _loading_data(self, file_path, file_count):
        """
        _loading_data:
        :param file_path:
        :return:
        """
        return self._loading_transformed_data(file_path, file_count)

    def _loading_transformed_data(self, file_path, file_count):
        """
        _loading_transformed_data:
        :param file_path:
        :return: data_processed:
        """
        if self.modality == "Depth":

            temporal_param = self.temporal_param

            if temporal_param is not None:
                # using file idx to reduce the loading time
                if "snapshot_sampling" in temporal_param.keys():
                    segments = temporal_param["snapshot_sampling"]["segments"]
                    sampling_type = temporal_param["snapshot_sampling"]["sampling_type"]
                    file_idx = sparse_sampling_frames_from_segments_dual(file_count,
                                                                          segments,
                                                                          sampling_type)

                elif "adjoin_snapshot_sampling" in temporal_param.keys():
                    segments = temporal_param["adjoin_snapshot_sampling"]["segments"]
                    sampling_type = temporal_param["adjoin_snapshot_sampling"]["sampling_type"]
                    file_idx = sparse_sampling_frames_from_segments_dual(file_count,
                                                                          segments//2,
                                                                          sampling_type)
                else:
                    file_idx = None
            else:
                file_idx = None

            vector_convert = None
            norm_val = True

            data_processed = read_ntu_depth_maps(file_path,
                                                 img_loader=cv2_loader,
                                                 file_idx=file_idx,
                                                 vector_convert=vector_convert,
                                                 norm_val=True,
                                                 add_noise=self.add_noise)

            if self.temporal_transform is not None:
                data_processed = self.temporal_transform(data_processed)

            if self.spatial_transform is not None:
                data_processed = self.spatial_transform(data_processed)



        else:
            raise ValueError("Unknown modality: '{:s}'".format(self.modality))

        return data_processed


def read_ntu_depth_maps(depth_file, img_loader=cv2_loader, file_idx=None, vector_convert=None, norm_val=False, add_noise=False):
    """

    :param depth_file:
    :param img_loader:
    :param size:
    :return: CxFxHxW
    """

    if file_idx is not None:
        depth_file_list = [depth_format.format(idx=idx+1) for idx in file_idx]
    else:
        depth_file_list = sorted(os.listdir(depth_file))

    img_array_list = []

    for img_i in depth_file_list:
        img_path = depth_file + "/" + img_i
        img = img_loader(img_path)

        if add_noise:
            img = add_gaussian_shifts(img)

        img_array_list.append(img)

    img_array = np.concatenate([np.expand_dims(x, 0) for x in img_array_list], axis=0)

    # import ipdb;ipdb.set_trace()
    # img_array_crop = img_array[:, 90: 370, 110: 390]
    img_array_crop = img_array[:, 90: 410, 90: 410]

    img_array_crop = img_array_crop.astype(np.float)

    if norm_val:
        # img_array_crop = (img_array_crop - 500.0) / 4000.0  # valid 500-4500mm
        # import ipdb; ipdb.set_trace()
        # img_array_crop[img_array_crop>4500] = 0
        img_array_crop[img_array_crop<500] = 0
        img_array_crop = img_array_crop / 4500.0
        depth_map = np.clip(img_array_crop, 0.0, 1.0)
        img_array_crop = depth_map * 255.0

    if vector_convert is not None:
        img_array_crop = vector_convert(img_array_crop)

    else:
        img_array_crop = img_array_crop[:, np.newaxis]

    return img_array_crop


if __name__ == '__main__':
    from configs.param_config import ConfigClass
    config = ConfigClass()

    config.set_environ()

    from matplotlib import pyplot as plt
    from transforms.group_transforms import group_data_transforms

    is_gradient = False
    use_depth_seq = False
    index_param = {}
    index_param["modality"] = "Depth"
    index_param["is_gradient"] = is_gradient
    index_param["is_segmented"] = False
    index_param["file_format"] = "/home/432/zhaoqingsong/data/ntu_rgb+d/depth_masked"
    index_param["file_subset_list"] = "/home/432/zhaoqingsong/deployment/NTU_RGB+D_output/protocols/NTU_RGB+D_2_cross_view_train.txt"

    index_param["data_dosage"] = 1.0
    index_param["subset_type"] = "train"
    index_param["small_validation"] = True
    index_param["logger"] = None
    index_param["add_noise"] = False
    modality = "Depth"
    spatial_method = "spatial_crop"
    temporal_method = "snapshot_sampling"

    loader_param = config.get_loader_param("NTU_RGB+D", modality)

    spatial_param = loader_param["spatial_transform"][spatial_method]["train"]
    temporal_param = loader_param["temporal_transform"][temporal_method]["train"]

    # spatial_param["normalization"] = 1.0
    # spatial_param["standardization"]["mean"] = 0.0
    # spatial_param["standardization"]["std"] = 1.0

    spatial_transform, temporal_transform = group_data_transforms(spatial_param,
                                                                   temporal_param,
                                                                   modality, is_gradient, False, use_depth_seq)

    index_param["temporal_param"] = temporal_param

    data_loader = NTURGBD(index_param, spatial_transform=spatial_transform,
                                temporal_transform=temporal_transform)



    #print("length: {:d}".format(data_loader.__len__()))

    data_len = data_loader.__len__()

    # torch.utils.data.DataLoader(target_dataset,
    #                             batch_size=args.batch_size,
    #                             shuffle=args.train_shuffle,
    #                             num_workers=args.num_workers,
    #                             pin_memory=args.pin_memory)

    # mask = torch.zeros((1, 424, 512))
    # for i in range(data_len):
    #     depth_maps, label, length = data_loader.__getitem__(i)
    #
    #     mask += (((depth_maps > 0).sum(dim=1)) > 0).float()
    #
    #     if i % 100 == 0:
    #         print("=> {:d}/{:d}".format(i, data_len))
    #depth_maps_tensor = ToTensor(depth_maps[0])
    # plt.imshow(depth_maps_tensor[0,:,:])
    # plt.colorbar()
    # plt.show()
    # print(depth_maps.shape)
    # plt.imshow(depth_maps[0, 6, :, :])
    # plt.colorbar()
    # plt.show()

    # print(mask.shape)
    # from torchvision import transforms
    # mask_img = ((mask[0,:,:]>5000).numpy())*255
    #
    # mask_img = mask_img[:, :, np.newaxis]
    #
    # print(mask_img.shape)
    #
    # img_2 = transforms.ToPILImage()(mask_img)
    # print("img_2 = ", img_2)
    # img_2.save("/home/xp_ji/workspace/pt-work/action-net/test-cv_mask_L5000.png")

    # plt.imshow(mask[0,:,:]>0)
    # plt.savefig("/home/xp_ji/workspace/pt-work/action-net/test-cs_mask.png")
    # plt.colorbar()
    # plt.show()
    # plt.savefig("/home/xp_ji/workspace/pt-work/action-net/test-2.png")


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








