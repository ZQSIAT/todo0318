import collections
import numpy as np
import torch
from scipy import misc
import torchvision.transforms as transforms
import random
from utils.depth_utils import DepthMapsRotation
import cv2
cv2.setNumThreads(0)
# cv2.ocl.setUseOPenCL(False)
import scipy

class SpatialCompose(object):
    """Composes several transforms together.

    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, depth_map):
        for trans_method in self.transforms:
            if trans_method is None:
                continue
            depth_map = trans_method(depth_map)

        return depth_map

    def randomize_parameters(self):
        for trans_method in self.transforms:
            if trans_method is None:
                continue
            trans_method.randomize_parameters()


class DepthMapsResize(object):
    """Resize the input depth maps to the given size.
    :param resize_shape: format [height, width]
    """
    def __init__(self, resize_shape, interpolation='bilinear', mode=None):
        assert isinstance(resize_shape, collections.Iterable) \
               and len(resize_shape) == 2, "Invalid input data!"

        self.resize_shape = tuple(resize_shape)
        self.interpolation = interpolation

        self.mode = mode if mode is not None else 'F'

    def __call__(self, depth_maps):
        """
        :param depth_maps: shape is [(C x F x H x W)] or [(C x H x W)]
        :return:
        """
        assert isinstance(depth_maps, np.ndarray), "Invalid input data!"

        if len(depth_maps.shape) == 4:
            size_original = [depth_maps.shape[2], depth_maps.shape[3]]

        elif len(depth_maps.shape) == 3:
            size_original = [depth_maps.shape[1], depth_maps.shape[2]]

        else:
            raise ValueError("Invalid shape of input data!")

        if not size_original == self.resize_shape:
            if len(depth_maps.shape) == 4:
                depth_maps_resized = np.zeros([depth_maps.shape[0],
                                               depth_maps.shape[1],
                                               self.resize_shape[0],
                                               self.resize_shape[1]])

                for chx_i in range(depth_maps.shape[0]):
                    for frm_i in range(depth_maps.shape[1]):
                        depth_maps_resized[chx_i, frm_i, :, :] = misc.imresize(depth_maps[chx_i, frm_i, :, :],
                                                                               self.resize_shape,
                                                                               mode=self.mode)

                depth_maps = depth_maps_resized

            else:
                depth_maps = misc.imresize(depth_maps, self.resize_shape, mode=self.mode)

        return depth_maps

    def randomize_parameters(self):
        pass


class DepthMapsRandomRotation(object):
    """rotation depth maps from 'numpy'
    """
    def __init__(self, alphas, betas, p, dz):
        self.alphas = alphas
        self.betas = betas
        self.p = p
        self.dz = dz

    def __call__(self, depth_maps):
        assert isinstance(depth_maps, np.ndarray), "Invalid input data!"
        #assert len(depth_maps.shape) == 4, "Invalid dim of input data!"

        if random.random() < self.p:
            if random.random() < self.p:
                r_alpha = random.randint(0, len(self.alphas)-1)
                r_beta = 0
            else:
                r_alpha = 0
                r_beta = random.randint(0, len(self.betas) - 1)
        else:
            r_alpha = 0
            r_beta = 0

        alpha = self.alphas[r_alpha]
        beta = self.betas[r_beta]

        dmr = DepthMapsRotation(alpha, beta, dz=self.dz)
        depth_roted_maps = dmr(depth_maps)
        depth_roted_maps[depth_roted_maps > 4095] = 0
        return depth_roted_maps


class DepthMapsCombineRotationView(object):
    """rotation depth maps from 'numpy'
    """
    def __init__(self, alphas, betas, dz):
        self.alphas = alphas
        self.betas = betas
        self.dz = dz

    def __call__(self, depth_maps):
        assert isinstance(depth_maps, np.ndarray), "Invalid input data!"

        depth_roted_maps = []
        for i in range(len(self.alphas)):
            alpha = self.alphas[i]
            beta = self.betas[i]

            dmr = DepthMapsRotation(alpha, beta, dz=self.dz)
            depth_roted = dmr(depth_maps)
            depth_roted_maps.append(depth_roted.copy())

        depth_roted_maps_array = np.concatenate(depth_roted_maps, axis=0)

        return depth_roted_maps_array


class DepthMapsToTensor(object):
    """Convert depth maps from 'numpy' [(C x F x H x W)] to 'tensor' [C x F x H x W].

    """
    def __init__(self, norm_value=1):
        assert isinstance(norm_value, int) and norm_value > 0, "Invalid input data!"

        self.norm_value = norm_value

    def __call__(self, depth_maps):

        assert isinstance(depth_maps, np.ndarray), "Invalid input data!"

        depth_maps_tensor = torch.from_numpy(depth_maps)

        return depth_maps_tensor.float().div(self.norm_value)

    def randomize_parameters(self):
        pass

class DepthMapsSpatialCenterCrop(object):
    """Spatial center crop on a spatio-temporal numpy or tensor
    input [C x F x H x W] or [C x H x W]
    """
    def __init__(self, size=[224, 224]):
        self.size = size

    def __call__(self, depth_maps):
        h, w = self.size

        if len(depth_maps.shape) == 4:
            _, _, tensor_h, tensor_w = depth_maps.shape
        else:
            _, tensor_h, tensor_w = depth_maps.shape

        if w > tensor_w or h > tensor_h:
            error_msg = (
                'Initial tensor spatial size should be larger then '
                'cropped size but got cropped sizes : ({w}, {h}) while '
                'initial tensor is ({t_w}, {t_h})'.format(
                    t_w=tensor_w, t_h=tensor_h, w=w, h=h))
            raise ValueError(error_msg)

        x1 = (tensor_w - w) // 2
        y1 = (tensor_h - h) // 2

        if len(depth_maps.shape) == 4:
            cropped = depth_maps[:, :, y1:y1 + h, x1:x1 + w]
        else:
            cropped = depth_maps[:, y1:y1 + h, x1:x1 + w]
        return cropped


class DepthMapsSpatialRandomCrop(object):
    """Crops a random spatial crop in a spatio-temporal numpy or tensor
    input [C x F x H x W] or [C x H x W]
    """
    # ToDo: add padding

    def __init__(self, size=[224, 224], padding=0):
        """
        Args:
            size (tuple): in format (height, width)
        """
        self.size = size
        self.padding = padding

    def __call__(self, depth_maps):
        h, w = self.size

        if len(depth_maps.shape) == 4:
            _, _, tensor_h, tensor_w = depth_maps.shape
        else:
            _, tensor_h, tensor_w = depth_maps.shape

        if w > tensor_w or h > tensor_h:
            error_msg = (
                'Initial tensor spatial size should be larger then '
                'cropped size but got cropped sizes : ({w}, {h}) while '
                'initial tensor is ({t_w}, {t_h})'.format(
                    t_w=tensor_w, t_h=tensor_h, w=w, h=h))
            raise ValueError(error_msg)
        x1 = random.randint(0, tensor_w - w)
        y1 = random.randint(0, tensor_h - h)

        if len(depth_maps.shape) == 4:
            cropped = depth_maps[:, :, y1:y1 + h, x1:x1 + w]
        else:
            cropped = depth_maps[:, y1:y1 + h, x1:x1 + w]
        return cropped


class DepthMapsSpatialRandomHorizontalFlip(object):
    """Randomly horizontally flips the given depthMaps with a probability of 0.5
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, depth_maps, is_flow=False):
        if random.random() < self.p:
            if len(depth_maps.shape) == 4:
                depth_maps = depth_maps[:, :, :, ::-1]
            else:
                depth_maps = depth_maps[:, :, ::-1]

        return depth_maps


class DepthMapsNormalizedToTensor(object):
    """Normalize the depth maps from 'numpy' to 'tensor' with range of [0-1].
    """
    def __init__(self, norm_value=1.0):
        assert isinstance(norm_value, float) and norm_value > 0, "Invalid input data!"

        self.norm_value = norm_value

    def __call__(self, depth_maps):

        assert isinstance(depth_maps, np.ndarray), "Invalid input data!"

        depth_maps_tensor = torch.from_numpy(depth_maps.copy())

        return depth_maps_tensor.float().div(self.norm_value)

    def randomize_parameters(self):
        pass


class DepthMapsStandardization(object):
    """Standardize the input tensor with .
    :param mean:
    :param std:
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, depth_maps_tensor):
        # TODO: make efficient

        if isinstance(self.mean, float):
            depth_maps_tensor.sub_(self.mean).div_(self.std)
        else:
            for t, m, s in zip(depth_maps_tensor, self.mean, self.std):
                t.sub_(m).div_(s)
        return depth_maps_tensor

    def randomize_parameters(self):
        pass


if __name__ == '__main__':
    depth_maps = np.random.randint(0, 4095, size=[3, 16, 240, 320])

    spatial_transform = SpatialCompose([DepthMapsSpatialCentorCrop([224, 224])])

    depth_maps = spatial_transform(depth_maps)
    print(depth_maps.shape)

    # flip_h = DepthMapsSpatialRandomHorizontalFlip()
    #
    # depth_maps_h = flip_h(depth_maps)
    # print(depth_maps_h.shape)
    # print(depth_maps[0, 0, 0, :])
    # print(depth_maps_h[0, 0, 0, :])










