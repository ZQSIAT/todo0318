import math
import numpy as np
from scipy.sparse import coo_matrix
import cv2
# from numba import jit

class DepthMaps2VectorMaps(object):
    def __init__(self,  intrinsic_param, z_range, map_size, is_norm=False):
        fx, fy, cx, cy = intrinsic_param
        self.fx = fx
        self.fy = fy
        self.z_min, self.z_max = z_range
        self.is_norm = is_norm

        # row -- height -- y
        # col -- width --- x
        self.h, self.w = map_size
        x_idx, y_idx = np.meshgrid(np.arange(self.w),
                                   np.arange(self.h),
                                   sparse=True)

        self.homo = np.ones((self.h, self.w))

        self.x_idx = x_idx# - cx
        self.y_idx = y_idx# - cy

    def __call__(self, depth_data):
        vector_data = self.depth2vector(depth_data)
        return vector_data

    def depth2vector(self, depth):
        valid = (depth > self.z_min) & (depth < self.z_max)
        z = np.where(valid, depth, 0)

        x = np.where(valid, z * self.x_idx / self.fx, 0)
        y = np.where(valid, z * self.y_idx / self.fy, 0)

        vector_map = np.stack((x, y, z), axis=-1)

        if self.is_norm:
            v_norm_homo = np.linalg.norm(vector_map, ord=2, axis=3, keepdims=True)
            vector_map = vector_map/(v_norm_homo + 1e-5)
            # vector_map = vector_map[0:3]

        return vector_map

class DepthMap2VectorMap(object):
    def __init__(self,  intrinsic_param, z_range, map_size, is_norm=False):
        fx, fy, cx, cy = intrinsic_param
        self.fx = fx
        self.fy = fy
        self.z_min, self.z_max = z_range
        self.is_norm = is_norm



        # row -- height -- y
        # col -- width --- x
        self.h, self.w = map_size
        x_idx, y_idx = np.meshgrid(np.arange(self.w),
                                   np.arange(self.h),
                                   sparse=True)

        self.homo = np.ones((self.h, self.w))

        self.x_idx = x_idx# - cx
        self.y_idx = y_idx# - cy

    def __call__(self, depth_data):
        vector_data = self.depth2vector(depth_data)
        return vector_data

    def depth2vector(self, depth):
        valid = (depth > self.z_min) & (depth < self.z_max)
        z = np.where(valid, depth, 0)

        x = np.where(valid, z * self.x_idx / self.fx, 0)
        y = np.where(valid, z * self.y_idx / self.fy, 0)

        vector_map = np.stack((x, y, z, self.homo), axis=0)

        if self.is_norm:
            v_norm_homo = np.linalg.norm(vector_map, ord=2, axis=0, keepdims=True)
            vector_map = vector_map/v_norm_homo
            vector_map = vector_map[0:3]

        return vector_map

class DepthMapsRotation(object):
    def __init__(self, alpha, beta, dz, fx=525.0, fy=525.0, cx=315.5, cy=239.5):
        r_alpha = alpha * math.pi / 180.0
        r_beta = beta * math.pi / 180.0
        self.fx = fx/2
        self.fy = fy/2
        self.cx = cx/2
        self.cy = cy/2

        if alpha == 0 and beta == 0:
            self.rotation = False
        else:
            self.rotation = True

        self.af_mat = np.array((
            (math.cos(r_alpha), 0, math.sin(r_alpha), -1 * dz * math.sin(r_alpha)),
            (math.sin(r_alpha) * math.sin(r_beta), math.cos(r_beta), -math.cos(r_alpha) * math.sin(r_beta),
             dz * math.cos(r_alpha) * math.sin(r_beta)),
            (-math.sin(r_alpha) * math.cos(r_beta), math.sin(r_beta), math.cos(r_alpha) * math.cos(r_beta),
             -1 * dz * math.cos(r_alpha) * math.cos(r_beta) + dz),
            (0, 0, 0, 1),
        ))

    def __call__(self, depth_maps_data):
        if self.rotation:
            depth_maps_r_data = self.rotate_depth_maps(depth_maps_data)
            return depth_maps_r_data
        else:
            return depth_maps_data
    # @jit
    def rotate_depth_maps(self, depth_maps_data):
        """
        :param depth_maps_data: CxFxHxW
        :return:
        """
        lens = depth_maps_data.shape[1]
        for i in range(lens):
            depth_data = depth_maps_data[0, i, :, :]

            w_points = self.point2cloud(depth_data)
            r_points = self.points_rotation(w_points)
            depth_r_data = self.cloud2point(r_points)
            depth_maps_data[0, i, :, :] = depth_r_data
        return depth_maps_data

    def point2cloud(self, depth):
        rows, cols = depth.shape
        self.width = cols
        self.height = rows
        c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
        valid = (depth > 0) & (depth < 4095)
        z = np.where(valid, depth, 0)
        x = np.where(valid, z * (c - self.cx) / self.fx, 0)
        y = np.where(valid, z * (r - self.cy) / self.fy, 0)
        clouds = np.dstack((x, y, z)).reshape(-1, 3)
        return clouds

    def points_rotation(self, points):
        n_points, _ = points.shape
        t_points = np.hstack((points, np.ones((n_points, 1))))
        t_r_points = np.dot(self.af_mat, t_points.transpose()).transpose()
        return t_r_points[:, :3]

    def cloud2point(self, w_points, eps=1e-6):
        x = w_points[:, 0] * self.fx / (w_points[:, 2] + eps) + self.cx
        y = w_points[:, 1] * self.fy / (w_points[:, 2] + eps) + self.cy
        z = w_points[:, 2]
        s_points = np.round(np.stack((x, y, z), axis=1))

        width = self.width
        height = self.height

        valid = (s_points[:, 0] < width) & (s_points[:, 0] > 0) & \
                (s_points[:, 1] < height) & (s_points[:, 1] > 0) & \
                (s_points[:, 2] > 0)

        v_points = s_points[np.where(valid)]  #select the valid value
        v_points = v_points[np.argsort(v_points[:, 2])]  #order by depth value

        _, uq_idx = np.unique(v_points[:, 0]*1000.0 + v_points[:, 1], return_index=True)
        v_points = v_points[uq_idx]

        full_map = coo_matrix((v_points[:, 2], (v_points[:, 1], v_points[:, 0])), shape=(height, width)).toarray()

        return full_map

def add_gaussian_shifts(depth, mean=0, std=1):

    rows, cols = depth.shape 
    gaussian_shifts = np.random.normal(mean, std, size=(rows, cols, 2))
    gaussian_shifts = gaussian_shifts.astype(np.float32)

    # creating evenly spaced coordinates  
    xx = np.linspace(0, cols-1, cols)
    yy = np.linspace(0, rows-1, rows)

    # get xpixels and ypixels 
    xp, yp = np.meshgrid(xx, yy)

    xp = xp.astype(np.float32)
    yp = yp.astype(np.float32)

    depth_hole = depth == 0 
    depth_hole = depth_hole.astype(int)

    xp_interp = np.minimum(np.maximum(xp + gaussian_shifts[:, :, 0], 0.0), cols)
    yp_interp = np.minimum(np.maximum(yp + gaussian_shifts[:, :, 1], 0.0), rows)

    depth_hole = cv2.remap(depth_hole, xp_interp, yp_interp, cv2.INTER_NEAREST)
    depth_interp = depth * (1 - depth_hole)


    return depth_interp
    
