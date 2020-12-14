import numpy as np 
import cv2 
import os
import ipdb
from scipy.interpolate import griddata
from scipy import ndimage


import numpy as np 
np.set_printoptions(suppress=True)
import cv2 

from scipy.interpolate import griddata
from scipy import ndimage


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
    

def filterDisp(disp, dot_pattern_, invalid_disp_):

    size_filt_ = 9

    xx = np.linspace(0, size_filt_-1, size_filt_)
    yy = np.linspace(0, size_filt_-1, size_filt_)

    xf, yf = np.meshgrid(xx, yy)

    xf = xf - int(size_filt_ / 2.0)
    yf = yf - int(size_filt_ / 2.0)

    sqr_radius = (xf**2 + yf**2)
    vals = sqr_radius * 1.2**2 

    vals[vals==0] = 1 
    weights_ = 1 /vals  

    fill_weights = 1 / ( 1 + sqr_radius)
    fill_weights[sqr_radius > 9] = -1.0 

    disp_rows, disp_cols = disp.shape 
    dot_pattern_rows, dot_pattern_cols = dot_pattern_.shape

    lim_rows = np.minimum(disp_rows - size_filt_, dot_pattern_rows - size_filt_)
    lim_cols = np.minimum(disp_cols - size_filt_, dot_pattern_cols - size_filt_)

    center = int(size_filt_ / 2.0)

    window_inlier_distance_ = 1

    out_disp = np.ones_like(disp) * invalid_disp_

    interpolation_map = np.zeros_like(disp)

    for r in range(0, lim_rows):

        for c in range(0, lim_cols):

            if dot_pattern_[r+center, c+center] > 0:
                                
                # c and r are the top left corner 
                window  = disp[r:r+size_filt_, c:c+size_filt_] 
                dot_win = dot_pattern_[r:r+size_filt_, c:c+size_filt_] 
  
                valid_dots = dot_win[window < invalid_disp_]

                n_valids = np.sum(valid_dots) / 255.0 
                n_thresh = np.sum(dot_win) / 255.0 

                # ipdb.set_trace()

                if n_valids > n_thresh / 1.2: 

                    mean = np.mean(window[window < invalid_disp_])

                    diffs = np.abs(window - mean)
                    diffs = np.multiply(diffs, weights_)

                    cur_valid_dots = np.multiply(np.where(window<invalid_disp_, dot_win, 0), 
                                                 np.where(diffs < window_inlier_distance_, 1, 0))

                    n_valids = np.sum(cur_valid_dots) / 255.0

                    if n_valids > n_thresh / 1.2: 
                    
                        accu = window[center, center] 

                        assert(accu < invalid_disp_)

                        out_disp[r+center, c + center] = round((accu)*8.0) / 8.0

                        interpolation_window = interpolation_map[r:r+size_filt_, c:c+size_filt_]
                        disp_data_window     = out_disp[r:r+size_filt_, c:c+size_filt_]

                        substitutes = np.where(interpolation_window < fill_weights, 1, 0)
                        interpolation_window[substitutes==1] = fill_weights[substitutes ==1 ]

                        disp_data_window[substitutes==1] = out_disp[r+center, c+center]

    return out_disp



if __name__ == "__main__":

    eg_depth = "/data/jixiaopeng/dataset/ntu_rgb+d/depth/S012C001P017R002A010/"  # /data1/lhj/_zqs/S021C001P060R001A080
    dot_pattern_ = cv2.imread("./kinect-pattern_3x3.png", 0)
    list_dir = os.listdir(eg_depth)
    # count = 50

    # various variables to handle the noise modelling
    scale_factor = 100  # converting depth from m to cm
    # focal_length = 5800  # focal length of the camera used 480
    # baseline_m = 0.0  # baseline in m 0.075
    invalid_disp_ = 99999999.9

    for i, ii in enumerate(list_dir):

        depth_ii = eg_depth + ii
        

        depth_uint16 = cv2.imread(depth_ii, cv2.IMREAD_UNCHANGED)
        h, w = depth_uint16.shape

        # Our depth images were scaled by 5000 to store in png format so dividing to get
        # depth in meters
        # depth = depth_uint16.astype('float') / 1000.0
        depth_interp = add_gaussian_shifts(depth, mean=0, std=1)

        # disp_ = focal_length * baseline_m / (depth_interp + 1e-10)
        # depth_f = np.round(disp_ * 8.0) / 8.0 # 8.0

        # # ipdb.set_trace()
        # out_disp = filterDisp(depth_f, dot_pattern_, invalid_disp_)

        # depth = focal_length * baseline_m / out_disp
        # depth[out_disp == invalid_disp_] = 1
        # depth = depth_interp

        # The depth here needs to converted to cms so scale factor is introduced 
        # though often this can be tuned from [100, 200] to get the desired banding / quantisation effects 
        # import ipdb; ipdb.set_trace()
        # noisy_depth = (35130/np.round((35130/np.round(depth*scale_factor)) + np.random.normal(size=(h, w))*(10.0/6.0) + 0.5))/scale_factor 
        # noisy_depth = (35130/np.round((35130/np.round(depth*scale_factor + 1)) + np.random.normal(size=(h, w))*(1.0/6.0)))/scale_factor 
        noisy_depth = depth_interp
        # noisy_depth = noisy_depth * 1000.0
        noisy_depth = noisy_depth.astype('uint16')
        # Displaying side by side the orignal depth map and the noisy depth map with barron noise cvpr 2013 model
        # cv2.namedWindow('Adding Kinect Noise', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('Adding Kinect Noise', np.hstack((depth_uint16, noisy_depth)))


        depth2save = np.hstack((depth_uint16, noisy_depth))

        depth2save = depth2save / depth2save.max() * 255.0 

        depth2save.astype('uint8')



        cv2.imwrite('./noised_{}'.format(ii), depth2save)
        print("{} has add noise!!".format(ii))
        import matplotlib
        matplotlib.use('Qt5Agg')
        from matplotlib import pyplot as plt
        # plt.switch_backend('Tkagg')
        # plt.imshow(depth2save)
        # plt.show()
        # cv2.imshow('Adding Kinect Noise', depth2save)
        # key = cv2.waitKey(1)

        # # Press esc or 'q' to close the image window
        # if key & 0xFF == ord('q') or key == 27:
        #     cv2.destroyAllWindows()
        #     break

        break
        pass

