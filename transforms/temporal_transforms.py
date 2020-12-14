import collections
import numpy as np
import math


class TemporalCompose(object):
    """Composes several transforms together.

    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, depth_seqs):
        for trans_method in self.transforms:
            if trans_method is None:
                continue
            depth_seqs = trans_method(depth_seqs)

        return depth_seqs

    def randomize_parameters(self):
        for trans_method in self.transforms:
            if trans_method is None:
                continue
            trans_method.randomize_parameters()


class DepthSeqsCoarseCrop(object):
    """Crop the given depth sequences with linear sampling.

    """
    def __init__(self, crop_length, crop_stride, crop_opts):
        assert isinstance(crop_length, int) and isinstance(crop_stride, int) \
               and crop_length > 0 and crop_stride > 0, "Invalid input data!"

        assert crop_opts in ["left", "center", "linear"], "Invalid input data!"

        self.crop_length = crop_length
        self.crop_stride = crop_stride
        self.crop_opts = crop_opts

    def __call__(self, depth_seqs):
        """
        :param depth_seqs: shape[(C x F x H x W)]
        :return:
        """
        length_original = depth_seqs.shape[1]
        assert isinstance(depth_seqs, collections.Iterable) \
               and len(depth_seqs.shape) == 4 and length_original > 1, "Invalid input data!"

        if length_original > self.crop_length * 1.5:
            depth_seqs = depth_seqs[:, ::self.crop_stride, :, :]

        length_original = depth_seqs.shape[1]

        if length_original > self.crop_length:
            if self.crop_opts == "left":
                depth_seqs = depth_seqs[:, :self.crop_length, :, :]

            elif self.crop_opts == "center":
                center_index = length_original // 2
                start_index = max(0, center_index - (self.crop_length // 2))
                end_index = min(start_index + self.crop_length, length_original)
                depth_seqs = depth_seqs[:, start_index:end_index, :, :]

            elif self.crop_opts == "linear":
                frame_idx = np.around(np.linspace(0, length_original - 1, num=self.crop_length)).astype(np.int)
                depth_seqs = depth_seqs[:, frame_idx, :, :]

            else:
                raise ValueError("Unknown crop methods!")

        return depth_seqs


class DepthSeqsCoarsePadding(object):
    """Padding the given depth sequences to uniform length.
    """
    def __init__(self, padding_length, padding_opts):
        assert isinstance(padding_length, int) and padding_length > 0, "Invalid input data!"

        assert padding_opts in ["loop", "last", "zero"], "Invalid input data!"

        self.padding_length = padding_length
        self.padding_opts = padding_opts

    def __call__(self, depth_seqs):
        """
        :param depth_seqs: shape[(C x F x H x W)]
        :return:
        """
        length_original = depth_seqs.shape[1]
        assert isinstance(depth_seqs, collections.Iterable) \
               and len(depth_seqs.shape) == 4 and length_original > 1, "Invalid input data!"

        if self.padding_length > length_original:
            if self.padding_opts == "loop":
                frame_idx = list(range(depth_seqs.shape[0]))

                for index in range(self.padding_length):
                    if len(frame_idx) >= self.padding_length:
                        break
                    frame_idx.append(index)

                depth_seqs = depth_seqs[:, frame_idx, :, :]

            elif self.padding_opts == "last":
                frame_idx = list(range(depth_seqs.shape[0]))

                for index in range(self.padding_length):
                    if len(frame_idx) >= self.padding_length:
                        break
                    frame_idx.append(frame_idx[depth_seqs.shape[0]-1])

                depth_seqs = depth_seqs[:, frame_idx, :, :]

            elif self.padding_opts == "zero":
                depth_seqs_out = np.zeros([depth_seqs.shape[0],
                                           self.padding_length,
                                           depth_seqs.shape[2],
                                           depth_seqs.shape[3]])
                depth_seqs_out[:, :length_original, :, :] = depth_seqs
                depth_seqs = depth_seqs_out

            else:
                raise ValueError("Unknown padding methods!")

        return depth_seqs


class DepthSeqsSnapshotSampling(object):
    """Snapshot sampling the sequences from given segments.
    """
    def __init__(self, segments, sampling_type):
        assert isinstance(segments, int) and segments > 0, "Invalid input data!"
        self.segments = segments
        self.sampling_type = sampling_type

    def __call__(self, depth_seqs):
        """
        :param depth_seqs: shape[(C x F x H x W)]
        :return:
        """
        length_original = depth_seqs.shape[1]
        assert isinstance(depth_seqs, collections.Iterable) \
               and len(depth_seqs.shape) == 4 and length_original > 1, "Invalid input data!"

        frame_idx = sparse_sampling_frames_from_segments(length_original,
                                                         self.segments,
                                                         self.sampling_type)

        depth_seqs = depth_seqs[:, frame_idx, :, :]

        return depth_seqs


class DepthSeqsSnapshotPooling(object):
    """Snapshot sampling the sequences from given segments.
    """
    def __init__(self, segments, pool_type, sampling_type="order"):
        assert isinstance(segments, int) and segments > 0, "Invalid input data!"
        self.segments = segments
        self.pool_type = pool_type
        self.sampling_type = sampling_type

    def __call__(self, depth_seqs):
        """
        :param depth_seqs: shape[(C x F x H x W)]
        :return:
        """
        length_original = depth_seqs.shape[1]
        assert isinstance(depth_seqs, collections.Iterable) \
               and len(depth_seqs.shape) == 4 and length_original > 1, "Invalid input data!"

        assert depth_seqs.shape[0] == 4, "Invalid input data!"

        if self.sampling_type == "random":
            frame_idx = sparse_random_pooling_frames_from_segments(length_original, self.segments)
        else:
            frame_idx = sparse_pooling_frames_from_segments(length_original, self.segments)

        depth_seqs_pooling = []
        if self.pool_type == "average":
            for i in frame_idx:
                depth_seqs_pooling.append(depth_seqs[0:3, i, :, :].mean(axis=1))

        elif self.pool_type == "max":
            for i in frame_idx:
                depth_seqs_pooling.append(depth_seqs[0:3, i, :, :].max(axis=1))

        elif self.pool_type == "norm_max":
            depth_gradient_norm = depth_seqs[3, :, :, :]
            print(depth_gradient_norm.shape)
            for i in frame_idx:
                print(i)
                idx = np.argmax(depth_gradient_norm[3], axis=0)
                print(idx.shape)
                print(idx)
                tartget_max = depth_seqs[0:3, :, idx]
                print(tartget_max.shape)
                depth_seqs_pooling.append(tartget_max)
        else:
            raise ValueError("Unknown pool type")

        depth_seqs_pooling = np.stack(depth_seqs_pooling, axis=1)
        return depth_seqs_pooling


class DepthSeqsGradient(object):
    def __init__(self, is_gradient, is_identity=False):
        self.is_gradient = is_gradient
        self.is_identity = is_identity

    def __call__(self, depth_seqs):
        """
        :param depth_seqs: shape[(C x F x H x W)]
        :return:
        """
        if depth_seqs.dtype != np.float:
            depth_seqs = depth_seqs.astype(np.float)

        if self.is_gradient:
            depth_seqs = depth_seqs[0, :, :, :]

            data_gradient = np.gradient(depth_seqs)

            [dt, dy, dx] = data_gradient
            d_norm = np.sqrt(np.square(dx) + np.square(dy) + np.square(dt) + 1)

            data_processed = np.stack([di / d_norm for di in data_gradient], axis=0)

            if self.is_identity:
                data_processed = np.concatenate([data_processed, d_norm[np.newaxis,:]], axis=0)


            #data_gradient = np.concatenate((data_gradient, np.ones(depth_seqs.shape)[np.newaxis,:]), axis=0)
            #data_gradient_norm = 1 / (np.sqrt(np.square(data_gradient).sum(axis=0, keepdims=True)))
            # old method
            # data_gradient = np.gradient(depth_seqs)
            # data_gradient_norm = 1 / (np.sqrt(np.square(data_gradient).sum(axis=0, keepdims=True)+1))
            # data_processed = data_gradient * data_gradient_norm.repeat(3, axis=0)
        else:
            data_processed = depth_seqs




        return data_processed


def sparse_pooling_frames_from_segments(length_original, segments):
    """
    :param length_original:
    :param segments:
    :return:
    """
    segments_list = np.array_split(np.arange(0, length_original), segments)
    if length_original > segments:
        frame_idx = segments_list

    else:
        frame_idx = []
        last_array = []
        for i, si in enumerate(segments_list):
            if len(si) > 0:
                frame_idx.append(si)
                last_array = si
            else:
                frame_idx.append(last_array)

    return frame_idx


def sparse_random_pooling_frames_from_segments(length_original, segments):
    """
    :param length_original:
    :param segments:
    :return:
    """
    if length_original - 1 > segments:
        s_points = np.random.choice(np.arange(1, length_original-1), segments-1, replace=False)
        s_points = np.sort(np.append(s_points, [0, length_original-1]))

        segments_list = []

        for ss in range(segments):
            a = list(range(s_points[ss], s_points[ss+1]+1))
            segments_list.append(a)
    else:
        frame_idx = list(range(length_original))

        segments_list = []
        i = 0
        for index in range(segments):
            segments_list.append(frame_idx[i])
            i += 1
            if i >= length_original:
                i = 0



    return segments_list


def sparse_sampling_frames_from_segments(length_original, segments, sampling_type="random"):
    """
    sampling_frames_from_segments
    :param length_original:
    :param segments:
    :return:
    """
    if length_original > segments:

        segments_list = np.array_split(np.arange(0, length_original), segments)
        if sampling_type == "order":
            frame_idx = [si[0] for si in segments_list]
        else:
            frame_idx = [np.random.choice(si, 1)[0] for si in segments_list]

    else:

        frame_idx = list(range(length_original))

        for index in range(segments - length_original):
            if len(frame_idx) >= segments:
                break
            frame_idx.append(index)

    return frame_idx


def sparse_sampling_frames_from_segments_dual(length_original, segments, sampling_type="random", offset=2):
    """
    sampling_frames_from_segments
    :param length_original:
    :param segments:
    :return:
    """
    length_original = length_original - offset
    if length_original > segments:
        segments_list = np.array_split(np.arange(0, length_original), segments)
        if sampling_type == "order":
            frame_idx = []
            for si in segments_list:
                sidx = len(si)//2
                frame_idx.append(si[sidx])
                frame_idx.append(si[sidx] + offset)
        else:
            frame_idx = []
            for si in segments_list:
                pre_idx = np.random.choice(si, 1)[0]
                frame_idx.append(pre_idx)
                frame_idx.append(pre_idx + offset)
    else:

        frame_idx = list(range(length_original))

        for index in range(segments - length_original):
            if len(frame_idx) >= segments:
                break
            frame_idx.append(index)

    return frame_idx



def dense_sampling_frames_from_segments(length_original, segments):
    """
    dense_sampling_frames_from_segments
    :param length_original:
    :param segments:
    :return:
    """
    if length_original > segments:

        segments_list = np.array_split(np.arange(0, length_original), segments)
        frame_idx = [np.random.choice(si, 1)[0] for si in segments_list]

    else:
        frame_idx = list(range(length_original))

        for index in range(segments - length_original):
            if len(frame_idx) >= segments:
                break
            frame_idx.append(index)

    return frame_idx




if __name__ == '__main__':
    depth_seqs = np.random.randint(0, 4095, size=[4, 30, 240, 320])

    # expected_len = 20
    # stride = 2

    """
    temporal_transform = TemporalCompose([DepthSeqsCoarseCrop(expected_len, stride, "linear"),
                                          DepthSeqsCoarsePadding(expected_len, "last")])
    depth_seqs = temporal_transform.__call__(depth_seqs)

    print(depth_seqs.shape)
    """

    # length = depth_seqs.shape[1]

    # segments = 7
    #
    # pooling = DepthSeqsGradient(segments)
    #
    # depth_seqs = pooling(depth_seqs)
    #
    # print(depth_seqs.shape)
    # segments = 7
    # sampling_type = "average"
    # sampling = DepthSeqsSnapshotPooling(segments, sampling_type, "random")
    #
    # depth_seqs = sampling(depth_seqs)
    #
    # print(depth_seqs.shape)
    a = sparse_sampling_frames_from_segments_dual(20, 7, "order")
    print(a)
    #
    # a = sparse_sampling_frames_from_segments(3, 7)
    # #np.sort(np.random.choice(np.arange(10), 7, replace=False))
    # print(a)





