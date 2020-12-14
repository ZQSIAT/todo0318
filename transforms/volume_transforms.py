from transforms import spatial_transforms as spatial
from transforms import temporal_transforms as temporal
import torchvision.transforms as transforms


def volume_data_transforms(spatial_param, temporal_param, modality, is_gradient):
    """
    volume_data_transforms
    :param spatial_param:
    :param temporal_param:
    :param spatial_method:
    :param temporal_method:
    :param modality:
    :return:
    """

    if modality == "Depth":
        # ---------- configure spatial transform ----------

        if "combine_views" in spatial_param.keys():
            rotation_method = spatial.DepthMapsCombineRotationView(alphas=spatial_param["combine_views"]["alphas"],
                                                               betas=spatial_param["combine_views"]["betas"],
                                                               dz=spatial_param["combine_views"]["dz"])

        elif "random_rotation" in spatial_param.keys():
            rotation_method = spatial.DepthMapsRandomRotation(spatial_param["random_rotation"]["alpha"],
                                                              spatial_param["random_rotation"]["beta"],
                                                              spatial_param["random_rotation"]["p"],
                                                              spatial_param["random_rotation"]["dz"])
        else:
            rotation_method = None

        if "resize_shape" in spatial_param.keys():
            resize_method = spatial.DepthMapsResize(spatial_param["resize_shape"]["size"])
        else:
            resize_method = None

        if "random_crop" in spatial_param.keys():
            crop_method = spatial.DepthMapsSpatialRandomCrop(size=spatial_param["random_crop"]["size"],
                                                             padding=spatial_param["random_crop"]["padding"])
        elif "center_crop" in spatial_param.keys():
            crop_method = spatial.DepthMapsSpatialCenterCrop(size=spatial_param["center_crop"]["size"])
        else:
            crop_method = None

        if "random_horizontal_flip" in spatial_param.keys():
            flip_method = spatial.DepthMapsSpatialRandomHorizontalFlip(spatial_param["random_horizontal_flip"])
        else:
            flip_method = None

        if "normalization" in spatial_param.keys():
            norm_value = spatial_param["normalization"]
        else:
            norm_value = 1.0

        totensor_method = spatial.DepthMapsNormalizedToTensor(norm_value)

        if "standardization" in spatial_param.keys():
            standard_method = spatial.DepthMapsStandardization(spatial_param["standardization"]["mean"],
                                                               spatial_param["standardization"]["std"])
        else:
            standard_method = None

        spatial_transform = spatial.SpatialCompose([rotation_method,
                                                    resize_method,
                                                    crop_method,
                                                    flip_method,
                                                    totensor_method,
                                                    standard_method])

        temporal_transform = None
        # # ---------- configure temporal transform ----------
        #
        # if "snapshot_sampling" in temporal_param.keys():
        #
        #     if "random_rotation" in spatial_param.keys() and is_gradient:
        #         rotation_method = spatial.DepthMapsRandomRotation(spatial_param["random_rotation"]["alpha"],
        #                                                           spatial_param["random_rotation"]["beta"],
        #                                                           spatial_param["random_rotation"]["p"],
        #                                                           spatial_param["random_rotation"]["dz"])
        #     else:
        #         rotation_method = None
        #
        #     if "resize_shape" in spatial_param.keys() and is_gradient:
        #         resize_method = spatial.DepthMapsResize(spatial_param["resize_shape"]["size"])
        #     else:
        #         resize_method = None
        #
        #     if is_gradient:
        #         gradient_method = temporal.DepthSeqsGradient(True)
        #     else:
        #         gradient_method = None
        #
        #     sampling_method = temporal.DepthSeqsSnapshotSampling(temporal_param["snapshot_sampling"]["segments"],
        #                                                          temporal_param["snapshot_sampling"]["sampling_type"])
        #
        #     temporal_transform = temporal.TemporalCompose([rotation_method,
        #                                                    resize_method,
        #                                                    gradient_method,
        #                                                    sampling_method])
        # elif "snapshot_pooling" in temporal_param.keys():
        #
        #     if "random_rotation" in spatial_param.keys() and is_gradient:
        #         rotation_method = spatial.DepthMapsRandomRotation(spatial_param["random_rotation"]["alpha"],
        #                                                           spatial_param["random_rotation"]["beta"],
        #                                                           spatial_param["random_rotation"]["p"],
        #                                                           spatial_param["random_rotation"]["dz"])
        #     else:
        #         rotation_method = None
        #
        #     if "resize_shape" in spatial_param.keys() and is_gradient:
        #         resize_method = spatial.DepthMapsResize(spatial_param["resize_shape"]["size"])
        #     else:
        #         resize_method = None
        #
        #     if is_gradient:
        #         gradient_method = temporal.DepthSeqsGradient(True, is_identity=True)
        #     else:
        #         gradient_method = None
        #
        #     poolling_method = temporal.DepthSeqsSnapshotPooling(temporal_param["snapshot_pooling"]["segments"],
        #                                                         temporal_param["snapshot_pooling"]["pool_type"],
        #                                                         temporal_param["snapshot_pooling"]["sampling_type"])
        #
        #     temporal_transform = temporal.TemporalCompose([rotation_method,
        #                                                    resize_method,
        #                                                    gradient_method,
        #                                                    poolling_method])
        #
        # elif "single_gradient" in temporal_param.keys():
        #     if is_gradient:
        #         gradient_method = temporal.DepthSeqsGradient(True)
        #     else:
        #         gradient_method = None
        #     temporal_transform = temporal.TemporalCompose([gradient_method])
        #
        # else:
        #     raise ValueError("Unknown temporal transform method!")

    elif modality == "Pre_gradient":
        # ---------- configure spatial transform ----------

        if "combine_views" in spatial_param.keys():
            rotation_method = spatial.DepthMapsCombineRotationView(alphas=spatial_param["combine_views"]["alphas"],
                                                               betas=spatial_param["combine_views"]["betas"],
                                                               dz=spatial_param["combine_views"]["dz"])

        elif "random_rotation" in spatial_param.keys() and not is_gradient:
            rotation_method = spatial.DepthMapsRandomRotation(spatial_param["random_rotation"]["alpha"],
                                                              spatial_param["random_rotation"]["beta"],
                                                              spatial_param["random_rotation"]["p"],
                                                              spatial_param["random_rotation"]["dz"])

        else:
            rotation_method = None

        if "resize_shape" in spatial_param.keys() and not is_gradient:
            resize_method = spatial.DepthMapsResize(spatial_param["resize_shape"]["size"])
        else:
            resize_method = None

        # if "resize_shape" in spatial_param.keys():
        #     resize_method = spatial.DepthMapsResize(spatial_param["resize_shape"]["size"])
        # else:
        #     resize_method = None
        # resize_method = None

        if "random_crop" in spatial_param.keys():
            crop_method = spatial.DepthMapsSpatialRandomCrop(size=spatial_param["random_crop"]["size"],
                                                             padding=spatial_param["random_crop"]["padding"])
        elif "center_crop" in spatial_param.keys():
            crop_method = spatial.DepthMapsSpatialCenterCrop(size=spatial_param["center_crop"]["size"])
        else:
            crop_method = None

        if "random_horizontal_flip" in spatial_param.keys():
            flip_method = spatial.DepthMapsSpatialRandomHorizontalFlip(spatial_param["random_horizontal_flip"])
        else:
            flip_method = None

        if "normalization" in spatial_param.keys():
            norm_value = spatial_param["normalization"]
        else:
            norm_value = 1.0

        totensor_method = spatial.DepthMapsNormalizedToTensor(norm_value)

        if "standardization" in spatial_param.keys():
            standard_method = spatial.DepthMapsStandardization(spatial_param["standardization"]["mean"],
                                                               spatial_param["standardization"]["std"])
        else:
            standard_method = None

        spatial_transform = spatial.SpatialCompose([rotation_method,
                                                    resize_method,
                                                    crop_method,
                                                    flip_method,
                                                    totensor_method,
                                                    standard_method])

        # ---------- configure temporal transform ----------
        if "clips" in temporal_param.keys():

            max_durations = temporal_param["max_durations"]
            exceed_crop = temporal_param["exceed_crop"]
            crop_stride = temporal_param["crop_stride"]
            lack_padding = temporal_param["lack_padding"]

            crop_method = temporal.DepthSeqsCoarseCrop(max_durations, crop_stride, exceed_crop)
            padding_method = temporal.DepthSeqsCoarsePadding(max_durations, lack_padding)

            temporal_transform = temporal.TemporalCompose([crop_method, padding_method])

        elif "snapshot_sampling" in temporal_param.keys():

            if "random_rotation" in spatial_param.keys() and is_gradient:
                rotation_method = spatial.DepthMapsRandomRotation(spatial_param["random_rotation"]["alpha"],
                                                                  spatial_param["random_rotation"]["beta"],
                                                                  spatial_param["random_rotation"]["p"],
                                                                  spatial_param["random_rotation"]["dz"])
            else:
                rotation_method = None

            if "resize_shape" in spatial_param.keys() and is_gradient:
                resize_method = spatial.DepthMapsResize(spatial_param["resize_shape"]["size"])
            else:
                resize_method = None

            if is_gradient:
                gradient_method = temporal.DepthSeqsGradient(True)
            else:
                gradient_method = None

            sampling_method = temporal.DepthSeqsSnapshotSampling(temporal_param["snapshot_sampling"]["segments"],
                                                                 temporal_param["snapshot_sampling"]["sampling_type"])

            temporal_transform = temporal.TemporalCompose([rotation_method,
                                                           resize_method,
                                                           gradient_method,
                                                           sampling_method])
        elif "snapshot_pooling" in temporal_param.keys():

            if "random_rotation" in spatial_param.keys() and is_gradient:
                rotation_method = spatial.DepthMapsRandomRotation(spatial_param["random_rotation"]["alpha"],
                                                                  spatial_param["random_rotation"]["beta"],
                                                                  spatial_param["random_rotation"]["p"],
                                                                  spatial_param["random_rotation"]["dz"])
            else:
                rotation_method = None

            if "resize_shape" in spatial_param.keys() and is_gradient:
                resize_method = spatial.DepthMapsResize(spatial_param["resize_shape"]["size"])
            else:
                resize_method = None

            if is_gradient:
                gradient_method = temporal.DepthSeqsGradient(True, is_identity=True)
            else:
                gradient_method = None

            poolling_method = temporal.DepthSeqsSnapshotPooling(temporal_param["snapshot_pooling"]["segments"],
                                                                temporal_param["snapshot_pooling"]["pool_type"],
                                                                temporal_param["snapshot_pooling"]["sampling_type"])

            temporal_transform = temporal.TemporalCompose([rotation_method,
                                                           resize_method,
                                                           gradient_method,
                                                           poolling_method])

        elif "single_gradient" in temporal_param.keys():
            if is_gradient:
                gradient_method = temporal.DepthSeqsGradient(True)
            else:
                gradient_method = None
            temporal_transform = temporal.TemporalCompose([gradient_method])

        else:
            raise ValueError("Unknown temporal transform method!")

    elif modality == "RGB":
        # ---------- configure spatial transform ----------

        resize_method = None

        if "random_crop" in spatial_param.keys():
            crop_method = spatial.DepthMapsSpatialRandomCrop(size=spatial_param["random_crop"]["size"],
                                                             padding=spatial_param["random_crop"]["padding"])
        elif "center_crop" in spatial_param.keys():
            crop_method = spatial.DepthMapsSpatialCenterCrop(size=spatial_param["center_crop"]["size"])
        else:
            crop_method = None

        if "random_horizontal_flip" in spatial_param.keys():
            flip_method = spatial.DepthMapsSpatialRandomHorizontalFlip(spatial_param["random_horizontal_flip"])
        else:
            flip_method = None

        if "normalization" in spatial_param.keys():
            norm_value = spatial_param["normalization"]
        else:
            norm_value = 1.0

        totensor_method = spatial.DepthMapsNormalizedToTensor(norm_value)

        if "standardization" in spatial_param.keys():
            standard_method = spatial.DepthMapsStandardization(spatial_param["standardization"]["mean"],
                                                               spatial_param["standardization"]["std"])
        else:
            standard_method = None

        spatial_transform = spatial.SpatialCompose([resize_method,
                                                    crop_method,
                                                    flip_method,
                                                    totensor_method,
                                                    standard_method])

        # ---------- configure temporal transform ----------
        if "clips" in temporal_param.keys():

            max_durations = temporal_param["max_durations"]
            exceed_crop = temporal_param["exceed_crop"]
            crop_stride = temporal_param["crop_stride"]
            lack_padding = temporal_param["lack_padding"]

            crop_method = temporal.DepthSeqsCoarseCrop(max_durations, crop_stride, exceed_crop)
            padding_method = temporal.DepthSeqsCoarsePadding(max_durations, lack_padding)

            temporal_transform = temporal.TemporalCompose([crop_method, padding_method])

        elif "snapshot_sampling" in temporal_param.keys():
            if "resize_shape" in spatial_param.keys():
                resize_method = spatial.DepthMapsResize(spatial_param["resize_shape"]["size"])
            else:
                resize_method = None

            if is_gradient:
                gradient_method = temporal.DepthSeqsGradient(True)
            else:
                gradient_method = None


            sampling_method = temporal.DepthSeqsSnapshotSampling(temporal_param["snapshot_sampling"]["segments"],
                                                                 temporal_param["snapshot_sampling"]["sampling_type"])

            temporal_transform = temporal.TemporalCompose([resize_method,
                                                           gradient_method,
                                                           sampling_method])

        elif "single_gradient" in temporal_param.keys():
            if is_gradient:
                gradient_method = temporal.DepthSeqsGradient(True)
            else:
                gradient_method = None
            temporal_transform = temporal.TemporalCompose([gradient_method])

        else:
            raise ValueError("Unknown temporal transform method!")

    else:
        raise NotImplementedError("{:s}: has not been implemented!".format(modality))

    return spatial_transform, temporal_transform



