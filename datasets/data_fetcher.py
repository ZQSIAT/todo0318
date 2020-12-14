import json
import torch
from utils.trivial_definition import separator_line
from datasets.MSRDailyAct3D import MSRDailyAct3D
from datasets.MSRAction3D import MSRAction3D
from datasets.NTU_RGBD import NTURGBD
from utils.generate_protocol_files import generate_protocol_files
from transforms.volume_transforms import volume_data_transforms
from transforms.group_transforms import group_data_transforms


def get_local_protocol_config(data_name, config, logger, regenerate_protocol_files=False):
    """
    get_dataset_config:
    :param data_name:
    :param config:
    :param regenerate_protocol_files:
    :return:
    """
    dataset_param = config.get_dataset_param(data_name)

    # regenerate the protocol files
    if regenerate_protocol_files:
        protocol_config_file = generate_protocol_files(data_name, dataset_param, logger)
    else:
        protocol_config_file = dataset_param["eval_config_file"]
        logger.info("Employ protocol from '{:s}'".format(protocol_config_file))
        logger.info(separator_line())

    # read protocol params
    with open(protocol_config_file, "r") as pcf:
        protocol_config = json.load(pcf)

    return protocol_config


def fetching_dataset(args, config, logger, subset):
    """
    fetching_dataset:
    :param args:
    :param config:
    :param subset:
    :return:
    """
    assert args.dataset in config.get_defined_datasets_list(), "Unknown dataset: {:s}".format(args.dataset)

    if args.regenerate_protocol_files:
        protocol_config = get_local_protocol_config(args.dataset, config, logger, subset == "train")
    else:
        protocol_config = get_local_protocol_config(args.dataset, config, logger)

    index_param = {}

    index_param["file_format"] = protocol_config["file_format"][args.modality]
    index_param["file_subset_list"] = protocol_config["eval_Protocol"][args.eval_protocol][subset]
    index_param["subset_type"] = subset
    index_param["small_validation"] = args.small_validation
    index_param["modality"] = args.modality
    index_param["data_dosage"] = args.data_dosage
    index_param["is_gradient"] = args.is_gradient
    index_param["is_segmented"] = args.is_segmented
    index_param["logger"] = logger
    # index_param["dual_input"] = args.dual_input

    index_param["add_noise"] = args.add_noise
    # index_param["is_gradient_normal"] = args.is_gradient_normal
    index_param["use_depth_seq"] = args.use_depth_seq

    # configure transforms
    loader_param = config.get_loader_param(args.dataset, args.modality)

    spatial_param = loader_param["spatial_transform"][args.spatial_transform][subset]
    temporal_param = loader_param["temporal_transform"][args.temporal_transform][subset]

    index_param["temporal_param"] = temporal_param


    spatial_transform, temporal_transform = group_data_transforms(spatial_param,
                                                                  temporal_param,
                                                                  args.modality,
                                                                  args.is_gradient,
                                                                  args.is_gradient_normal,
                                                                  args.use_depth_seq)

    if args.dataset == 'MSRDailyAct3D':
        target_dataset = MSRDailyAct3D(index_param,
                                       spatial_transform=spatial_transform,
                                       temporal_transform=temporal_transform)

    elif args.dataset == "MSRAction3D":
        target_dataset = MSRAction3D(index_param,
                                       spatial_transform=spatial_transform,
                                       temporal_transform=temporal_transform)

    elif args.dataset == "NTU_RGB+D":
        target_dataset = NTURGBD(index_param,
                                       spatial_transform=spatial_transform,
                                       temporal_transform=temporal_transform)

    else:
        raise ValueError("Unknown dataset: '{:s}'".format(args.dataset))

    logger.info("[{:s}] spatial parameters of {:s} are: ".format(subset,
                                                                 args.spatial_transform))
    logger.info(json.dumps(spatial_param, indent=4))
    logger.info("[{:s}] temporal parameters of {:s} are: ".format(subset,
                                                                  args.temporal_transform))
    logger.info(json.dumps(temporal_param, indent=4))

    logger.info(separator_line())

    if subset == "train":
        data_to_fetch = torch.utils.data.DataLoader(target_dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=args.train_shuffle,
                                                    num_workers=args.num_workers,
                                                    pin_memory=args.pin_memory)
    else:
        data_to_fetch = torch.utils.data.DataLoader(target_dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    num_workers=args.num_workers,
                                                    pin_memory=args.pin_memory)

    if args.plot_confusion_matrix:
        data_to_fetch.class_names = protocol_config["action_Names"]
    sample_data, _, _ = target_dataset.__getitem__(0)


    return data_to_fetch, sample_data.shape

