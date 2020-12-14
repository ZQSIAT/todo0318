import os
import json
import argparse


# default option file
option_dir = os.path.dirname(os.path.realpath(__file__))
args_option_default = option_dir + "/args_option_default.json"


def loading_args_option_from_file(args_option_file=args_option_default):
    """
    loading_args_option_from_file:
    :param args_option_file:
    :return args_option_param:
    """
    args_outs_stream = ""
    if args_option_file is None:
        args_option_file = args_option_default
        args_outs_stream += "Using default args option in '{:s}'\n".format(args_option_file)
    else:
        args_outs_stream += "Using custom args option in '{:s}'\n".format(args_option_file)

    assert os.path.isfile(args_option_file), \
        "Unknown option files in {:s}!".format(args_option_file)

    with open(args_option_file, "r") as aof:
        args_option_param = json.load(aof)
        return args_option_param, args_option_file, args_outs_stream


def set_argparse_opts(args):
    """
    set_argparse_opts
    :param args_option_file:
    :return:
    """
    args_option_file = args.option
    resume_file = args.resume
    code = args.code
    pretrained = args.pretrained
    evaluate = args.evaluate

    args_option_param, args_option_file, args_outs_stream = loading_args_option_from_file(args_option_file)

    # ----------------------------------------------------------
    parser = argparse.ArgumentParser()

    # -------------------- option_file --------------------
    parser.add_argument("--option", default=args_option_file,
                        type=str, help="Args option file")

    parser.add_argument('--evaluate', default=evaluate, action='store_true',
                        help="evaluate model from checkpoint")

    # -------------------- dataset_args --------------------
    dataset_args = args_option_param["dataset"]

    parser.add_argument("--dataset", default=dataset_args["data_name"],
                        type=str, help="Evaluated dataset")

    # parser.add_argument("--data_shape", default=dataset_args["data_shape"],
    #                     type=int, help="Shape of input data")

    parser.add_argument("--data_dosage", default=dataset_args["data_dosage"],
                        type=float, help="The dosage of dataset for training")

    parser.add_argument('--small_validation', default=dataset_args["small_validation"],
                        action="store_false", help="using small samples for validation")

    parser.add_argument('--modality', default=dataset_args["modality"],
                        type=str, choices=["Depth", "Skeleton", 'RGB'])

    parser.add_argument('--is_gradient', default=dataset_args["is_gradient"],
                        action="store_false", help="using the gradient as input")

    parser.add_argument('--add_noise', default=dataset_args["add_noise"],
                        action="store_false", help="add noise to depth maps")

    parser.add_argument('--is_gradient_normal', default=dataset_args["is_gradient_normal"],
                        action="store_false", help="dogv normalization")

    parser.add_argument('--use_depth_seq', default=dataset_args["use_depth_seq"],
                        action="store_false", help="dogv normalization")

    # parser.add_argument('--is_vector', default=dataset_args["is_vector"],
    #                     action="store_false", help="using the vector as input")

    # parser.add_argument('--dual_input', default=dataset_args["dual_input"],
    #                     action="store_false", help="using dual input")

    # parser.add_argument('--is_cotrain', default=dataset_args["is_cotrain"],
    #                     action="store_false", help="co train")

    parser.add_argument('--is_segmented', default=dataset_args["is_segmented"],
                        action="store_false", help="using segmented maps")

    parser.add_argument('--eval_protocol', default=dataset_args["eval_protocol"],
                        type=str, help="Evaluation protocol")

    parser.add_argument('--regenerate_protocol_files', default=dataset_args["regenerate_protocol_files"],
                        action="store_false", help="Regenerate the protocol files")

    parser.add_argument("--batch_size", default=dataset_args["batch_size"],
                        type=int, help="Batch size")

    parser.add_argument("--train_shuffle", default=dataset_args["train_shuffle"],
                        action="store_false", help="Shuffle")

    parser.add_argument('--spatial_transform', default=dataset_args["spatial_transform"],
                        type=str, help="Employed spatial transform method")

    parser.add_argument('--temporal_transform', default=dataset_args["temporal_transform"],
                        type=str, help="Employed temporal transform method")

    # -------------------- cuda_args --------------------
    cuda_args = args_option_param["cuda"]

    parser.add_argument("--no-cuda", default=cuda_args["disable"],
                        action="store_true", help="Disables CUDA training")

    parser.add_argument("--gpu_card", default=cuda_args["gpu_card"],
                        type=str, help="visible gou card")

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_args["gpu_card"]
    args_outs_stream += ("=> executing: os.environ['CUDA_VISIBLE_DEVICES']='{:s}'\n".format(cuda_args["gpu_card"]))

    num_workers = len(cuda_args["gpu_card"].split(",")) * 4
    parser.add_argument("--num_workers", default=num_workers,
                        type=int, help="Number of data loading workers")

    parser.add_argument("--pin_memory", default=cuda_args["pin_memory"],
                        action="store_false", help="Memory allocated to transfer data to the GPU")

    # random seed
    parser.add_argument("--manual_seed", default=args_option_param["random_seed"],
                        type=int, help='Manually set random seed')

    # -------------------- model --------------------
    model_args = args_option_param["model"]

    parser.add_argument("--net_arch", default=model_args["net_arch"],
                        type=str, help="network architecture")

    parser.add_argument("--optimizer", default=model_args["optimizer"],
                        type=str, help="optimizer option")

    parser.add_argument("--adjust_lr", default=model_args["adjust_lr"],
                        action="store_false", help="adjust the learning rate")

    parser.add_argument("--criterion", default=model_args["criterion"],
                        type=str, help="criterion option")

    parser.add_argument("--metrics", default=model_args["metrics"],
                        type=str, help="metrics option")

    # -------------------- train --------------------
    train_args = args_option_param["train"]

    parser.add_argument('--resume', default=resume_file, metavar='PATH',
                        type=str, help="resume file path (default: None)")

    parser.add_argument('--pretrained', default=pretrained, metavar='PATH',
                        type=str, help="resume file path (default: None)")

    parser.add_argument('--epochs', default=train_args["epochs"], type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--log_interval', default=train_args["log_interval"], type=int,
                        metavar='N', help='frequency of logging training status (default: 10)')

    parser.add_argument('--checkpoint_interval', default=train_args["checkpoint_interval"],
                        type=int, metavar='N', help='checkpoint frequency (default: 10)')

    parser.add_argument('--overfit_threshold', default=train_args["overfit_threshold"], type=float,
                        metavar='N', help='over fitting threshold')

    parser.add_argument("--mail_notification", default=train_args["mail_notification"],
                        action="store_false", help="Result notification by email")

    parser.add_argument("--plot_net_arch", default=train_args["plot_net_arch"],
                        action="store_false", help="Plot the net architecture")

    parser.add_argument("--plot_confusion_matrix", default=train_args["plot_confusion_matrix"],
                        action="store_false", help="Plot confusion matrix")

    parser.add_argument("--tensorboard", default=train_args["tensorboard"],
                        action="store_false", help="Monitor by tensorboard")

    if code is None:
        code = train_args["default_code"]
    parser.add_argument('--code', default=code, type=str,
                        help="identify code for another checkpoint")

    # ----------------------------------------------------------

    args = parser.parse_args()

    return args, args_outs_stream
