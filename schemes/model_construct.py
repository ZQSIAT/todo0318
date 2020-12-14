import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from utils.trivial_definition import separator_line
# from schemes.net_arch.shuttle_net import ShuttleNet as net_arch
# from schemes.net_arch import shuttle_net
# from utils.plot_utilities import plot_net_architecture
# from utils.log_recorder import net_arch_file_name


def constructing_model(args, config, logger):
    model_param = config.get_model_param()
    assert args.net_arch in model_param["net_arch"], "Unknown net_arch!"
    for metric in args.metrics:
        assert metric in model_param["metrics"], "Unknown metrics!"
    if args.adjust_lr is not None:
        assert args.adjust_lr in model_param["adjust_lr"], "Unknown adjust lr!"
    assert args.criterion in model_param["criterion"], "Unknown criterion!"
    assert args.optimizer in model_param["optimizer"], "Unknown optimizer!"

    # read the number of classes from previous generated protocol file
    num_classes = config.get_number_classes(args.dataset)

    # -------------------- net_arch --------------------------
    assert model_param["net_arch"][args.net_arch] is not None, "Invalid parameter!"

    net_arch_str = model_param["net_arch"][args.net_arch].split(".")

    arch_import_str = "from schemes.net_arch." + net_arch_str[0] + " import " + net_arch_str[1] + " as net_arch"

    exec(arch_import_str)

    # args.data_shape in format of [CxFxHxW] without batch_zise


    if len(args.data_shape) == 5:
        in_channel = args.data_shape[1]
        num_segments = args.data_shape[2]
    else:
        # in_channel = args.data_shape[1] #FxC
        # num_segments = args.data_shape[0]

        in_channel = args.data_shape[0] #CxF
        num_segments = args.data_shape[1]
    #
    # print(args.data_shape)
    #
    # print(in_channel)
    # print(num_segments)
    # raise RuntimeError

    net_arch_kwargs = "(num_classes={:d}, in_channel={:d}, num_segments={:d})".format(num_classes,
                                                                                      in_channel,
                                                                                      num_segments)

    net_func_string = "net_arch" + net_arch_kwargs
    model = eval(net_func_string)

    # -------------------- metrics --------------------------
    metrics = args.metrics

    # -------------------- criterion --------------------------
    assert model_param["criterion"][args.criterion] is not None, "Invalid parameter!"

    criterion_func_string = model_param["criterion"][args.criterion] + "()"
    criterion = eval(criterion_func_string)

    if not args.evaluate:

        # -------------------- optimizer --------------------------
        opt_param = model_param["optimizer"][args.optimizer]

        if "SGD" in args.optimizer:
            optimizer = optim.SGD(model.parameters(),
                                  lr=opt_param["lr"],
                                  momentum=opt_param["momentum"],
                                  weight_decay=opt_param["weight_decay"])
        elif "Adam" in args.optimizer:
            optimizer = optim.Adam(model.parameters(),
                                   lr=opt_param["lr"],
                                   weight_decay=opt_param["weight_decay"])
        else:
            raise NotImplementedError("Unknown optimizer!")
    else:
        optimizer = None

    if args.pretrained is not None:
        pretrained_model = torch.load(args.pretrained)
        pretrained_dict = pretrained_model["state_dict"]
        model_dict = model.state_dict()

        pretrained_dict_new = {}
        for k, v in pretrained_dict.items():
            kn = k.replace("module.", "")
            if kn in model_dict and "fc" not in kn:
                pretrained_dict_new[kn] = v

        model_dict.update(pretrained_dict_new)
        model.load_state_dict(model_dict)

        logger.info("=> using pre-trained model '{}' from '{:s}'".format(args.net_arch, args.pretrained))
        # raise RuntimeError
        # # freezing layers except the fc
        # for name, param in model.named_parameters():
        #     if "fc" in name:
        #         continue
        #     if "conv1" in name:
        #         continue
        #     param.requires_grad = False
        #     logger.info("--> freezing '{:s}'".format(name))


    # ------------------ plot and save model arch
    if args.plot_net_arch:
        input_size = args.data_shape
        # file_name = net_arch_file_name.format(args.log_name)
        # plot_net_architecture(model, input_size, file_name)

        dummy_input = torch.randn(tuple(input_size))
        dummy_input.unsqueeze_(0)
        logger.info("dummy input shape: {:s}".format(str(dummy_input.shape)))

        # if args.tensorboard:
        #     args.tb_writer.add_graph(model, (dummy_input,))

    # ----------------------- cuda transfer -----------------------
    if args.cuda:
        if len(args.gpu_card.split(",")) > 1:
            logger.info("=> Using mult GPU")
            model = torch.nn.DataParallel(model).cuda()
        else:
            logger.info("=> Using single GPU")
            model = model.cuda()

        criterion = criterion.cuda()
        cudnn.benchmark = True
        # cudnn.enabled = True
        # cudnn.deterministic = True

    # -----------------------------------------------------------------
    logger.info("model summary: ")

    logger.info(separator_line(dis_len="half"))
    logger.info("net_arch is: ")
    logger.info(model)

    logger.info(separator_line(dis_len="half"))
    net_parameter_statistics(logger, model.named_parameters())

    logger.info(separator_line(dis_len="half"))
    logger.info("metrics is: ")
    logger.info(metrics)

    logger.info(separator_line(dis_len="half"))
    logger.info("criterion is: ")
    logger.info(criterion)

    logger.info(separator_line(dis_len="half"))
    logger.info("optimizer is: ")
    logger.info(optimizer)
    logger.info(separator_line())

    return [model, metrics, criterion, optimizer]


def save_checkpoint(state, filename):
    """
    save_checkpoint
    :param state:
    :param filename:
    :return:
    """
    torch.save(state, filename)
    # shutil.copyfile(filename, 'model_best.pth.tar')


def net_parameter_statistics(logger, net_parameter_obeject):
    """net_parameter_statistics:
    :param logger:
    :param net_parameter_obeject:
    :return:
    """
    params = list(net_parameter_obeject)
    count_all = 0
    num_layers = 0
    for name, layer_i in params:
        layer_count = 1
        if "bn" in str(name):
            continue

        for j in layer_i.size():
            layer_count *= j

        if "conv" in str(name):
            num_layers += 1
        logger.info("{:s}  -- Parameter: {:s} count: {:d}".format(str(name),
                                                                   str(list(layer_i.size())),
                                                                   layer_count))
        count_all += layer_count
    logger.info("=> Convolution layers: {:d}".format(num_layers))
    logger.info("=> Trainable params: {:s}".format(format(count_all, ',')))

