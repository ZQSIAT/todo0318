import argparse
import torch
import numpy as np
import datetime
from options.args_option import set_argparse_opts
from datasets.data_fetcher import fetching_dataset
from configs.param_config import ConfigClass
from utils.trivial_definition import separator_line
from utils.trivial_definition import args_format
from utils.trivial_definition import tb_symbolic_link
from schemes.model_construct import constructing_model
from schemes.model_train import training_model
from utils.log_recorder import verbose_logs, shutil_logs
from utils.mail_notify import send_mail_notification
from tensorboardX import SummaryWriter
from schemes.model_test import test_model

def main(args, config, logger):
    start_time = datetime.datetime.now()

    if args.evaluate:
        if args.resume is None:
            logger.error("Checkpoint should be declared!")
            raise RuntimeError
        else:
            test_loader, data_shape = fetching_dataset(args, config, logger, "test")
            args.data_shape = data_shape

            # ------- constructing model ------------
            model_combined = constructing_model(args, config, logger)

            test_model(args, config, logger, test_loader, model_combined)



    else:
        # ------- fetching target dataset -------
        train_loader, data_shape = fetching_dataset(args, config, logger, "train")
        test_loader, _ = fetching_dataset(args, config, logger, "test")
        args.data_shape = data_shape

        # ------- constructing model ------------
        model_combined = constructing_model(args, config, logger)

        # ------- training model ----------------
        training_model(args, config, logger, train_loader, model_combined, test_loader)


    end_time = datetime.datetime.now()

    args.running_time = end_time - start_time

    # ------- generating final logs ---------
    shutil_logs(args.log_name)

    # ------- send notification  ------------
    send_mail_notification(args)

    if args.tb_writer is not None:
        args.tb_writer.close()
        # --create archive symbolic link
        tensorboard_root = config.get_path_param(args.dataset)["tensorboard_root"]
        tb_symbolic_link(args.log_name, tensorboard_root)


if __name__ == '__main__':
    print('hollow word!')
    exit()


    stream_pool = separator_line()

    # acquire argparse options
    parser = argparse.ArgumentParser()

    parser.add_argument('--option', default="./options/com_cs_G3_base.json", metavar='PATH',
                        type=str, help="args option file path (default: None)")

    parser.add_argument('--resume', default=None, metavar='PATH',
                        type=str, help="resume file path (default: None)")

    parser.add_argument('--code', default=None, type=str,
                        help="identify code for another checkpoint")

    parser.add_argument('--evaluate', default=None, action='store_true',
                        help="evaluate model from checkpoint")

    parser.add_argument('--pretrained', default=None, type=str,
                        help="identify code for another checkpoint")
    args = parser.parse_args()

    args, stream_pool_sub = set_argparse_opts(args)

    stream_pool += "\n" + stream_pool_sub

    # initially loading config
    config = ConfigClass()

    # set the environmental variables
    stream_pool_sub = config.set_environ()
    stream_pool += stream_pool_sub

    # acquire the defined path parameter
    path_param = config.get_path_param(args.dataset)

    logger_name = args.code

    if args.evaluate:
        logger_name = "Evaluate_" + logger_name

    # create verbose logger
    logger, log_name = verbose_logs(path_param["log_root"], logger_name=args.code)

    args.log_name = log_name
    logger.info(stream_pool)
    logger.info(separator_line())

    # cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)

    if args.cuda:
        torch.cuda.manual_seed_all(args.manual_seed)

    # summary the args param
    logger.info("parameters of args are: ")
    logger.info(args_format(args))
    logger.info(separator_line())

    # tensorboard
    if args.tensorboard:
        args.tb_writer = SummaryWriter(args.log_name)

        # --create runnning symbolic link
        tensorboard_root = config.get_path_param(args.dataset)["tensorboard_root"]
        tb_symbolic_link(args.log_name, tensorboard_root, isrunning=True)

    else:
        args.tb_writer = None

    main(args, config, logger)

