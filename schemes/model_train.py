import os
import time
import torch
import torchvision
import numpy as np
from schemes.model_construct import save_checkpoint
from schemes.model_metrics import AverageMeter
from schemes.model_metrics import calculate_accuracy_percent
from schemes.model_metrics import adjust_learning_rate
from schemes.model_validate import validate_model
from utils.trivial_definition import ensure_directory
from utils.trivial_definition import separator_line
from utils.trivial_definition import datetime_now_string
from utils.trivial_definition import epochs_format, bacth_format
from utils.trivial_definition import rescale_per_image
from utils.log_recorder import CSVLogger
from utils.log_recorder import train_epoch_csv_name, train_batch_csv_name


def training_model(args, config, logger, train_loader, model_combined, val_loader):

    [model, metrics, criterion, optimizer] = model_combined

    # optionally resume from a checkpoint
    if args.resume:
        logger.info("=> Recovery model training from checkpoint")
        if os.path.isfile(args.resume):
            logger.info("{:s} loading checkpoint '{}'".format(datetime_now_string(), args.resume))
            checkpoint = torch.load(args.resume)

            # load param from checkpoint file
            start_epoch = checkpoint['epoch']
            best_prec = checkpoint['best_prec']
            best_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            logger.info("{:s} loaded checkpoint '{}' (epoch {})".format(datetime_now_string(),
                                                                        args.resume, checkpoint['epoch']))

        else:
            logger.warning("Warning: No checkpoint found at '{}'".format(args.resume))

        logger.info(separator_line())

    else:
        start_epoch = 0
        best_prec = 0
        best_epoch = 0

    # get checkpoint saved directory
    checkpoint_root = config.get_path_param(args.dataset)["checkpoint_root"]
    ensure_directory(checkpoint_root)

    # logger.info(separator_line())
    logger.info("{:s} Start model training".format(datetime_now_string()))
    logger.info(separator_line())

    lr_initial = optimizer.param_groups[0]["lr"]

    # record epoch and batch logger with csv file
    train_epoch_logger = CSVLogger(train_epoch_csv_name.format(args.log_name) + ".temp",
                             ['epoch', 'train_loss', 'train_accuracy',
                              'validate_loss', 'validate_accuracy', 'lr'])

    train_batch_logger = CSVLogger(train_batch_csv_name.format(args.log_name) + ".temp",
                                   ['epoch', 'batch', 'iteration', 'loss', 'accuracy', 'lr'])

    # ---- early stop the training ----
    is_overfit_before = False
    is_overfit_count = 0

    is_loss_steady = False
    is_loss_count = 10
    is_loss_list = np.zeros(is_loss_count)

    # ---- start the epochs ----

    for epoch in range(start_epoch, args.epochs):

        logger.info(epochs_format(epoch, args.epochs))
        logger.info(separator_line())

        # if adjust the learning rate
        if args.adjust_lr is not None and args.adjust_lr not in "disable":
            adjust_lr_param = config.get_model_param()["adjust_lr"][args.adjust_lr]
            adjust_lr_param["lr_method"] = args.adjust_lr

            adjust_learning_rate(adjust_lr_param, lr_initial, optimizer, epoch)

        epoch_idx = [epoch, args.epochs]

        # train for one epoch
        train_prec, train_loss = train_epoch(args, logger, train_loader, model_combined,
                                             epoch_idx, train_batch_logger)

        # evaluate on validation set
        valid_prec, valid_loss = validate_model(args, logger, val_loader, model_combined, epoch)

        # remember the best predict accuracy and save the checkpoint
        is_best = valid_prec > best_prec
        best_prec = max(valid_prec, best_prec)

        is_check = epoch % args.checkpoint_interval == 0

        if is_best:
            logger.info("{:s} checkpoint at epoch: {:d} with accuracy: {:.2f}".format(datetime_now_string(),
                                                                                         epoch, best_prec))
            logger.info(separator_line())
            checkpoint_file_name = "{:s}/{:s}_{:2.0f}_{:s}_model_best.pth.tar".format(checkpoint_root, args.code,
                                                                                      args.data_dosage * 100,
                                                                                      args.net_arch)

            best_epoch = epoch

        else:
            checkpoint_file_name = "{:s}/{:s}_{:s}_model_checkpoint.pth.tar".format(checkpoint_root,
                                                                                    args.code,
                                                                                    args.net_arch)
        # save checkpoint
        if is_check or is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.net_arch,
                'state_dict': model.state_dict(),
                'best_prec': best_prec,
                'optimizer': optimizer.state_dict(),
            }, checkpoint_file_name)

        # csv epoch logger
        train_epoch_logger.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_prec,
            'validate_loss': valid_loss,
            'validate_accuracy': valid_prec,
            'lr': optimizer.param_groups[0]['lr']
        })

        # tensorboard writer for epoch
        args.tb_writer.add_scalars('data/epoch_loss', {'train_loss': train_loss,
                                                 'validate_loss': valid_loss},
                                   epoch + 1)
        args.tb_writer.add_scalars('data/epoch_accuracy', {'train_accuracy': train_prec,
                                                 'validate_accuracy': valid_prec},
                                   epoch + 1)
        args.tb_writer.add_scalar('data/epoch_lr', optimizer.param_groups[0]['lr'], epoch + 1)


        # ---- break down epochs if the model is over fitting
        if train_prec > args.overfit_threshold[0] and valid_prec < args.overfit_threshold[1]:
            if is_overfit_before:
                is_overfit_count += 1
            else:
                is_overfit_count = 0

            is_overfit_before = True
            if is_overfit_count > args.overfit_threshold[2]:
                # generate the logs in advance
                logger.warning("Warning: Iteration has been terminated as the model is over fitting! "
                               "Best accuracy is {:.2f}% on epoch {:d}.".format(best_prec, best_epoch))
                # logger.info(separator_line())
                break
        else:
            is_overfit_before = False

        # ---- early stop for no loss declination
        is_loss_list = np.append(is_loss_list[-(is_loss_count - 1):], train_loss)

        if is_loss_list.std() < 1e-4:
            is_loss_steady = True
            logger.warning("Warning: Training Loss has been no more declining! "
                           "Best accuracy is {:.2f}% on epoch {:d}.".format(best_prec, best_epoch))
            # logger.info(separator_line())
            break
        else:
            is_loss_steady = False

    if not is_overfit_before and not is_loss_steady:
        logger.info("Total {:d} epochs of model training have finished, "
                    "best accuracy is {:.2f}% on epoch {:d}.".format(args.epochs, best_prec, best_epoch))


def train_epoch(args, logger, train_loader, model_combined, epoch_idx, train_batch_logger):
    """
    train_epoch:
    :param args:
    :param logger:
    :param train_loader:
    :param model_combined:
    :param epoch_idx:
    :param train_batch_logger:
    :return:
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    processed = AverageMeter()

    [model, metrics, criterion, optimizer] = model_combined
    [epoch, epochs] = epoch_idx

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target, _) in enumerate(train_loader):
        # measure data loading time
        torch.cuda.synchronize()   #增加同步操作
        data_time.update(time.time() - end)

        # print(input.shape)
        # from matplotlib import pyplot as plt
        # for m in range(7):
        #     plt.imshow(input[0, 0, m, :, :])
        #     plt.colorbar()
        #     plt.show()
        #     plt.imshow(input[0, 1, m, :, :])
        #     plt.colorbar()
        #     plt.show()
        #     plt.imshow(input[0, 2, m, :, :])
        #     plt.colorbar()
        #     plt.show()
        #     plt.imshow(input[0, 3, m, :, :])
        #     plt.colorbar()
        #     plt.show()
        # raise RuntimeError

        if args.cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)


        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure metrics, eg., accuracy and record loss
        if "accuracy_percent" in metrics:
            predicted_accuracy, _ = calculate_accuracy_percent(output, target)
            accuracy.update(predicted_accuracy.item(), input.size(0))

        # Todo: add more metrics such as recall for special dataset

        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()   #增加同步操作
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        n_iter = epoch * len(train_loader) + i

        if i % args.log_interval == 0:
            #print("min: {:.2f} max: {:.2f}".format(input.min(), input.max()))
            process_rate = (100.0 * (epoch * len(train_loader) + i) / (epochs * len(train_loader)))
            logger.info(bacth_format(process_rate, i+1, args.batch_size, len(train_loader.dataset), losses, accuracy))

            """
            args.tb_writer.add_embedding(output.data,
                                         metadata=target.cpu().data.numpy(),
                                         label_img=input.data,
                                         global_step=n_iter)
            """

            # weight_conv_l1 = model

            # args.tb_writer.add_histogram('hist', array, n_iter)

            # for name, param in model.named_parameters():
            #     # if 'bn' not in name:
            #         # print("{:s}--{:s}".format(str(name), str(param.shape)))
            #     #    args.tb_writer.add_histogram(name, param, n_iter)
            #     # if 'conv1.weight' in name:
            #     # #     args.tb_writer.add_image('conv1_filter', rescale_per_image(param[:,0:3,:,:,:]), n_iter)
            #     # if 'module.features.0.conv.weight' in name:
            #     #     args.tb_writer.add_image('I3D_conv1_filter', rescale_per_image(param), n_iter)
            #     if "module.conv1.weight" in name:
            #         if param.shape[1] <=3:
            #             args.tb_writer.add_image('conv1_filter', rescale_per_image(param), n_iter)


        # csv batch logger
        train_batch_logger.log({
            'epoch': epoch + 1,
            'batch': i + 1,
            'iteration': epoch * len(train_loader) + i + 1,
            'loss': losses.avg,
            'accuracy': accuracy.val,
            'lr': optimizer.param_groups[0]['lr']
        })

        # tensorboard writer
        args.tb_writer.add_scalar('data/batch_loss', losses.avg, n_iter)
        args.tb_writer.add_scalar('data/batch_accuracy', accuracy.val, n_iter)
        args.tb_writer.add_scalar('data/batch_lr', optimizer.param_groups[0]['lr'], n_iter)

    # print the average loss and metrics
    logger.info(separator_line(dis_len="half"))
    logger.info("=> Training:  "
                "Elapse: {data_time.sum:.2f}/{sum_time.sum:.2f}s  "
                "Loss: {loss.avg:.4f}  "
                "Accuracy: {acc.avg:.2f}%".format(loss=losses,
                                                  data_time=data_time,
                                                  sum_time=batch_time,
                                                  acc=accuracy))
    return accuracy.avg, losses.avg







