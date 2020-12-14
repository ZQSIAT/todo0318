import time
import torch
from schemes.model_metrics import AverageMeter
from schemes.model_metrics import calculate_accuracy_percent
from utils.trivial_definition import datetime_now_string
from utils.trivial_definition import separator_line
from utils.trivial_definition import ensure_directory
from torch.nn import functional as F
from utils.plot_utilities import generate_confusion_matrix
import os


def test_model(args, config, logger, val_loader, model_combined):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    samples_count = AverageMeter()
    samples_right = AverageMeter()

    [model, metrics, criterion, _] = model_combined

    logger.info("=> Recovery model dict state from checkpoint")
    if os.path.isfile(args.resume):
        logger.info("{:s} loading checkpoint '{}'".format(datetime_now_string(), args.resume))
        checkpoint = torch.load(args.resume)

        # load param from checkpoint file
        best_prec = checkpoint['best_prec']
        arch = checkpoint['arch']
        state_dict = checkpoint['state_dict']
        assert arch == args.net_arch, "The arch of checkpoint is not consistent with current model!"

        model.load_state_dict(state_dict)

        logger.info("{:s} loaded checkpoint '{}' (epoch {})".format(datetime_now_string(),
                                                                    args.resume, checkpoint['epoch']))

    else:
        logger.error("Error: No checkpoint found at '{}'".format(args.resume))
        raise RuntimeError

    # get checkpoint saved directory
    evaluate_root = config.get_path_param(args.dataset)["evaluate_root"]
    ensure_directory(evaluate_root)

    logger.info(separator_line())
    # switch to evaluate mode
    model.eval()
    logger.info("{:s} Start model evaluation".format(datetime_now_string()))
    logger.info(separator_line())

    with torch.no_grad():
        end = time.time()
        y_true = []
        y_pred = []
        output_list = []

        for i, (input, target, _) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            if args.cuda is not None:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            output_list.append(output)

            y_true.extend(target.tolist())
            _, pred = output.topk(1, 1, True, True)
            y_pred.extend(pred.t().tolist()[0])

            # measure accuracy and record loss
            if "accuracy_percent" in metrics:
                predicted_accuracy, n_correct_elems = calculate_accuracy_percent(output, target)
                samples_count.update(input.size(0))
                samples_right.update(n_correct_elems.item())
                accuracy.update(predicted_accuracy.item(), input.size(0))


            # Todo: add more metrics such as recall for special dataset

            losses.update(loss.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.log_interval == 0:
                logger.info(
                    '{:4.1f}% evaluate:[{:05d}/{:05d}] '
                    'Loss: {loss.val:7.4f} ({loss.avg:7.4f})  '
                    'Accuracy: {acc.val:6.2f} ({acc.avg:6.2f})  '
                    .format(100.0*i/len(val_loader), i*args.batch_size, len(val_loader.dataset),
                            loss=losses,
                            acc=accuracy)
                )

        output_score = F.softmax(torch.cat(output_list, dim=0))
        output_file_path = "{:s}/{:s}_{:.2f}_{:s}_{:s}".format(evaluate_root,
                                                               args.eval_protocol,
                                                               accuracy.avg,
                                                               args.code,
                                                               args.net_arch)
        save_predict_result(
            logger,
            {'output_score': output_score,
             'true_label': y_true,
             'pred_label': y_pred},
            output_file_path
        )

        if args.plot_confusion_matrix:
            class_names = val_loader.class_names
            plt_fig = generate_confusion_matrix(y_true, y_pred, class_names)
            args.tb_writer.add_figure('confusion matrix', plt_fig)

        logger.info('=> Evaluate:  '
                    'Elapse: {data_time.sum:.2f}/{sum_time.sum:.2f}s  '
                    'Loss: {loss.avg:.4f}  '
                    'Model record accuracy: {best_acc:.2f}% '
                    'Evaluate accuracy: {acc.avg:.2f}% '
                    '[{right:.0f}/{count:.0f}]'.format(loss=losses,
                                                       data_time=data_time,
                                                       sum_time=batch_time,
                                                       best_acc=best_prec,
                                                       acc=accuracy,
                                                       right=samples_right.sum,
                                                       count=samples_count.sum))
        logger.info(separator_line())

    return accuracy.avg, losses.avg


def save_predict_result(logger, state, output_file_path):
    ensure_directory(output_file_path)
    output_pth_name = "{:s}/Evaluate_result.pth.tar".format(output_file_path)
    torch.save(state, output_pth_name)

    # write label to txt
    txt_file_name = output_pth_name.replace(".pth.tar", ".txt")
    true_label = state["true_label"]
    pred_label = state["pred_label"]

    with open(txt_file_name, "w") as trlf:
        for i in range(len(pred_label)):
            line = "{:d}\t{:d}\t{:d}\n".format(i, true_label[i], pred_label[i])
            trlf.write(line)
        trlf.close()
    logger.info("Evaluation result has been stored in '{:s}'".format(output_file_path))
