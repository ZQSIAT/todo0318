import time
import torch
from schemes.model_metrics import AverageMeter
from schemes.model_metrics import calculate_accuracy_percent
from utils.trivial_definition import separator_line
from torch.nn import functional as F
from utils.plot_utilities import generate_confusion_matrix


def validate_model(args, logger, val_loader, model_combined, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    samples_count = AverageMeter()
    samples_right = AverageMeter()

    [model, metrics, criterion, _] = model_combined

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        y_true = []
        y_pred = []
        for i, (input, target, _) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            if args.cuda is not None:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

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

            """
            if i % args.log_interval == 0:
                print('Test: [{0}/{1}]--'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})--'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})--'
                      'Prec@1 {acc.val:.3f} ({acc.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       acc=accuracy))
            """

        if args.plot_confusion_matrix:
            class_names = val_loader.class_names
            plt_fig = generate_confusion_matrix(y_true, y_pred, class_names)
            args.tb_writer.add_figure('confusion matrix', plt_fig, epoch)

        logger.info('=> Validate:  '
                    'Elapse: {data_time.sum:.2f}/{sum_time.sum:.2f}s  '
                    'Loss: {loss.avg:.4f}  '
                    'Accuracy: {acc.avg:.2f}% '
                    '[{right:.0f}/{count:.0f}]'.format(loss=losses,
                                                  data_time=data_time,
                                                  sum_time=batch_time,
                                                  acc=accuracy,
                                                  right=samples_right.sum,
                                                  count=samples_count.sum))
        logger.info(separator_line())

    return accuracy.avg, losses.avg
