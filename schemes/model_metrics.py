import torch


class AverageMeter(object):
    """Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = max(val, self.max)


def calculate_accuracy_percent(outputs, targets):
    """
    calculate_accuracy_percent:
    :param outputs:
    :param targets:
    :return:
    """
    with torch.no_grad():
        batch_size = targets.size(0)

        _, pred = outputs.topk(1, 1, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))
        n_correct_elems = correct.float().sum()
        percent_accuracy = n_correct_elems.mul(100.0 / batch_size)

        return percent_accuracy, n_correct_elems


# ----------------------------------
# Todo: recall for uow dataset
# ----------------------------------
def calculate_recall(outputs, targets):
    batch_size = targets.size(0)
    raise NotImplementedError("ToDo")


def accuracy_topk(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k
    accuracy_topk:
    :param output:
    :param target:
    :param topk:
    :return:
    """
    """"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(adjust_lr_param, lr_initial, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by defined adjust method
    :param config:
    :param adjust_method:
    :param lr_initial:
    :param optimizer:
    :param epoch:
    :return:
    """
    lr_method = adjust_lr_param["lr_method"]

    if "decay_every_epoch" in lr_method:
        lr_steps = adjust_lr_param["lr_steps"]
        lr_decay = adjust_lr_param["lr_decay"]

        lr = lr_initial * (lr_decay ** (epoch // lr_steps))

    elif "decay_every_epoch" in lr_method:
        lr_steps = adjust_lr_param["lr_steps"]
        lr_decay = adjust_lr_param["lr_decay"]

        assert len(lr_steps) == len(lr_decay) + 1, "Invalid configuration!"

        lr = lr_initial

        for i, el in enumerate(lr_steps):
            if epoch > el:
                lr = lr_initial * lr_steps[i]

    else:
        raise ValueError("Unknown adjust method!")

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def to_one_hot(x, length):
    """
    to_one_hot: Converts batches of class indices to classes of one-hot vectors
    :param x:
    :param length:
    :return:
    """
    batch_size = x.size(0)
    x_one_hot = torch.zeros(batch_size, length)
    for i in range(batch_size):
        x_one_hot[i, x[i]] = 1.0
    return x_one_hot
