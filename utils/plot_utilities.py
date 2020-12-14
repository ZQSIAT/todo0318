import torch
from torchviz import make_dot
from sklearn.metrics import confusion_matrix
from torchvision.utils import make_grid

def plot_net_architecture(model, input_size, file_name, file_format="pdf"):
    """
    plot_net_architecture
    :param model:
    :param input_size:
    :param file_name:
    :param file_format:
    :return:
    """
    x = torch.randn(tuple(input_size))
    # print(x.shape)
    y = model(x)
    grp_dot = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
    grp_dot.format = file_format
    grp_dot.render(file_name, view=False)


def plot_confusion_matrix(cm, classes, normalize=False, title=None, cmap=None, ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    from matplotlib import pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))

    if title is None:
        title = "Confusion matrix"

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt_fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes, rotation=20)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] > 0.0:
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label\n accuracy={:0.4f}".format(accuracy))
    plt.tight_layout()

    return plt_fig


def generate_confusion_matrix(y_true, y_pred, target_names, title=None):
    cnf_matrix = confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)
    plt_fig = plot_confusion_matrix(cnf_matrix,
                          target_names,
                          title=title,
                          cmap=None,
                          normalize=True)
    return plt_fig


def generate_sprite_tensor(x, is_color=False):
    assert x.dim() == 5 or x.dim() == 4
    if is_color:
        if x.dim() == 5:
                x = x.clone().permute(0, 2, 3, 4, 1)  # NxCxKxKxK --> NxKxKxKxC
                x = x.contiguous().view(-1, x.shape[2], x.shape[3], x.shape[4])  # NxKxKxKxC --> MxKxKxC
        else:
            x = x.clone().permute(0, 2, 3, 1)
        for i in range(x.shape[0]):
            min_val = x[i, :, :, :].min().item()
            max_val = x[i].max().item()
            x[i, :, :, :] = (x[i, :, :, :] - min_val) / (max_val - min_val)
        y = make_grid(x.permute(0, 3, 1, 2), nrow=21, padding=2, normalize=True, range=None, scale_each=False)
        # y = y.permute(1, 2, 0).cpu().detach().numpy()
    else:
        if x.dim() == 5:
            ln_row = x.shape[2]
            x = x.clone().view(-1, x.shape[3], x.shape[4])  # NxCxKxKxK --> MxKxK
        else:
            n_row = 4
            x = x.clone().view(-1, x.shape[2], x.shape[3])  # NxCxKxK --> MxKxK
        for i in range(x.shape[0]):
            min_val = x[i, :, :, :].min().item()
            max_val = x[i].max().item()
            x[i, :, :, :] = (x[i, :, :, :] - min_val) / (max_val - min_val)
        y = make_grid(x.squeeze(1), nrow=n_row, padding=2, normalize=True, range=None, scale_each=False)
        # y = y.permute(1, 2, 0).cpu().detach().numpy()

    return y



if __name__ == "__main__":
    from configs.param_config import ConfigClass
    config = ConfigClass()
    config.set_environ()

    y_true = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    y_pred = [1, 2, 3, 4, 5, 1, 3, 2, 4, 5]
    label_name = ["a", "b", "c", "d", "e"]

    plt_fig = generate_confusion_matrix(y_true, y_pred, label_name)


