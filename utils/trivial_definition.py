import os
from datetime import datetime
from torchvision.utils import make_grid

max_dis_length = 120


def dispel_dict_variable(target_dict, var_string, def_string):
    """
    dispel_dict_variable:
    :param target_dict:
    :param var_string:
    :param def_string:
    :return:
    """
    for key, value in target_dict.items():
        if type(value) is dict:
            target_dict[key] = dispel_dict_variable(value, var_string, def_string)
        else:
            if type(value) is str:
                target_dict[key] = value.replace(var_string, def_string)
    return target_dict


def separator_line(dis_variable="", str_type="-", line_style="single", dis_len=None):
    """
    separator_line:
    :param dis_variable:
    :param str_type:
    :param line_style:
    :param dis_len:
    :return:
    """
    dis_length = max_dis_length/2 if dis_len == "half" else max_dis_length

    var_len = len(dis_variable)
    if var_len > 0:
        dis_variable = " "+dis_variable+" "
        hld_len = round((dis_length - var_len - 2) / 2)
    else:
        hld_len = round(dis_length/2)

    if line_style == "single":
        line_string = "{:s}{:s}{:s}".format(str_type * hld_len, dis_variable, str_type * hld_len)

    elif line_style == "triple":
        line_string = str_type * dis_length + "\n"
        line_string += "{:s}{:s}{:s}\n".format(" " * hld_len, dis_variable, " " * hld_len)
        line_string += str_type * dis_length

    else:
        raise NotImplementedError("{:s}: not implemented!".format(line_style))

    return line_string


def ensure_directory(path):
    """
    ensure_directory
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


def datetime_now_string(is_filename=False):
    """
    datetime_now_string
    :param is_filename:
    :return:
    """
    if is_filename:
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        return datetime.now().strftime("[%Y/%m/%d %H:%M:%S]")


def args_format(args):
    """
    args_format:
    :param args:
    :return:
    """
    return str(args).replace(", ", ",\n" + " "*4)


def epochs_format(epoch, epochs):
    """
    epochs_format:
    :param epoch:
    :param epochs:
    :return:
    """
    str_left = "Train Epoch:[{:d}/{:d}]".format(epoch, epochs)

    str_right = datetime_now_string()
    place_hold = (max_dis_length - len(str_left+str_right) - 2) * " "
    str_format = str_left + place_hold + str_right + "  "

    return str_format


def bacth_format(rates, batch_idx, batch_len, dataset_len, loss, accuracy):
    """
    bacth_format
    :param rates:
    :param batch_idx:
    :param batch_len:
    :param dataset_len:
    :param loss:
    :param accuracy:
    :return:
    """
    str_left = "{:4.1f}% Train: [{:0{str_len}d}/{:d}]".format(rates,
                                                              batch_idx * batch_len,
                                                              dataset_len,
                                                              str_len=len(str(dataset_len)))
    str_middle = "Loss: {loss.val:7.4f} ({loss.avg:7.4f})".format(loss=loss)
    str_right = "Accuracy: {acc.val:6.2f} ({acc.avg:6.2f})".format(acc=accuracy)

    # place_hold = max((max_dis_length - len(str_left + str_middle + str_right) - 10) // 2, 0) * " "
    place_hold = " " * 4
    str_format = str_left + place_hold + str_middle + place_hold + str_right

    return str_format


def tb_symbolic_link(dir_src, tb_root=None, isrunning=False):
    runing_id_str = "00-running-"
    running_dir_dst = "{:s}/{:s}{:s}".format(tb_root, runing_id_str, dir_src.split("/")[-1])
    archive_dir_dst = "{:s}/{:s}".format(tb_root, dir_src.split("/")[-1])
    if isrunning:
        os.symlink(dir_src, running_dir_dst)
    else:
        os.unlink(running_dir_dst)
        os.symlink(dir_src, archive_dir_dst)


def rescale(x, x_min=None, x_max=None):
    if x_min is None:
        x_min = x.min().asscalar()
    if x_max is None:
        x_max = x.max().asscalar()
    return (x - x_min) / (x_max - x_min)


def rescale_per_image(x):
    assert x.dim() == 5 or x.dim() == 4
    if x.dim() == 5:
        x = x.clone().permute(0, 2, 3, 4, 1) # NxCxKxKxK --> NxKxKxKxC
        x = x.contiguous().view(-1, x.shape[2], x.shape[3], x.shape[4]) # NxKxKxKxC --> MxKxKxC
    else:
        x = x.clone().permute(0, 2, 3, 1)
    for i in range(x.shape[0]):
        min_val = x[i, :, :, :].min().item()
        max_val = x[i].max().item()
        x[i, :, :, :] = (x[i, :, :, :] - min_val) / (max_val - min_val)
    y = make_grid(x.permute(0, 3, 1, 2), nrow=21, padding=2, normalize=True, range=None, scale_each=False)
    #y = y.permute(1, 2, 0).cpu().detach().numpy()
    return y
    """
    from matplotlib import pyplot as plt
    plt_figure = plt.figure()
    plt.imshow(y)
    plt.colorbar()

    return plt_figure
    """



if __name__ == "__main__":
    import torch

    from configs.param_config import ConfigClass
    config = ConfigClass()

    config.set_environ()
    from matplotlib import pyplot as plt

    a = torch.randn(64, 1, 7, 7, 7)
    dmaps = rescale_per_image(a)
    plt.show()







