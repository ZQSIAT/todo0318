import os
import logging
import csv
from utils.trivial_definition import datetime_now_string
from utils.trivial_definition import ensure_directory

verbose_logfile_name = "{:s}_verbose"

train_epoch_csv_name = "{:s}_train_epoch"
train_batch_csv_name = "{:s}_train_batch"

net_arch_file_name = "{:s}_net_arch"


class CSVLogger(object):
    """CSVLogger
    """
    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def verbose_logs(log_path, log_level="INFO", logger_name=None):
    """
    verbose_logs:
    :param log_path:
    :param log_level:
    :param logger_name:
    :return:
    """
    ensure_directory(log_path)

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    log_identity_name = "{:s}/{:s}_{:s}".format(log_path, logger_name, datetime_now_string(is_filename=True))

    file_handle = logging.FileHandler(verbose_logfile_name.format(log_identity_name) + ".temp")
    console_handle = logging.StreamHandler()

    """
    datetime_fmt = "%Y/%m/%d %H:%M:%S"
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datetime_fmt)

    file_handle.setFormatter(formatter)
    console_handle.setFormatter(formatter)
    """

    logger.addHandler(file_handle)
    logger.addHandler(console_handle)

    return logger, log_identity_name


def shutil_logs(logger_name):
    """
    shutil_logs:
    :param logger_name:
    :return:
    """
    # verbose log
    verbose_name_temp = verbose_logfile_name.format(logger_name) + ".temp"
    verbose_name_final = verbose_logfile_name.format(logger_name) + ".txt"

    if os.path.isfile(verbose_name_temp):
        os.rename(verbose_name_temp, verbose_name_final)
    else:
        print("Warning: Fail to read {:s}".format(verbose_name_temp))

    # epoch csv
    train_epoch_name_temp = train_epoch_csv_name.format(logger_name) + ".temp"
    train_epoch_name_final = train_epoch_csv_name.format(logger_name) + ".csv"

    if os.path.isfile(train_epoch_name_temp):
        os.rename(train_epoch_name_temp, train_epoch_name_final)
    else:
        print("Warning: Fail to read {:s}".format(train_epoch_name_temp))

    # batch csv
    train_batch_name_temp = train_batch_csv_name.format(logger_name) + ".temp"
    train_batch_name_final = train_batch_csv_name.format(logger_name) + ".csv"

    if os.path.isfile(train_batch_name_temp):
        os.rename(train_batch_name_temp, train_batch_name_final)
    else:
        print("Warning: Fail to read {:s}".format(train_batch_name_temp))