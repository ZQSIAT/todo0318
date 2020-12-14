# -*- coding: utf-8 -*-
"""
function: global config file
"""
import os
import json
from utils.trivial_definition import dispel_dict_variable


class ConfigClass(object):
    def __init__(self,
                 config_dir=os.path.dirname(os.path.realpath(__file__)),
                 datasets_config="datasets_config.json",
                 environ_config="environ_config.json",
                 loader_config="loader_config.json",
                 model_config="model_config.json",
                 path_config="path_config.json"):

        self.config_dir = config_dir
        self.datasets_config = config_dir + "/" + datasets_config
        self.environ_config = config_dir + "/" + environ_config
        self.loader_config = config_dir + "/" + loader_config
        self.model_config = config_dir + "/" + model_config
        self.path_config = config_dir + "/" + path_config

        assert os.path.isfile(self.datasets_config), "Invalid config files: '{:s}'!".format(self.datasets_config)
        assert os.path.isfile(self.environ_config), "Invalid config files: '{:s}'!".format(self.environ_config)
        assert os.path.isfile(self.path_config), "Invalid config files: '{:s}'!".format(self.path_config)
        assert os.path.isfile(self.loader_config), "Invalid config files: '{:s}'!".format(self.loader_config)

    def set_environ(self, is_print=False):
        """
        set_environ
        :return:
        """
        with open(self.environ_config, "r") as ecf:
            environ_config_param = json.load(ecf)
            stream_pool_sub = ""
            # adding environment variable
            if "environ" in environ_config_param.keys():
                environ_param_to_add = environ_config_param["environ"]
                for epta_item in environ_param_to_add:
                    epta_value = environ_param_to_add[epta_item]
                    os.environ[epta_item] = epta_value
                    if type(epta_value) == str:
                        stream_pool_sub += ("=> executing: os.environ['{:s}']='{:s}'\n".format(epta_item, epta_value))
                    else:
                        stream_pool_sub += ("=> executing: os.environ['{:s}']={:d}\n".format(epta_item, epta_value))

            # setting matplotlib backend
            if "on_plot" in environ_config_param.keys():
                onplot_param_to_set = environ_config_param["on_plot"]
                if "backend" in onplot_param_to_set.keys():
                    backend = onplot_param_to_set["backend"]
                    import matplotlib
                    matplotlib.use(backend)
                    stream_pool_sub += ("=> executing: matplotlib.use('{:s}')\n".format(backend))

            stream_pool_sub = stream_pool_sub[:-1]
            if is_print:
                print(stream_pool_sub)

            return stream_pool_sub

    def get_loader_param(self, data_name, modality):
        """
        get_loader_param
        :param data_name:
        :param modality:
        :param treaty:
        :return:
        """
        with open(self.loader_config, "r") as lcf:
            loader_config_param = json.load(lcf)
            loader_param = loader_config_param[data_name][modality]

            return loader_param

    def get_model_param(self):
        """
        get_model_param
        :return:
        """
        with open(self.model_config, "r") as lcf:
            loader_model_param = json.load(lcf)
            return loader_model_param

    def get_dataset_param(self, data_name):
        """
        get_dataset_param
        :param data_name:
        :return:
        """
        path_config_param = self.get_path_param(data_name)

        with open(self.datasets_config, "r") as dcf:
            datasets_config_param = json.load(dcf)
            dataset_param = datasets_config_param[data_name]

            var_replaced_pool = {
                "<$data_name>": data_name,
                "<$data_dir>": dataset_param["data_dir"],
                "<$file_list_dir>": dataset_param["file_list_dir"],
                "<$deploy_root>": path_config_param["deploy_root"],
                "<$protocol_root>": path_config_param["protocol_root"],
                "<$data_root>": path_config_param["data_root"],
            }

            for var_key, var_value in var_replaced_pool.items():
                dataset_param = dispel_dict_variable(dataset_param, var_key, var_value)

            return dataset_param

    def get_path_param(self, data_name):
        """
        get_path_param:
        :param path_config_file:
        :return:
        """
        with open(self.path_config, "r") as pcf:
            path_config_param = json.load(pcf)

            path_config_param = path_config_param['10.60.102.249:4022']   # 10.10.1.32 10.60.102.249:4022 10.60.102.249:5022
            # print(type(path_config_param), path_config_param)
            # exit()
            path_config_param = dispel_dict_variable(path_config_param, "<$data_name>",
                                                     data_name)

            path_config_param = dispel_dict_variable(path_config_param, "<$deploy_root>",
                                                     path_config_param["deploy_root"])


            return path_config_param

    def get_defined_datasets_list(self):
        """
        get_defined_datasets_list:
        :param datasets_config_file:
        :return:
        """
        with open(self.datasets_config, "r") as dcf:
            datasets_param = json.load(dcf)
            datasets_list = list(datasets_param.keys())

            assert len(datasets_list) > 0, "Improper datasets config file!"

            return datasets_list

    def get_number_classes(self, data_name):
        """
        get_number_classes:
        :param data_name:
        :return:
        """
        dataset_param = self.get_dataset_param(data_name)
        protocol_config_file = dataset_param["eval_config_file"]

        # read protocol params
        with open(protocol_config_file, "r") as pcf:
            protocol_config = json.load(pcf)
            num_classes = protocol_config["num_classes"]

        return num_classes


if __name__ == '__main__':
    # set_environ(environ_config)
    # get_path_param()

    config = ConfigClass()

    config.set_environ()
    config.get_defined_datasets_list()

    #get_dataset_param("MSRDailyAct3D")

