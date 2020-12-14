import json
import os
from utils.trivial_definition import separator_line
from configs.param_config import ConfigClass
from datasets.MSRDailyAct3D import read_msr_depth_maps
# from numba import jit


def generate_protocol_for_MSRDailyAct3D(dataset_param, logger):
    """
    generate_protocol_for_MSRDailyAct3D:
    :param dataset_param:
    :return:
    """
    data_Name = "MSRDailyAct3D"
    dataset_dir = dataset_param["data_dir"]

    data_attr = dataset_param["data_attr"]
    data_type = dataset_param["data_type"]

    protocols = list(dataset_param["eval_protocols"].keys())

    # AssertionError for none protocols
    assert len(protocols) > 0, "evaluation protocol should be declared!"

    # Preparing new dict for json file
    data_param_dict = {}
    data_param_dict["data_Name"] = data_Name
    data_param_dict["data_Path"] = dataset_dir
    data_param_dict["file_format"] = {}
    data_param_dict["file_format"]["Depth"] = dataset_param["data_type"]["Depth"]
    data_param_dict["file_format"]["Skeleton"] = dataset_param["data_type"]["Skeleton"]
    data_param_dict["file_format"]["RGB"] = dataset_param["data_type"]["RGB"]
    data_param_dict["file_format"]["Pre_gradient"] = dataset_param["data_type"]["Pre_gradient"]
    data_param_dict["eval_Protocol"] = {}
    data_param_dict["action_Names"] = data_attr["n_names"]

    file_list_dir = dataset_param["file_list_dir"]

    if not os.path.exists(file_list_dir):
        os.makedirs(file_list_dir)

    for p_i, protocol_i in enumerate(protocols):

        protocol_param = dataset_param["eval_protocols"][protocol_i]
        train_list = []
        test_list = []

        action_list = []

        for n_si in range(1, data_attr["n_subs"] + 1):
            for n_ei in range(1, data_attr["n_exps"] + 1):
                for n_ai in range(1, data_attr["n_acts"] + 1):

                    file_str = dataset_param["file_format"].format(n_ai, n_si, n_ei)

                    depth_file = data_type["Depth"].replace("<$file_format>", file_str)
                    skeleton_file = data_type["Skeleton"].replace("<$file_format>", file_str)

                    # constant check
                    if dataset_param["data_constant_check"]:
                        if os.path.isfile(depth_file) and os.path.isfile(skeleton_file):
                            action_list.append(n_ai)

                            # get the depth temporal length
                            header_info = read_msr_depth_maps(depth_file, seqs_idx=None, header_only=True)
                            line_str = "{file_str:s}\t{label:d}\t{frames:d}".format(file_str=file_str,
                                                                                    label=n_ai-1,
                                                                                    frames=header_info[0])

                            if n_si in protocol_param["eval_subs"]:
                                test_list.append(line_str)
                            else:
                                train_list.append(line_str)

                    else:
                        if os.path.isfile(depth_file):
                            action_list.append(n_ai)

                            # get the depth temporal length
                            header_info = read_msr_depth_maps(depth_file, seqs_idx=None, header_only=True)
                            line_str = "{file_str:s}\t{label:d}\t{frames:d}".format(file_str=file_str,
                                                                                    label=n_ai-1,
                                                                                    frames=header_info[0])

                            if n_si in protocol_param["eval_subs"]:
                                test_list.append(line_str)
                            else:
                                train_list.append(line_str)

        # constant check for num classes
        num_classes = len(set(action_list))

        if num_classes != len(data_attr["n_names"]):
            logger.warn("Warning: num classes: {:d} is not equal to class names: {}.".format(num_classes, len(data_attr["n_names"])))

        # protocol name
        protocol_item = "{:d}_{:s}".format(p_i+1, protocol_i)

        logger.info(protocol_item+": ")

        # Write train list to file
        train_list_file = protocol_param["file_list_train"].replace("<$protocol_item>",
                                                                    protocol_item)
        with open(train_list_file, "w") as trlf:
            for train_line in train_list:
                trlf.write(train_line + "\n")
            trlf.close()
            logger.info("    Train filelist has been stored in '{:s}'".format(train_list_file))

        # Write test list to file
        test_list_file = protocol_param["file_list_test"].replace("<$protocol_item>",
                                                                  protocol_item)
        with open(test_list_file, "w") as telf:
            for test_line in test_list:
                telf.write(test_line + "\n")
            telf.close()
            logger.info("    Test filelist has been stored in '{:s}'".format(test_list_file))

        logger.info("    => Summary: {:d} samples for training and {:d} samples for test.".format(len(train_list),
                                                                                        len(test_list)))
        logger.info("    => Number of classes: {:d}".format(num_classes))

        logger.info(separator_line(dis_len="half"))

        assert len(train_list) > 0 and len(test_list) > 0, "Target dataset has no samples to read."
        data_param_dict["eval_Protocol"][protocol_item] = {}
        data_param_dict["eval_Protocol"][protocol_item]["train"] = train_list_file
        data_param_dict["eval_Protocol"][protocol_item]["test"] = test_list_file

    # num_classes to eval_Protocol
    data_param_dict["num_classes"] = num_classes


    # write protocol param to json file
    data_param_dict_file = dataset_param["eval_config_file"].format(file_list_dir, data_Name)

    with open(data_param_dict_file, 'w') as jsf:
        json.dump(data_param_dict, jsf, indent=4)
        logger.info("Evaluation protocols have been stored in '{:s}'".format(data_param_dict_file))
        logger.info(separator_line())

    return data_param_dict_file

def generate_protocol_for_MSRAction3D(dataset_param, logger):
    """
    generate_protocol_for_MSRAction3D:
    :param dataset_param:
    :return:
    """
    data_Name = "MSRAction3D"
    dataset_dir = dataset_param["data_dir"]

    data_attr = dataset_param["data_attr"]
    data_type = dataset_param["data_type"]

    protocols = list(dataset_param["eval_protocols"].keys())

    # AssertionError for none protocols
    assert len(protocols) > 0, "evaluation protocol should be declared!"

    # Preparing new dict for json file
    data_param_dict = {}
    data_param_dict["data_Name"] = data_Name
    data_param_dict["data_Path"] = dataset_dir
    data_param_dict["file_format"] = {}
    data_param_dict["file_format"]["Depth"] = dataset_param["data_type"]["Depth"]
    data_param_dict["file_format"]["Skeleton"] = dataset_param["data_type"]["Skeleton"]
    data_param_dict["eval_Protocol"] = {}
    data_param_dict["action_Names"] = data_attr["n_names"]

    file_list_dir = dataset_param["file_list_dir"]

    if not os.path.exists(file_list_dir):
        os.makedirs(file_list_dir)

    for p_i, protocol_i in enumerate(protocols):

        protocol_param = dataset_param["eval_protocols"][protocol_i]
        train_list = []
        test_list = []

        action_list = []

        for n_si in range(1, data_attr["n_subs"] + 1):
            for n_ei in range(1, data_attr["n_exps"] + 1):
                for n_ai in range(1, data_attr["n_acts"] + 1):

                    file_str = dataset_param["file_format"].format(n_ai, n_si, n_ei)

                    depth_file = data_type["Depth"].replace("<$file_format>", file_str)
                    skeleton_file = data_type["Skeleton"].replace("<$file_format>", file_str)

                    # constant check
                    if dataset_param["data_constant_check"]:
                        if os.path.isfile(depth_file) and os.path.isfile(skeleton_file):
                            action_list.append(n_ai)

                            # get the depth temporal length
                            header_info = read_msr_depth_maps(depth_file, header_only=True)
                            line_str = "{file_str:s}\t{label:d}\t{frames:d}".format(file_str=file_str,
                                                                                    label=n_ai-1,
                                                                                    frames=header_info[0])

                            if n_si in protocol_param["eval_subs"]:
                                test_list.append(line_str)
                            else:
                                train_list.append(line_str)

                    else:
                        if os.path.isfile(depth_file):
                            action_list.append(n_ai)

                            # get the depth temporal length
                            header_info = read_msr_depth_maps(depth_file, header_only=True)
                            line_str = "{file_str:s}\t{label:d}\t{frames:d}".format(file_str=file_str,
                                                                                    label=n_ai-1,
                                                                                    frames=header_info[0])

                            if n_si in protocol_param["eval_subs"]:
                                test_list.append(line_str)
                            else:
                                train_list.append(line_str)

        # constant check for num classes
        num_classes = len(set(action_list))

        if num_classes != len(data_attr["n_names"]):
            logger.warn("Warning: num classes: {:d} is not equal to class names: {}.".format(num_classes, len(data_attr["n_names"])))

        # protocol name
        protocol_item = "{:d}_{:s}".format(p_i+1, protocol_i)

        logger.info(protocol_item+": ")

        # Write train list to file
        train_list_file = protocol_param["file_list_train"].replace("<$protocol_item>",
                                                                    protocol_item)
        with open(train_list_file, "w") as trlf:
            for train_line in train_list:
                trlf.write(train_line + "\n")
            trlf.close()
            logger.info("    Train filelist has been stored in '{:s}'".format(train_list_file))

        # Write test list to file
        test_list_file = protocol_param["file_list_test"].replace("<$protocol_item>",
                                                                  protocol_item)
        with open(test_list_file, "w") as telf:
            for test_line in test_list:
                telf.write(test_line + "\n")
            telf.close()
            logger.info("    Test filelist has been stored in '{:s}'".format(test_list_file))

        logger.info("    => Summary: {:d} samples for training and {:d} samples for test.".format(len(train_list),
                                                                                        len(test_list)))
        logger.info("    => Number of classes: {:d}".format(num_classes))

        logger.info(separator_line(dis_len="half"))

        assert len(train_list) > 0 and len(test_list) > 0, "Target dataset has no samples to read."
        data_param_dict["eval_Protocol"][protocol_item] = {}
        data_param_dict["eval_Protocol"][protocol_item]["train"] = train_list_file
        data_param_dict["eval_Protocol"][protocol_item]["test"] = test_list_file

    # num_classes to eval_Protocol
    data_param_dict["num_classes"] = num_classes


    # write protocol param to json file
    data_param_dict_file = dataset_param["eval_config_file"].format(file_list_dir, data_Name)

    with open(data_param_dict_file, 'w') as jsf:
        json.dump(data_param_dict, jsf, indent=4)
        logger.info("Evaluation protocols have been stored in '{:s}'".format(data_param_dict_file))
        logger.info(separator_line())

    return data_param_dict_file


def generate_protocol_for_UOWCombined3D(dataset_param):
    """
    generate_protocol_for_UOWCombined3D
    :param dataset_param:
    :return:
    """
    data_Name = "UOWCombined3D"

    #------------------------------------------
    #
    #
    #           ToDo
    #
    #
    #------------------------------------------


def generate_protocol_for_NTU_RGBD(dataset_param, logger):
    """
    generate_protocol_for_MSRAction3D:
    :param dataset_param:
    :return:
    """
    data_Name = "NTU_RGB+D"
    dataset_dir = dataset_param["data_dir"]

    data_attr = dataset_param["data_attr"]
    data_type = dataset_param["data_type"]

    protocols = list(dataset_param["eval_protocols"].keys())

    # AssertionError for none protocols
    assert len(protocols) > 0, "evaluation protocol should be declared!"

    # Preparing new dict for json file
    data_param_dict = {}
    data_param_dict["data_Name"] = data_Name
    data_param_dict["data_Path"] = dataset_dir
    data_param_dict["file_format"] = {}
    data_param_dict["file_format"]["Depth"] = dataset_param["data_type"]["Depth"]
    data_param_dict["file_format"]["Skeleton"] = dataset_param["data_type"]["Skeleton"]
    data_param_dict["eval_Protocol"] = {}
    data_param_dict["action_Names"] = data_attr["n_names"]

    file_list_dir = dataset_param["file_list_dir"]

    if not os.path.exists(file_list_dir):
        os.makedirs(file_list_dir)

    for p_i, protocol_i in enumerate(protocols):

        protocol_param = dataset_param["eval_protocols"][protocol_i]
        train_list = []
        test_list = []

        action_list = []

        data_path = dataset_param["data_type"]["Depth"]

        file_list = os.listdir(data_path)

        for filename in file_list:

            action_class = int(
                filename[filename.find('A') + 1:filename.find('A') + 4])
            subject_id = int(
                filename[filename.find('P') + 1:filename.find('P') + 4])
            camera_id = int(
                filename[filename.find('C') + 1:filename.find('C') + 4])

            if protocol_i == "cross_view":
                istraining = (camera_id in protocol_param["train_cam"])

            elif protocol_i == "cross_subjects":
                istraining = (subject_id in protocol_param["train_subs"])

            else:
                raise ValueError()

            img_path = data_path + "/" + filename

            img_count = 0

            for img_name in os.listdir(img_path):
                if ".png" in img_name:
                    img_count += 1

            #img_count = len(os.listdir(img_path))

            assert img_count > 0, ValueError("Empty folder!")

            action_list.append(action_class)

            line_str = "{file_str:s}\t{label:d}\t{frames:d}".format(file_str=filename,
                                                                    label=action_class - 1,
                                                                    frames=img_count)

            if istraining:
                train_list.append(line_str)
            else:
                test_list.append(line_str)

        # constant check for num classes
        num_classes = len(set(action_list))

        if num_classes != len(data_attr["n_names"]):
            logger.warn("Warning: num classes: {:d} is not equal to class names: {}.".format(num_classes,
                                                                                             len(data_attr["n_names"])))
        # protocol name
        protocol_item = "{:d}_{:s}".format(p_i+1, protocol_i)

        logger.info(protocol_item+": ")

        # Write train list to file
        train_list_file = protocol_param["file_list_train"].replace("<$protocol_item>",
                                                                    protocol_item)
        with open(train_list_file, "w") as trlf:
            for train_line in train_list:
                trlf.write(train_line + "\n")
            trlf.close()
            logger.info("    Train filelist has been stored in '{:s}'".format(train_list_file))

        # Write test list to file
        test_list_file = protocol_param["file_list_test"].replace("<$protocol_item>",
                                                                  protocol_item)
        with open(test_list_file, "w") as telf:
            for test_line in test_list:
                telf.write(test_line + "\n")
            telf.close()
            logger.info("    Test filelist has been stored in '{:s}'".format(test_list_file))

        logger.info("    => Summary: {:d} samples for training and {:d} samples for test.".format(len(train_list),
                                                                                        len(test_list)))
        logger.info("    => Number of classes: {:d}".format(num_classes))

        logger.info(separator_line(dis_len="half"))

        assert len(train_list) > 0 and len(test_list) > 0, "Target dataset has no samples to read."
        data_param_dict["eval_Protocol"][protocol_item] = {}
        data_param_dict["eval_Protocol"][protocol_item]["train"] = train_list_file
        data_param_dict["eval_Protocol"][protocol_item]["test"] = test_list_file

    # num_classes to eval_Protocol
    data_param_dict["num_classes"] = num_classes


    # write protocol param to json file
    data_param_dict_file = dataset_param["eval_config_file"].format(file_list_dir, data_Name)

    with open(data_param_dict_file, 'w') as jsf:
        json.dump(data_param_dict, jsf, indent=4)
        logger.info("Evaluation protocols have been stored in '{:s}'".format(data_param_dict_file))
        logger.info(separator_line())

    return data_param_dict_file

def generate_protocol_files(data_name, dataset_param, logger):
    """

    :param data_name:
    :param dataset_param:
    :return:
    """
    if data_name == "MSRDailyAct3D":
        return generate_protocol_for_MSRDailyAct3D(dataset_param, logger)

    elif data_name == "MSRAction3D":
        return generate_protocol_for_MSRAction3D(dataset_param, logger)

    elif data_name == "UOWCombined3D":
        return generate_protocol_for_UOWCombined3D(dataset_param, logger)

    elif data_name == "NTU_RGB+D":
        return generate_protocol_for_NTU_RGBD(dataset_param, logger)

    else:
        raise ValueError("Unknown dataset: '{:s}'".format(data_name))


if __name__ == '__main__':

    config = ConfigClass()

    dataset_param = config.get_dataset_param("NTU_RGB+D")
    import logging
    logger = logging.getLogger()
    generate_protocol_for_NTU_RGBD(dataset_param, logger)

