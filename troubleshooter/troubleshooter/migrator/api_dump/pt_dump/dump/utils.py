import os
import shutil
from functools import lru_cache
from pathlib import Path

from troubleshooter import log as logger

from ..common.utils import (Const, DumpException, get_api_name_from_matcher, get_time,
                            print_error_log)

range_begin_flag, range_end_flag = False, False


class DumpUtil(object):
    dump_data_dir = None
    dump_ori_dir = None
    dump_path = None
    dump_file_name = None
    dump_stack_file = None
    dump_switch = None
    dump_switch_mode = Const.ALL
    dump_switch_scope = []
    dump_init_enable = False
    dump_api_list = []
    dump_filter_switch = None
    dump_mode = Const.ALL
    dump_type = Const.ALL
    backward_input = {}
    dump_config = None
    dump_stack_dic = {}
    dump_filter_stack = True
    dump_count = 0

    @staticmethod
    def set_ori_dir(path):
        DumpUtil.dump_ori_dir = path

    @staticmethod
    def set_dump_path(dump_file_path, dump_file_name, dump_stack_file):
        DumpUtil.dump_path = dump_file_path
        DumpUtil.dump_file_name = dump_file_name
        DumpUtil.dump_stack_file = dump_stack_file
        DumpUtil.dump_init_enable = True

    @staticmethod
    def set_dump_config(dump_config):
        DumpUtil.dump_config = dump_config

    @staticmethod
    def set_dump_switch(switch, mode, scope, api_list,
                        filter_switch, dump_mode, dump_type, filter_stack):
        DumpUtil.dump_switch = switch
        DumpUtil.dump_switch_mode = mode
        DumpUtil.dump_switch_scope = scope
        DumpUtil.dump_api_list = [api.lower() for api in api_list]
        DumpUtil.dump_filter_switch = filter_switch
        DumpUtil.dump_mode = dump_mode
        DumpUtil.dump_type = dump_type
        DumpUtil.dump_filter_stack = filter_stack
        if mode == Const.ACL:
            DumpUtil.dump_switch_scope = [api_name.replace("backward", "forward") for api_name in scope]

    @lru_cache()
    def check_list_or_acl_mode(name_prefix):
        return name_prefix in DumpUtil.dump_switch_scope

    def check_range_mode(name_prefix):
        global range_begin_flag
        global range_end_flag
        if name_prefix == DumpUtil.dump_switch_scope[0]:
            range_begin_flag = True
            return True
        if name_prefix == DumpUtil.dump_switch_scope[1]:
            range_end_flag = True
            return True
        if range_begin_flag and not range_end_flag:
            return True
        return False

    @lru_cache()
    def check_in_api_list(name_prefix):
        name_prefix = name_prefix.lower()
        for v in DumpUtil.dump_api_list:
            if v in name_prefix:
                return True
        return False

    def check_stack_mode(name_prefix):
        if len(DumpUtil.dump_switch_scope) == 0:
            return True
        elif len(DumpUtil.dump_switch_scope) == 1:
            return name_prefix.startswith(DumpUtil.dump_switch_scope[0])
        elif len(DumpUtil.dump_switch_scope) == 2:
            return DumpUtil.check_range_mode(name_prefix)
        else:
            print_error_log("dump scope is invalid, Please set the scope mode in"
                            " set_dump_switch with 'all', 'list', 'range', 'stack', 'acl', 'api_list'!")
        return False

    check_mapper = {
        Const.ALL: lambda _: True,
        Const.LIST: check_list_or_acl_mode,
        Const.API_LIST: check_in_api_list,
        Const.ACL: check_list_or_acl_mode,
        Const.RANGE: check_range_mode,
        Const.STACK: check_stack_mode
    }

    @staticmethod
    def check_switch_scope(name_prefix):
        if DumpUtil.dump_switch_mode in DumpUtil.check_mapper:
            check_func = DumpUtil.check_mapper[DumpUtil.dump_switch_mode]
            return check_func(name_prefix)
        return False

    @staticmethod
    def get_dump_path():
        if DumpUtil.dump_path and DumpUtil.dump_file_name and DumpUtil.dump_stack_file:
            return DumpUtil.dump_path, DumpUtil.dump_file_name, DumpUtil.dump_stack_file

        if DumpUtil.dump_switch_mode == Const.ALL:
            raise RuntimeError("get_dump_path: the file path is empty,"
                               " you must use set_dump_path to set a valid dump path!!!")
        else:
            dir_path = os.path.realpath("/")
            dump_file_name = "scope_dump_{}_{}_{}.pkl".format(
                DumpUtil.dump_switch_mode, DumpUtil.dump_switch_scope[0], get_time())
            DumpUtil.dump_path = os.path.join(dir_path, dump_file_name)
            return DumpUtil.dump_path

    @staticmethod
    def get_dump_switch():
        return DumpUtil.dump_switch == "ON"


def set_dump_path(fpath=None):
    if fpath is None:
        raise RuntimeError("set_dump_path '{}' error, please set a valid filename".format(fpath))
    real_path = os.path.realpath(fpath)
    if not os.path.isdir(real_path):
        logger.user_attention(
            "The path '{}' does not exist, the path will be created automatically.".format(real_path))
    DumpUtil.set_ori_dir(real_path)


def generate_dump_path_str():
    if DumpUtil.dump_switch_mode == 'acl':
        if DumpUtil.dump_config == '':
            print_error_log("Please provide dump config for register hook before turning on dump switch!")
            raise DumpException(DumpException.NONE_ERROR)
        dump_path = f"according to dump config {DumpUtil.dump_config}"
    else:
        dump_path = f"to {DumpUtil.dump_ori_dir}"
    return dump_path


def set_dump_switch(switch, mode=Const.ALL, scope=None, api_list=None,
                    filter_switch=Const.ON, dump_mode=Const.ALL, dump_type=Const.ALL,
                    filter_stack=True):
    if scope is None:
        scope = []
    if api_list is None:
        api_list = []

    DumpUtil.set_dump_switch(switch, mode=mode, scope=scope, api_list=api_list,
                             filter_switch=filter_switch, dump_mode=dump_mode, dump_type=dump_type,
                             filter_stack=filter_stack)

    if switch == "ON":
        dump_path_str = generate_dump_path_str()
        logger.user_attention(f"API dump has started. Dump data will be saved {dump_path_str}. ")
        DumpUtil.dump_count = 0
    else:
        dump_path_str = generate_dump_path_str()
        if DumpUtil.dump_count != 0:
            logger.user_attention(f"API dump has been stopped. Dump data has been saved to {dump_path_str}.")
        else:
            logger.user_warning(f"API dump has been stopped, but no data has been saved. Please check the dump scope!")
        DumpUtil.dump_count = 0


def _set_dump_switch4api_list(name):
    if DumpUtil.dump_api_list:
        api_name = get_api_name_from_matcher(name)
        DumpUtil.dump_switch = "ON" if api_name in DumpUtil.dump_api_list else "OFF"


def set_backward_input(backward_input):
    for index, api_name in enumerate(DumpUtil.dump_switch_scope):
        DumpUtil.backward_input[api_name] = backward_input[index]


def make_dump_data_dir(dump_file_name):
    dump_path, file_name = os.path.split(os.path.realpath(dump_file_name))
    name_body, name_extension = os.path.splitext(file_name)
    output_dir = os.path.join(dump_path, f"{name_body}")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir, mode=0o700)
    else:
        shutil.rmtree(output_dir, ignore_errors=True)
        os.mkdir(output_dir, mode=0o700)
    return output_dir


def make_dump_dirs(rank):
    dump_file_name, dump_path = "torch_api_dump_info.pkl", "torch_api_dump"
    dump_stack_file = "torch_api_dump_stack.json"
    dump_root_dir = DumpUtil.dump_ori_dir if DumpUtil.dump_ori_dir else "./"
    Path(dump_root_dir).mkdir(mode=0o700, parents=True, exist_ok=True)
    rank_dir = os.path.join(dump_root_dir, 'rank' + str(rank))
    if not os.path.exists(rank_dir):
        os.mkdir(rank_dir, mode=0o700)
    DumpUtil.dump_dir = rank_dir
    dump_file_path = os.path.join(rank_dir, dump_path)
    dump_file_name = os.path.join(rank_dir, dump_file_name)
    dump_stack_path = os.path.join(rank_dir, dump_stack_file)
    DumpUtil.set_dump_path(dump_file_path, dump_file_name, dump_stack_path)
    DumpUtil.dump_stack_dic = {}


def check_writable(dump_file):
    if not os.access(dump_file, os.W_OK):
        print_error_log(
            'The path {} does not have permission to write. Please check the path permission'.format(
                dump_file))
        raise DumpException(DumpException.INVALID_PATH_ERROR)


def remove_dump_file(dump_file):
    if os.path.exists(dump_file) and not os.path.isdir(dump_file):
        check_writable(dump_file)
        os.remove(dump_file)
