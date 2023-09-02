import inspect
import json
import math
import os
import random
import shutil
import stat
import sys
import threading
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import mindspore as ms
import numpy as np

from .utils import (CompareException, Const, __version__, check_mode_valid, get_time,
                    print_attent_log, print_error_log, remove_dump_file)

DumpCount = 0
forward_init_status = False
backward_init_status = False
range_begin_flag, range_end_flag = False, False
NNCount = defaultdict(int)
backward_threading_id = 0


class DumpUtil(object):
    dump_data_dir = None
    dump_ori_dir = None
    dump_path = None
    dump_file_name = None
    dump_stack_file = None
    dump_switch = None
    dump_switch_mode = Const.ALL
    dump_type = Const.ALL
    dump_switch_scope = []
    dump_init_enable = False
    dump_api_list = []
    backward_input = {}
    dump_stack_dic = {}
    dump_filter_switch = None
    dump_filter_stack = True

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
    def set_dump_switch(switch, mode, scope, api_list, filter_switch, dump_mode, dump_type, filter_stack):
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

    def check_list_or_acl_mode(name_prefix):
        global DumpCount
        for item in DumpUtil.dump_switch_scope:
            if name_prefix.startswith(item):
                DumpCount = DumpCount + 1
                return True

    def check_range_mode(name_prefix):
        global range_begin_flag
        global range_end_flag
        if name_prefix.startswith(DumpUtil.dump_switch_scope[0]):
            range_begin_flag = True
            return True
        if name_prefix.startswith(DumpUtil.dump_switch_scope[1]):
            range_end_flag = True
            return True
        if range_begin_flag and not range_end_flag:
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
        Const.LIST: check_list_or_acl_mode,
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
        if DumpUtil.dump_switch is None:
            return False
        return DumpUtil.dump_switch == "ON"


class DataInfo(object):
    def __init__(self, data, save_data, summary_data, dtype, shape):
        self.data = data
        self.save_data = save_data
        self.summary_data = summary_data
        self.dtype = dtype
        self.shape = shape


def get_not_float_tensor_info(data, compute_summary):
    saved_tensor = data.numpy()
    if compute_summary:
        if saved_tensor.size == 0 or saved_tensor.dtype == np.bool_:
            tensor_max = []
            tensor_min = []
            tensor_mean = []
        elif len(saved_tensor.shape) == 0:
            tensor_max = saved_tensor.astype(np.float32).tolist()
            tensor_min = saved_tensor.astype(np.float32).tolist()
            tensor_mean = saved_tensor.astype(np.float32).tolist()
        else:
            tensor_max = saved_tensor.max().astype(np.float32).tolist()
            tensor_min = saved_tensor.min().astype(np.float32).tolist()
            tensor_mean = saved_tensor.astype(np.float32).mean().tolist()
    else:
        tensor_max = math.nan
        tensor_min = math.nan
        tensor_mean = math.nan
    summary_data = [tensor_max, tensor_min, tensor_mean]
    return DataInfo(data, saved_tensor, summary_data, str(data.dtype), tuple(data.shape))


def get_scalar_data_info(data, compute_summary):
    if compute_summary:
        summary_data = [data, data, data]
    else:
        summary_data = [math.nan] * 3
    return DataInfo(data, data, summary_data, str(type(data)), str([]))


def get_float_tensor_info(data, compute_summary):
    saved_tensor = data.numpy()
    if compute_summary:
        tensor_max = saved_tensor.max().astype(np.float32).tolist()
        tensor_min = saved_tensor.min().astype(np.float32).tolist()
        tensor_mean = saved_tensor.mean().astype(np.float32).tolist()
    else:
        tensor_max = math.nan
        tensor_min = math.nan
        tensor_mean = math.nan
    summary_data = [tensor_max, tensor_min, tensor_mean]
    return DataInfo(data, saved_tensor, summary_data, str(data.dtype), tuple(data.shape))


def check_in_api_list(name):
    if not DumpUtil.dump_api_list:
        return True
    name = name.lower()
    for v in DumpUtil.dump_api_list:
        if v in name:
            return True
    return False


def set_dump_path(fpath=None):
    if fpath is None:
        raise RuntimeError("set_dump_path '{}' error, please set a valid filename".format(fpath))
    real_path = os.path.realpath(fpath)
    if not os.path.isdir(real_path):
        print_attent_log(
            "The path '{}' does not exist, the path will be created automatically.".format(real_path))
    DumpUtil.set_ori_dir(real_path)


def set_dump_switch(switch, mode=Const.ALL, scope=None, api_list=None,
                    filter_switch=Const.ON, dump_mode=Const.ALL, dump_type=Const.ALL,
                    filter_stack=True):
    if scope is None:
        scope = []
    if api_list is None:
        api_list = []
    try:
        check_mode_valid(mode)
        assert switch in ["ON", "OFF"], "Please set dump switch with 'ON' or 'OFF'."
        if mode == Const.RANGE:
            assert len(scope) == 2, "set_dump_switch, scope param set invalid, it's must be [start, end]."
        if mode == Const.LIST:
            assert len(scope) != 0, "set_dump_switch, scope param set invalid, it's should not be an empty list."
        if mode == Const.STACK:
            assert len(scope) <= 2, "set_dump_switch, scope param set invalid, it's must be [start, end] or []."
        if mode == Const.ACL:
            assert len(
                scope) == 1, "set_dump_switch, scope param set invalid, only one api name is supported in acl mode."
        if mode == Const.API_LIST:
            assert isinstance(api_list, list) and len(api_list) >= 1, \
                "Current dump mode is 'api_list', but the content of api_list parameter is empty or valid."
    except (CompareException, AssertionError) as err:
        print_error_log(str(err))
        sys.exit()

    DumpUtil.set_dump_switch(switch, mode=mode, scope=scope, api_list=api_list,
                             filter_switch=filter_switch, dump_mode=dump_mode, dump_type=dump_type,
                             filter_stack=filter_stack)

    global DumpCount
    if switch == "ON":
        print_attent_log(f"API dump has started. Dump data will be saved to {DumpUtil.dump_ori_dir}. ")
        if mode == Const.LIST:
            DumpCount = 0
    else:
        print_attent_log(f"API dump has stopped. Dump data has been saved to {DumpUtil.dump_ori_dir}. ")
        if mode == Const.LIST:
            print_attent_log("The number of matched dump is {}".format(DumpCount))


def set_backward_input(backward_input):
    for index, api_name in enumerate(DumpUtil.dump_switch_scope):
        DumpUtil.backward_input[api_name] = backward_input[index]


def dump_data(dump_file_name, dump_step, prefix, data_info, dump_type):
    with os.fdopen(os.open(dump_file_name, os.O_RDWR | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR),
                   "a") as f:
        if json_dump_condition(prefix):
            if dump_type:
                output_path = os.path.join(DumpUtil.dump_data_dir, f'{prefix}.npy')
                np.save(output_path, data_info.save_data)
                os.chmod(output_path, 0o600)
            json.dump([prefix, dump_step, [], data_info.dtype, data_info.shape, data_info.summary_data], f)
            f.write('\n')


def dump_tensor(x, prefix, dump_step, dump_file_name, dump_type):
    global data_info
    if isinstance(x, (tuple, list)) and x:
        for i, item in enumerate(x):
            dump_tensor(item, "{}.{}".format(prefix, i), dump_step, dump_file_name, dump_type)
        return
    elif isinstance(x, ms.Tensor):
        compute_summary = True if dump_type in ['all', 'statistics'] else False
        dump_npy = True if dump_type in ['all', 'npy'] else False
        if x.numel() == 0 or len(x.shape) == 0 or not x.is_floating_point():
            if DumpUtil.dump_filter_switch == Const.OFF:
                data_info = get_not_float_tensor_info(x, compute_summary)
                dump_data(dump_file_name, dump_step, prefix, data_info, dump_npy)
            else:
                return
        else:
            data_info = get_float_tensor_info(x, compute_summary)
            dump_data(dump_file_name, dump_step, prefix, data_info, dump_npy)

    elif DumpUtil.dump_filter_switch == Const.OFF:
        if isinstance(x, bool) or isinstance(x, int) or isinstance(x, float):
            data_info = get_scalar_data_info(x, compute_summary)
            dump_data(dump_file_name, dump_step, prefix, data_info, dump_npy)


def seed_all(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    ms.set_seed(seed)


def make_dump_dirs(rank):
    dump_file_name, dump_path = "mindspore_api_dump_info.pkl", "mindspore_api_dump"
    dump_stack_file = "mindspore_api_dump_stack.json"
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


def json_dump_condition(prefix):
    cur_threading_id = threading.current_thread().ident
    global backward_threading_id
    if not backward_threading_id and Const.BACKWARD in prefix:
        backward_threading_id = cur_threading_id
    return (Const.BACKWARD in prefix and backward_threading_id == cur_threading_id) or 'forward' in prefix


@lru_cache()
def is_not_blacklisted(stack_path):
    black_lists = ['/mindspore/ops/', '/mindspore/nn/', '/mindsproe/amp.py',
                   '/mindspore/numpy/', '/mindspore/dataset/', '/mindspore/common/',
                   '/mindspore/train/', '/mindspore/boost/', '/mindspore/parallel/',
                   '/mindspore/profiler/', '/mindspore/rewrite/', '/mindspore/scipy/',
                   '/troubleshooter/migrator/api_dump/']
    for black in black_lists:
        if black in stack_path:
            return False
    return True


def dump_stack_info(name_template, dump_file, filter_stack):
    if name_template.endswith('_backward_{}'):
        return
    stack_str = []
    prefix = name_template.format("stack_info")
    for (_, path, line, func, code, _) in inspect.stack()[3:]:
        if code:
            stack_line = " ".join([
                "File", ", ".join([path, " ".join(["line", str(line)]), " ".join(["in", func]),
                                   " ".join(["\n", code[0].strip() if code else code])])])
        else:
            stack_line = " ".join([
                "File", ", ".join([path, " ".join(["line", str(line)]), " ".join(["in", func]),
                                   " ".join(["\n", code])])])
        if not filter_stack or (filter_stack and is_not_blacklisted(path)):
            stack_str.append(stack_line)

    DumpUtil.dump_stack_dic[prefix] = stack_str
    json_str = json.dumps(DumpUtil.dump_stack_dic, indent=4)

    with os.fdopen(os.open(dump_file, os.O_RDWR | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR), "w") as f:
        if DumpUtil.dump_switch_mode in Const.DUMP_MODE:
            if json_dump_condition(prefix):
                f.write(json_str)
        else:
            f.write(json_str)


def dump_acc_cmp(name, in_feat, out_feat, dump_step):
    dump_path, dump_file_name, dump_stack_file = DumpUtil.get_dump_path()

    if DumpUtil.get_dump_switch():
        if DumpUtil.dump_init_enable:
            DumpUtil.dump_init_enable = False
            DumpUtil.dump_data_dir = make_dump_data_dir(dump_path)
            remove_dump_file(dump_file_name)
            remove_dump_file(dump_stack_file)

        name_prefix = name
        name_template = f"{name_prefix}" + "_{}"
        if DumpUtil.dump_switch_mode == Const.ALL:
            dump_api_tensor(dump_step, in_feat, name_template, out_feat, dump_file_name, DumpUtil.dump_type)
            dump_stack_info(name_template, dump_stack_file, DumpUtil.dump_filter_stack)
        elif DumpUtil.dump_switch_mode == Const.API_LIST:
            if not check_in_api_list(name):
                return
            dump_api_tensor(dump_step, in_feat, name_template, out_feat, dump_file_name, DumpUtil.dump_type)
            dump_stack_info(name_template, dump_stack_file, DumpUtil.dump_filter_stack)
        elif DumpUtil.dump_switch_mode in [Const.RANGE, Const.LIST]:
            if DumpUtil.check_switch_scope(name_prefix):
                dump_api_tensor(dump_step, in_feat, name_template, out_feat, dump_file_name, DumpUtil.dump_type)
                dump_stack_info(name_template, dump_stack_file, DumpUtil.dump_filter_stack)
        else:
            msg = f"Current mode '{DumpUtil.dump_switch_mode}' is not supported. Please use the field in {Const.DUMP_MODE}"
            raise ValueError(msg)


def dump_api_tensor(dump_step, in_feat, name_template, out_feat, dump_file, dump_type):
    dump_tensor(in_feat, name_template.format("input"), dump_step, dump_file, dump_type)
    if "_backward" not in name_template:
        dump_tensor(out_feat, name_template.format("output"), dump_step, dump_file, dump_type)


def acc_cmp_dump(name, **kwargs):
    dump_step = kwargs.get('dump_step', 1)
    pid = kwargs.get('pid')
    DumpUtil.dump_config = kwargs.get('dump_config')
    name_template = name
    if not pid:
        return RuntimeError("Not get the specified process pid.")

    def acc_cmp_hook(cell, in_feat, out_feat):
        nonlocal name, name_template
        global NNCount
        if "{}_" in name_template:
            # name_template like 'NN_Conv2d_{}_forward'
            nn_name = name_template.split('_')[1]
            if "backward" in name_template:
                id = NNCount[nn_name] - 1
                NNCount[nn_name] = id
            else:
                id = NNCount[nn_name]
                NNCount[nn_name] = id + 1
            name = name_template.format(id)
        if pid == os.getpid():
            dump_acc_cmp(name, in_feat, out_feat, dump_step)

    return acc_cmp_hook
