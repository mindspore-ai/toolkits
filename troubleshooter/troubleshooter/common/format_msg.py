# Copyright 2022-2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""format output"""

import math
import os
import re
import stat
import traceback
from textwrap import fill
import copy

from prettytable import PrettyTable

from troubleshooter.common.ms_utils import get_errmsg_dict
from troubleshooter.common.util import print_line

TABLE_WIDTH = 50
DELIMITER_LEN = 100

# GLog level and level name
_item_to_cn = {
    'item': '项目',
    'desc': '描述',
    'ms_version': '版本信息:',
    'ms_mode': '执行模式:',
    'ms_device': '配置设备:',  # GPU, CPU, Ascend
    'ms_status': '执行阶段:',
    'code_line': '代码行:',
    'cause': '可能原因:',
    'err_code': '示例错误代码：',
    'fixed_code': '示例正确代码：',
    'proposal': '处理建议:',
    'case': '相关案例:',
    'sink_mode': '下沉模式:',  # 图下沉，数据下沉，非下沉
    'case_id': 'ID:',
}


def _add_row(x, item, message, width=TABLE_WIDTH, break_long_words=False, break_on_hyphens=False):
    if message is None:
        return
    item_cn = _item_to_cn.get(item)
    format_message = _format_str_length(
        message) if os.linesep in message else message
    x.add_row([item_cn, fill(format_message, width=width, break_long_words=break_long_words,
                             break_on_hyphens=break_on_hyphens)])


def print_separator_line(content, length=40, character='='):
    separator = f'{content:{character}^{length}}'
    print(separator)

def print_weight_compare_result(result_list, title=None, print_level=1, **kwargs):
    # 0 Do not print
    # Print All
    # print False
    if print_level == 0:
        return
    title = kwargs.get('title', None)
    field_names = kwargs.get('field_names', None)
    x = PrettyTable()
    if title is None:
        x.title = 'The list of comparison results for shapes'
    else:
        x.title = title

    if field_names is None:
        x.field_names = ["Parameter name of original ckpt", "Parameter name of target ckpt", "Whether shape are equal",
                         "Parameter shape of original ckpt", "Parameter shape of target ckpt"]
    else:
        x.field_names = field_names

    for result in result_list:
        if print_level == 1:
            x.add_row([result[0], result[1], result[2], result[3], result[4]])
        elif print_level == 2 and result[2] is not True:
            x.add_row([result[0], result[1], result[2], result[3], result[4]])
    print(x.get_string())
    return x.get_string()


def print_convert_result(result_list):
    x = PrettyTable()
    x.title = 'The list of conversion result'
    x.field_names = ["Parameter name of torch", "Parameter name of MindSpore",
                     "Conversion status",
                     "Parameter shape of torch", "Parameter shape of MindSpore"]
    for result in result_list:
        x.add_row([result[0], result[1], result[2],
                  result[3], result[4]])
    print(x.get_string())


def truncate_decimal(number, decimals):
    if not math.isfinite(number):
        return number
    factor = 10 ** decimals
    truncated_number = int(number * factor) / factor
    return truncated_number


def _format_compare_result(allclose_res, ratio, cos_sim, mean_max):
    ratio = f"{truncate_decimal(ratio, 4):.2%}"
    cos_sim = f"{truncate_decimal(cos_sim, 5):.5f}"
    mean_max = f'{truncate_decimal(mean_max[0], 5):.5f}, {truncate_decimal(mean_max[1], 5):.5f}'
    return [allclose_res, ratio, cos_sim, mean_max]

def _adapter_diff_print(cmp):
    cmp0, cmp1 = truncate_decimal(cmp[0], 5), truncate_decimal(cmp[1], 5)
    if cmp1 != 0:
        diff_ratio = abs(cmp0 - cmp1) / abs(cmp1)
    else:
        diff_ratio = abs(cmp0 - cmp1)
    cmp_uncolored = cmp = f'{cmp0:.5f}, {cmp1:.5f}'
    if diff_ratio > 0.05:
        cmp = f"\033[1;31m{cmp}\033[0m"
    return cmp_uncolored, cmp

def _adapter_format_compare_result(ratio, mean_cmp, max_cmp, min_cmp, cos_sim):
    ratio_flt = ratio
    ratio_uncolored = ratio = f"{truncate_decimal(ratio, 4):.2%}"
    if ratio_flt < 0.95:
        ratio = f"\033[1;31m{ratio}\033[0m"

    cos_sim_flt = cos_sim
    cos_sim_uncolored = cos_sim = f"{truncate_decimal(cos_sim, 5):.5f}"
    if cos_sim_flt < 0.95:
        cos_sim = f"\033[1;31m{cos_sim}\033[0m"

    return [ratio_uncolored, cos_sim_uncolored,
            _adapter_diff_print(mean_cmp)[0], _adapter_diff_print(max_cmp)[0], _adapter_diff_print(min_cmp)[0]], [
            ratio, cos_sim,
            _adapter_diff_print(mean_cmp)[1], _adapter_diff_print(max_cmp)[1], _adapter_diff_print(min_cmp)[1]]

def print_diff_result(result_list, *, print_level=1, title=None, field_names=None,
                      show_shape_diff=False, output_file=None):
    # 0 Do not print
    # Print All
    # print False
    if print_level == 0:
        return
    if not result_list:
        return
    if field_names is None:
        field_names = ["orig array name", "target array name",
                       "result of allclose", "ratio of allclose",
                       "cosine similarity", "mean & max diffs"]
    if show_shape_diff:
        field_names = field_names[:2] + ["shape of orig", "shape of target"] + field_names[2:]

    x = PrettyTable()
    if title is None:
        x.title = 'The list of comparison results'
    else:
        x.title = title
    x.field_names = field_names

    for result in result_list:
        if show_shape_diff:
            orig_name, target_name, orig_shape, target_shape, allclose_res, ratio, cos_sim, mean_max = result
        else:
            orig_name, target_name, allclose_res, ratio, cos_sim, mean_max = result

        if (orig_name is not None and orig_name[:6] == 'LAYER_') or (
            target_name is not None and target_name[:6] == 'LAYER_'):
            continue

        if print_level == 2 and allclose_res is True:
            continue
        compare_res = _format_compare_result(allclose_res, ratio, cos_sim, mean_max)
        if show_shape_diff:
            basic_info = [orig_name, target_name, orig_shape, target_shape]
        else:
            basic_info = [orig_name, target_name]
        x.add_row([*basic_info, *compare_res])
    print(x.get_string())

    if output_file:
        if not os.path.exists(os.path.dirname(output_file)):
            raise ValueError(f"output_file {output_file} not exist")
        with os.fdopen(os.open(output_file, os.O_WRONLY | os.O_CREAT, stat.S_IRUSR | stat.S_IWUSR), 'w') as f:
            f.write(x.get_csv_string(dialect='unix') + os.linesep)
    return x.get_string()

def _parse_layer_name(name):
    pattern = re.compile(
        r"^LAYER_(\w+)?_(backward|forward)_(input|output)\.?(\d+)?$"
    )
    prefix_con = re.findall(pattern, name)
    if len(prefix_con) != 0:
        prefix_con = prefix_con[0]
        new_prefix_con = re.findall(re.compile(r"([A-Za-z0-9_]+?)_([0-9].?)"), prefix_con[0])
        if len(new_prefix_con) != 0:
            name = "LAYER: " + new_prefix_con[0][0] + "." + new_prefix_con[0][-1]
        else:
            name = "LAYER: " + prefix_con[0]

    line_len = (45 - len(name)) // 2
    name = "-" * line_len + name + "-" * line_len
    if len(name) == 44 :
        name = "-" + name
    return name

def _parse_api_name(name):
    pattern = re.compile(
        r"^(\w+?)_(\w+)_(\d+)+_(backward|forward)_([io][nu]t?put)\.?(\d+)?\.?(\d+)?$"
    )
    prefix_con = re.findall(pattern, name)
    is_output = False
    if len(prefix_con) != 0:
        prefix_con = prefix_con[0]
        suffix = prefix_con[1] + "(" + prefix_con[2] + ") " + prefix_con[4] \
                + ("[" + prefix_con[5] + "]" if prefix_con[5] != "" else "") \
                + ("[" + prefix_con[6] + "]" if prefix_con[6] != "" else "")
        if prefix_con[0] == 'Tensor':
            name = 'torch.Tensor.' + suffix
        elif prefix_con[0] == 'Torch':
            name = 'torch.' + suffix
        elif prefix_con[0] == 'NN':
            name = 'torch.nn.' + suffix
        elif prefix_con[0] == 'Functional':
            name = 'torch.nn.functional.' + suffix
        # don't draw line above input.0 because it might not be dumped if it's not float tensor
        if prefix_con[4] == 'output':
            is_output = True
    return name, is_output

def print_adapter_diff_result(result_list, *, print_level=1, title=None, field_names=None, show_dtype_diff=False,
                      show_shape_diff=False, output_file=None, frame_names=None):

    orig_frame, tgt_frame = frame_names
    orig_frame = "pt" if orig_frame == 'pytorch' else 'mt'
    tgt_frame = "pt" if tgt_frame == 'pytorch' else 'mt'

    # 0 Do not print
    # Print All
    # print False
    if print_level == 0:
        return
    if not result_list:
        return
    if field_names is None:
        field_names = ["orig array name", "target array name", "ratio of allclose", "cosine similarity",
                       "mean cmp (orig, target)", "max cmp (orig, target)", "min cmp (orig, target)"]
    column_width = [45, 45, 17, 17, 25, 25, 25]
    if show_shape_diff:
        field_names = field_names + [f"shape cmp ({orig_frame}, {tgt_frame})"]
        column_width += [18]
    if show_dtype_diff:
        field_names = field_names + [f"dtype cmp ({orig_frame}, {tgt_frame})"]
        column_width += [18]

    x = PrettyTable()
    x._max_width = dict(zip(field_names, column_width))
    x.padding_width = 0
    if title is None:
        x.title = 'The list of comparison results'
    else:
        x.title = title
    x.field_names = field_names
    csv_x = copy.deepcopy(x)

    is_below_output = is_below_layer = False
    for result in result_list:
        if show_shape_diff:
            if show_dtype_diff:
                orig_name, target_name, orig_dtype, target_dtype, orig_shape, target_shape, ratio, \
                    mean_cmp, max_cmp, min_cmp, cosine_sim = result
            else:
                orig_name, target_name, orig_shape, target_shape, ratio, mean_cmp, max_cmp, min_cmp, \
                    cosine_sim = result
        else:
            orig_name, target_name, ratio, mean_cmp, max_cmp, min_cmp, cosine_sim = result

        if print_level == 2 and ratio == 1:
            continue

        is_layer = False
        if target_name is not None:
            if target_name[:6] == "LAYER_":
                target_name = _parse_layer_name(target_name)
                is_layer = True
            else:
                target_name, is_output = _parse_api_name(target_name)
        if orig_name is not None:
            if orig_name[:6] == "LAYER_":
                orig_name = _parse_layer_name(orig_name)
                is_layer = True
            else:
                orig_name, is_output = _parse_api_name(orig_name)

        name_info = [orig_name, target_name]
        basic_info = []
        basic_info_uncolored = []
        if show_shape_diff:
            if orig_shape !=  target_shape:
                shape_cmp_uncolored = f"{orig_shape}, {target_shape}"
                shape_cmp = f"\033[1;31m{shape_cmp_uncolored}\033[0m"
            else:
                shape_cmp_uncolored = shape_cmp = str(orig_shape)
            basic_info = [shape_cmp]
            basic_info_uncolored = [shape_cmp_uncolored]
            if show_dtype_diff:
                if orig_dtype !=  target_dtype:
                    dtype_cmp_uncolored = f"{orig_dtype}, {target_dtype}"
                    dtype_cmp = f"\033[1;31m{dtype_cmp_uncolored}\033[0m"
                else:
                    dtype_cmp_uncolored = dtype_cmp = str(orig_dtype)
                basic_info = [shape_cmp, dtype_cmp]
                basic_info_uncolored = [shape_cmp_uncolored, dtype_cmp_uncolored]
        if is_layer:
            x.add_row([*name_info] + ["-" * i for i in column_width[2:]])
            csv_x.add_row([*name_info] + ["-" * i for i in column_width[2:]])
            is_below_layer = True
        else:
            if is_below_output and not is_below_layer:
                x.add_row(["-" * i for i in column_width])
                csv_x.add_row(["-" * i for i in column_width])
            compare_res_uncolored, compare_res = _adapter_format_compare_result(ratio, mean_cmp,
                                                                                max_cmp, min_cmp, cosine_sim)
            x.add_row([*name_info, *compare_res, *basic_info])
            csv_x.add_row([*name_info, *compare_res_uncolored, *basic_info_uncolored])
            is_below_output = True if is_output else False
            is_below_layer = False
    print(x.get_string())

    if output_file:
        if not os.path.exists(os.path.dirname(output_file)):
            raise ValueError(f"output_file {output_file} not exist")
        with os.fdopen(os.open(output_file, os.O_WRONLY | os.O_CREAT, stat.S_IRUSR | stat.S_IWUSR), 'w') as f:
            f.write(csv_x.get_csv_string(dialect='unix') + os.linesep)
    return x.get_string()

def print_net_infer_diff_result(result_list):
    x = PrettyTable()
    x.title = 'The list of comparison results'
    x.field_names = ["Pytorch data", "MindSpore data",
                     "result of allclose", "cosine similarity", "mean & max diffs"]
    for result in result_list:
        x.add_row([result[0], result[1], result[2], result[3], result[4]])
    print(x.get_string())


def print_result(expert_experience, write_file_path):
    """
    print MindSpore FAR
    """
    x = PrettyTable()
    item_desc = _item_to_cn
    x.title = 'MindSpore FAR(Failure Analysis Report)'
    x.field_names = [item_desc.get("item"), item_desc.get("desc")]
    x.align[item_desc.get("desc")] = 'l'
    mindspore_version = expert_experience.get("mindspore_version")
    mindspore_mode = expert_experience.get("mindspore_mode")
    mindspore_device = expert_experience.get("Device Type")
    x.add_row([item_desc.get("ms_version"), fill(
        mindspore_version, width=TABLE_WIDTH)])
    x.add_row([item_desc.get("ms_mode"), fill(
        mindspore_mode, width=TABLE_WIDTH)])
    if mindspore_device:
        x.add_row([item_desc.get("ms_device"), fill(
            mindspore_device, width=TABLE_WIDTH)])
    ms_status = expert_experience.get("ms_status")
    code_line = expert_experience.get("code_line")
    sink_mode = expert_experience.get("Sink Mode")
    if ms_status:
        x.add_row([item_desc.get("ms_status"),
                  fill(ms_status, width=TABLE_WIDTH)])
    if code_line:
        x.add_row([item_desc.get("code_line"),
                  fill(code_line, width=TABLE_WIDTH)])
    if sink_mode:
        x.add_row([item_desc.get("sink_mode"),
                  fill(sink_mode, width=TABLE_WIDTH)])

    # 可能原因
    fault_cause = expert_experience.get('Fault Cause')
    _add_row(x, "cause", fault_cause)

    # 错误代码
    err_code = expert_experience.get("Error Case")
    err_code = _format_code_str(err_code)

    _add_row(x, "err_code", err_code)
    # 处理建议
    suggestion = expert_experience.get("Modification Suggestion")
    _add_row(x, "proposal", suggestion)
    # 正确代码
    fixed_code = expert_experience.get("Fixed Case")
    fixed_code = _format_code_str(fixed_code)
    _add_row(x, "fixed_code", fixed_code)
    # 相关案例
    fault_case = expert_experience.get("Fault Case")
    fault_case = _format_case_str(fault_case, mindspore_version)
    _add_row(x, "case", fault_case)
    if write_file_path:
        case_id = expert_experience.get("ID")
        _add_row(x, "case_id", case_id)
        file = os.path.join(
            write_file_path, "mindspore_failure_analysis_report.log")
        with open(file, "w") as f:
            f.write(x.get_string() + os.linesep)
    print(x.get_string())


def _print_msg(title, msg=None, print_msg=True):
    if msg:
        print_line("-", DELIMITER_LEN)
        print("-  " + title)
        print_line("-", DELIMITER_LEN)
        if print_msg:
            print(msg.rstrip(os.linesep) + os.linesep)


def _print_stack(exc_traceback_obj):
    if exc_traceback_obj:
        _print_msg("Python Traceback (most recent call last):", "NULL", False)
        traceback.print_tb(exc_traceback_obj)
        print("")


def _format_str_length(string):
    str_list = string.split(os.linesep)
    result_str = ""
    for str_tmp in str_list:
        result_str = result_str + str_tmp.ljust(TABLE_WIDTH) + os.linesep
    return result_str


def _format_code_str(content, width=50):
    """
    format code str
    """
    if content:
        lines = content.split("\n")
        result = '+'.ljust(width - 2, '-') + '+' + os.linesep
        line = lines[1]
        if line == '':
            return content
        j = 0
        while line[j] == ' ':
            j = j + 1
        for line in lines:
            if line == lines[0]:
                continue
            line = line[j:]
            pre_line = "> " + line + os.linesep
            result += pre_line
        result += '+'.ljust(width - 2, '-') + '+'
        content = result
    return content


# replace the case link mindspore version to match with current mindspore version


# replace mindspore version in link
def _replace(link, keys, target):
    """
    :param link: str, the case web page's link
    :param keys: list[str], regular expressions for different web page, like note, api and faq.
    :param target: list[str], the link keywords of web pages of note, api and faq, corresponding to current
                   mindspore version
    :return: link: replaced web page's link, corresponding to current mindspore version
    """
    for i, key in enumerate(keys):
        match = re.search(key, link)
        if match:
            link = link[:match.start()] + target[i] + link[match.end():]
            break
    return link


def _replace_link_version(link, link_version, mindspore_version):
    """
    replace url link version for diff version
    """
    if mindspore_version < link_version:
        return link
    if mindspore_version < "r1.7" or link_version >= "r1.7":
        match = re.search(link_version, link)
        link = link[:match.start()] + mindspore_version + link[match.end():]
        return link
    if link_version < "r1.7" <= mindspore_version:
        keys = [r"note/zh-CN/{}".format(link_version),
                r"api/zh-CN/{}".format(link_version),
                r"faq/zh-CN/{}".format(link_version)]
        target = ["zh-CN/{}/note".format(mindspore_version),
                  "zh-CN/{}".format(mindspore_version),
                  "zh-CN/{}/faq".format(mindspore_version)]
        link = _replace(link, keys, target)
    return link


def _format_case_str(content, mindspore_version):
    """
    format case string
    """
    if content:
        # match, no replace
        lines = content.split(os.linesep)
        result = ""
        for line in lines:
            line = line.lstrip()
            match = re.search(mindspore_version, line)
            if not match:
                key = r"(r1.[6-9])|(r2.[0-9])"
                match = re.search(key, line)
                # no mindspore link, return
                if match:
                    link_version = line[match.start():match.end()]
                    line = _replace_link_version(
                        line, link_version, mindspore_version)
                result += line + os.linesep
            else:  # link version same with mindspore version, no replace
                return content
        content = result
    return content


def _filter_stack(stack):
    result = True
    black_stack_list = ["mindspore/nn/cell.py.*in __call__",
                        "mindspore/nn/cell.py.*in run_construct",
                        "mindspore/nn/layer/basic.py.*in proposal_wrapper",
                        "mindspore/common/api.py.*in wrapper",
                        "mindspore/common/api.py.*in real_run_op",
                        "mindspore/ops/primitive.py.*in __call__",
                        "proposal_action.py.*in proposal_wrapper",
                        "mindspore/nn/cell.py.*in _run_construct",
                        "mindspore/ops/primitive.py.*in __check__"]
    for black_stack in black_stack_list:
        if re.search(black_stack, stack):
            result = False
            break
    return result


def print_clear_exception(exc_type, exc_value, exc_traceback_obj):
    if exc_traceback_obj:
        _print_msg(
            "[TroubleShooter-Clear Stack] Python Traceback (most recent call last):", "NULL", False)
        org_err_stack = traceback.format_exception(
            exc_type, exc_value, exc_traceback_obj)
        for stack in org_err_stack:
            if _filter_stack(stack):
                print(stack.rstrip(os.linesep))


def print_format_exception(exc_type, exc_value, exc_traceback_obj):
    """
    print format exception, for old mindspore version
    """
    import mindspore
    ms_version = mindspore.__version__[:3]
    if ms_version >= '1.8':
        traceback.print_exc()
        return

    msg_dict = get_errmsg_dict(exc_type, exc_value)
    _print_stack(exc_traceback_obj)
    _print_msg("Error Message:", msg_dict.get("err_msg"))

    if msg_dict.get("construct_stack_msg"):
        _print_msg("The Traceback of Net Construct Code:",
                   msg_dict.get("construct_stack_msg"))
    else:
        _print_msg("The Traceback of Net Construct Code:",
                   msg_dict.get("construct_stack_in_file_msg"))
    _print_msg("C++ Function:", msg_dict.get("cpp_fun_msg"))
    _print_msg("Inner Message:", msg_dict.get("abstract_inner_msg"))


def format_error_message(error_message):
    """
    format error message, from string to dict
    """
    msg_list = error_message.split(
        '----------------------------------------------------')
    format_msg_dict = {}
    current_key = None
    for msg in msg_list:
        if msg_list.index(msg) == 0:
            format_msg_dict["error_message"] = msg
            continue
        msg = msg.strip().strip(os.linesep)
        if msg.startswith("- "):
            current_key = msg[2:]
            continue
        if current_key:
            format_msg_dict[current_key] = msg
    return format_msg_dict
