import re
import os
import glob
import pandas as pd
from tqdm import tqdm
from recompute_profile.static_value import NOT_OVERLAP_TIME

def contains_two_defaults(s, prefix_default):
    if s.count(prefix_default) > 2:
        print("Count Default > 2: ", s)
    return s.count(prefix_default) == 2

def prefix_split(ops_name, prefix):
    if prefix in ops_name:
        ops_name = ops_name[len(prefix):]
    return ops_name

def prefix_default_add_value(name, prefix, prefix_default):
    ops_name = name
    if "/" in ops_name:
        ops_name = ops_name.rsplit("/", 1)[0]
        if ops_name.endswith(prefix_default):
            ops_name = ops_name.rsplit("/", 1)[0]
    if "Default" == ops_name:
        return name
    ops_name = prefix_split(ops_name, prefix)
    return ops_name

def process(detail_df, summary):
    prefix = "Kernel::KernelLaunch::"
    iter_index = 0
    prefix_default = "Default"
    op_flag = "-op"
    op_flag_move_value = {}

    scope_data = {}
    print("--------analy kernel details start-----------")
    for index, row in enumerate(detail_df):
        ops_name = row['Name']
        if prefix in ops_name:
            iter_index = index
        prefix_default_ok = False
        if ops_name.startswith(prefix_default):
            ops_name_temp = ops_name.split("Default/", 1)[1]
            iter_index_temp = iter_index
            if ops_name_temp in detail_df[iter_index_temp]["Name"]:
                scope_data[ops_name] = scope_data[detail_df[iter_index_temp]["Name"]]
            else:
                scope_data[ops_name] = prefix_default_add_value(ops_name, prefix, prefix_default)
            prefix_default_ok = True

        prefix_default_two_ok = False
        if contains_two_defaults(ops_name, prefix_default) and (not prefix_default_ok):
            scope_data[ops_name] = prefix_default_add_value(ops_name, prefix, prefix_default)
            prefix_default_two_ok = True

        op_flag_ok = False
        if op_flag in ops_name and (not prefix_default_ok) and (not prefix_default_two_ok):
            ops_name_temp_first = ops_name.rsplit(op_flag, 1)[0]
            ops_name_temp_scend = ops_name.rsplit(op_flag, 1)[1]
            ops_name_temp_scend = ops_name_temp_scend if "/" not in ops_name_temp_scend else ops_name_temp_scend.split("/", 1)[0]
            ops_name_temp = ops_name_temp_first + op_flag + ops_name_temp_scend
            ops_name_temp = prefix_split(ops_name_temp, prefix)
            scope_data[ops_name] = ops_name_temp
            op_flag_ok = True
        if not op_flag_ok and not prefix_default_ok and not prefix_default_two_ok:
            print(ops_name)
            raise Exception("Ops name detect error!")
        
        if op_flag in ops_name:
            ops_name_temp = scope_data[ops_name]
            ops_name_temp_first = ops_name_temp.rsplit(op_flag, 1)[0]
            ops_name_temp_scend = ops_name_temp.rsplit(op_flag, 1)[1]
            ops_name_temp_scend = ops_name_temp_scend if "/" not in ops_name_temp_scend else ops_name_temp_scend.split("/", 1)[0]
            if ops_name_temp_first not in op_flag_move_value:
                op_flag_move_value[ops_name_temp_first] = int(ops_name_temp_scend)
            else:
                op_flag_move_value[ops_name_temp_first] = min(int(ops_name_temp_scend), op_flag_move_value[ops_name_temp_first])
    # 序号偏移纠正
    for key, value in scope_data.items():
        ops_name_temp_first = value.rsplit(op_flag, 1)[0]
        if ops_name_temp_first not in op_flag_move_value:
            continue
        ops_name_temp_scend = value.rsplit(op_flag, 1)[1]
        ops_name_temp_scend = ops_name_temp_scend if "/" not in ops_name_temp_scend else ops_name_temp_scend.split("/", 1)[0]
        ops_name_temp_scend = (int)(ops_name_temp_scend) - op_flag_move_value[ops_name_temp_first]
        new_name = ops_name_temp_first + op_flag + str(ops_name_temp_scend)
        scope_data[key] = new_name

    print("--------update recompute information start-----------")
    for index, row in enumerate(detail_df):
        ops_name = scope_data[row['Name']]
        if ops_name in summary:
            summary[ops_name]["visited"] = 1
            if summary[ops_name]["recompute"] == 1:
                detail_df[index]["recompute"] = "recompute"
            else:
                detail_df[index]["recompute"] = "no-recompute"
            if summary[ops_name]["dp"] == 1:
                detail_df[index]["is_dp_allgather"] = "dp"
            detail_df[index]["unique_id"] = summary[ops_name]["unique_id"]
            detail_df[index]["forward_unique_id"] = summary[ops_name]["forward_id"]
            detail_df[index]["micro"] = summary[ops_name]["micro"]
        else:
            detail_df[index]["recompute"] = "no-match"
    return detail_df, summary

def analy_kernel_details(detail_df, summary):
    detail_df, summary = process(detail_df, summary)
    return detail_df, summary