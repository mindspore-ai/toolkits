import re
import os
import glob
import pandas as pd

global micro_max_value
def extract_graph_build_ir(ir_file_path_dir):
    files = glob.glob(os.path.join(ir_file_path_dir, 'graph_build*'))
    files.sort()
    return files

def contains_dp_flag(line):
    """Determine whether the string contains dp's allgather."""
    pattern = r'= AllGather.*parallel_optimizer'
    return bool(re.search(pattern, line))

def contains_micro_flag(line):
    """Determine whether the string contains micro."""
    pattern = r'micro:\s*I64\((\d+)\)'
    match = re.search(pattern, line)
    if match:
        return str(match.group(1))
    else:
        return None

def extract_unique_id(input_string):
    forward_pattern = r'forward_unique_id:\s*"(\d+)"'
    unique_pattern = r'{unique_id:\s*"(\d+)"'
    forward_match = re.search(forward_pattern, input_string)
    unique_match = re.search(unique_pattern, input_string)
    forward_id = forward_match.group(1) if forward_match else None
    unique_id = unique_match.group(1) if unique_match else None
    return forward_id, unique_id

def check_re(line):
    dp_flag = contains_dp_flag(line)
    micro_flag = contains_micro_flag(line)
    unique_id_flag = extract_unique_id(line)
    return dp_flag, micro_flag is not None, micro_flag, unique_id_flag[0] is not None,\
           unique_id_flag[0], unique_id_flag[1] is not None, unique_id_flag[1]

def fill_re(op_attrs, dp_flag, micro_flag, micro, forward_id_flag,forward_id, unique_id_flag, unique_id, forward_unique_id_dict):
    op_attrs["dp"] = 1 if dp_flag else 0
    if micro_flag:
        op_attrs["micro"] = micro
        global micro_max_value
        micro_max_value = max(micro_max_value, int(micro))
    else:
        op_attrs["micro"] = ""
    op_attrs["unique_id"] = unique_id if unique_id_flag else ""
    if forward_id_flag:
        op_attrs["forward_id"] = forward_id
        if forward_id not in forward_unique_id_dict:
            forward_unique_id_dict[forward_id] = []
    else:
        op_attrs["forward_id"] = ""

def extract_recompute_ops_from_graph_build_ir(ir_file_path, summary, forward_unique_id_dict):
    dp_flag, micro_flag, forward_id_flag, unique_id_flag = False, False, False, False
    micro, forward_id, unique_id= "", "", ""
    with open(ir_file_path, "r", encoding="utf-8") as file:
        line = file.readline()
        while line:
            if not dp_flag and not micro_flag and not forward_id_flag and not unique_id_flag:
                dp_flag, micro_flag, micro, forward_id_flag,forward_id, unique_id_flag, unique_id = check_re(line)
            if "duplicated: Bool(1)" in line:
                dtype_and_shape = file.readline()
                fullname = file.readline()
                while "Fullname with scope" not in fullname:
                    fullname = file.readline()
                fullname = re.search(r"Fullname with scope: \((.*)\)", fullname).group(1)
                op_attrs = {}
                op_attrs["recompute"] = 1
                op_attrs["visited"] = 0
                fill_re(op_attrs, dp_flag, micro_flag, micro, forward_id_flag,forward_id,
                        unique_id_flag, unique_id, forward_unique_id_dict)
                dp_flag, micro_flag, forward_id_flag, unique_id_flag = False, False, False, False
                micro, forward_id, unique_id = "", "", ""
                if fullname in summary.keys():
                    raise Exception("extract ir graph information error.")
                summary[fullname] = op_attrs
            else:
                fullname = line
                if "Fullname with scope" in fullname:
                    fullname = re.search(r"Fullname with scope: \((.*)\)", fullname).group(1)
                    op_attrs = {}
                    op_attrs["recompute"] = 0
                    op_attrs["visited"] = 0
                    fill_re(op_attrs, dp_flag, micro_flag, micro, forward_id_flag, forward_id, unique_id_flag,
                            unique_id, forward_unique_id_dict)
                    dp_flag, micro_flag, forward_id_flag, unique_id_flag = False, False, False, False
                    micro, forward_id, unique_id = "", "", ""
                    if fullname in summary.keys():
                        raise Exception("extract ir graph information error.")
                    summary[fullname] = op_attrs

            line = file.readline()
    return summary

def analy_recompute_by_unique_id(summary, forward_unique_id_dict):
    data = 0
    global micro_max_value
    for key, value in summary.items():
        if value["unique_id"] == "":
            continue
        if value["unique_id"] in forward_unique_id_dict:
            forward_unique_id_dict[value["unique_id"]].append(key)
    for key, value in forward_unique_id_dict.items():
        if len(value) != 2 * micro_max_value:
            continue
        for index in range(1, len(value), 2):
            if summary[value[index]]["recompute"] != 1:
                data+=1
            summary[value[index]]["recompute"] = 1
    print("only unique_id recompute num:", data)

def analy_unique_id_data(summary):
    unique_id_num = 0
    forward_unique_id_num = 0
    for key, value in summary.items():
        if value["unique_id"] != "":
            unique_id_num += 1
        if value["forward_id"] != "":
            forward_unique_id_num += 1
    print("unique_id_num:", unique_id_num)
    print("forward_unique_id:", forward_unique_id_num)

def extract_recompute_ops_from_ir_floder(ir_file_path_dir):
    print("--------analy ir graph start-----------")
    global micro_max_value
    micro_max_value = -1
    ir_file_paths = extract_graph_build_ir(ir_file_path_dir)
    summary = {}
    forward_unique_id_dict = {}
    for ir_file_path in ir_file_paths:
        extract_recompute_ops_from_graph_build_ir(ir_file_path, summary, forward_unique_id_dict)
    micro_max_value += 1
    analy_recompute_by_unique_id(summary, forward_unique_id_dict)
    analy_unique_id_data(summary)
    return summary

def dump_as_file(summary, output_file):
    df = pd.DataFrame(summary).transpose()
    df.to_excel(output_file)