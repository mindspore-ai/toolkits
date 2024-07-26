import json
from pathlib import Path




def load_csv(path):
    ret = []
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line.strip():
                break
            ret.append(json.loads(line))
    return ret


def get_csv_list(path):
    csv = load_csv(path)
    dump_list = []
    for line in csv:
        name = line[0]
        dump_list.append(name)
    return dump_list


def get_stack_list(path):
    with open(path, 'r') as f:
        data = json.load(f)
    stack_list = list(data.keys())
    return stack_list


def get_npy_list(path):
    npy_list = []
    for file in Path(path).glob('*.npy'):
        name = file.stem
        npy_list.append(name)
    return npy_list




def get_csv_npy_stack_list(path, framework):
    assert framework in {
        'torch', 'mindspore'}, "framework must in 'torch' or 'mindspore'"
    csv_list = get_csv_list(path/'rank0'/f'{framework}_api_dump_info.csv')
    npy_list = get_npy_list(path/'rank0'/f'{framework}_api_dump')
    stack_list = get_stack_list(
        path/'rank0'/f'{framework}_api_dump_stack.json')
    return csv_list, npy_list, stack_list

def get_md5_list(path, framework):
    assert framework in {
        'torch', 'mindspore'}, "framework must in 'torch' or 'mindspore'"
    csv = load_csv(path/'rank0'/f'{framework}_api_dump_info.csv')
    md5_position = 6
    md5_list = []
    for line in csv:
        md5 = line[md5_position]
        md5_list.append(md5)
    return md5_list


def get_l2norm_list(path, framework):
    assert framework in {
        'torch', 'mindspore'}, "framework must in 'torch' or 'mindspore'"
    csv = load_csv(path/'rank0'/f'{framework}_api_dump_info.csv')
    l2norm_position = 7
    l2norm_list = []
    for line in csv:
        l2norm = line[l2norm_position]
        l2norm_list.append(l2norm)
    return l2norm_list

def get_summary_list(path, framework):
    assert framework in {
        'torch', 'mindspore'}, "framework must in 'torch' or 'mindspore'"
    csv = load_csv(path/'rank0'/f'{framework}_api_dump_info.csv')
    summary_position = 5
    summary_list = []
    for line in csv:
        summary = line[summary_position]
        summary_list.append(summary)
    return summary_list