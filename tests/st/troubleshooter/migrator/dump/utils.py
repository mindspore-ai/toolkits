import json
from pathlib import Path


def load_pkl(path):
    ret = []
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line.strip():
                break
            ret.append(json.loads(line))
    return ret


def get_pkl_list(path):
    pkl = load_pkl(path)
    dump_list = []
    for line in pkl:
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


def get_pkl_npy_stack_list(path, framework):
    assert framework in {
        'torch', 'mindspore'}, "framework must in 'torch' or 'mindspore'"
    pkl_list = get_pkl_list(path/'rank0'/f'{framework}_api_dump_info.pkl')
    npy_list = get_npy_list(path/'rank0'/f'{framework}_api_dump')
    stack_list = get_stack_list(
        path/'rank0'/f'{framework}_api_dump_stack.json')
    return pkl_list, npy_list, stack_list

def get_md5_list(path, framework):
    assert framework in {
        'torch', 'mindspore'}, "framework must in 'torch' or 'mindspore'"
    pkl = load_pkl(path/'rank0'/f'{framework}_api_dump_info.pkl')
    md5_list = []
    for line in pkl:
        md5 = line[6]
        md5_list.append(md5)
    return md5_list