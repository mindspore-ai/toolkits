import os
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

from troubleshooter.migrator.diff_handler import compare_npy_dir, min_edit_distance

from .apis_match import APIList, _print_apis_map_result, flow_match, load_pkl

__all__ = ['api_dump_compare']


def get_npy_map_list(
    apis_map: List,
    origin_npy_dir: str,
    target_npy_dir: str,
    origin_pkl_path: str,
    target_pkl_path: str,
    ignore_backward: bool = False,
    ignore_unmatched: bool = False,
):
    """covert apis_map to npy_map_list

    Args:
        apis_map (List): 通过toolkit中的api_match生成的api映射表
        origin_npy_dir (str): 原网络dump出的npy文件目录
        target_npy_dir (str): 目标网络dump出的npy文件目录
        origin_pkl_path (str): 原网络dump出的pkl文件路径
        target_pkl_path (str): 目标网络dump出的pkl文件路径
        ignore_backward (bool, optional): 是否忽略反向npy的比对. Defaults to False.

    Returns:
        List: npy文件映射表
    """

    def _get_npy_list(apis, io, file_dict):
        def _sorted_key(x):
            pattern = re.compile(
                r"^(\w+?)_(\w+)_(\d+)+_(\w+)_([io][nu]t?put)\.?(\d+)?\.?(\d+)?$"
            )
            prefix_con = re.findall(pattern, x[:-4])
            if len(prefix_con) == 0:
                print(f"ignore {x}")
                return None
            prefix_con = prefix_con[0]
            if prefix_con[5] == "":
                id0 = 0
                id1 = 0
            elif prefix_con[6] == "":
                id0 = int(prefix_con[5])
                id1 = 0
            else:
                id0 = int(prefix_con[5])
                id1 = int(prefix_con[6])
            return id0, id1

        npy_list = file_dict[str(apis)]
        forward_npy_list = []
        backward_npy_list = []

        for name in npy_list:
            if io in name:
                if 'forward' in name:
                    forward_npy_list.append(name)
                elif 'backward' in name:
                    backward_npy_list.append(name)

        forward_npy_list = sorted(forward_npy_list, key=_sorted_key)
        backward_npy_list = sorted(backward_npy_list, key=_sorted_key)
        return forward_npy_list, backward_npy_list

    def _get_npy_shape_map(pkl_path):
        def _read_line(line):
            prefix, dump_step, _, data_type, data_shape, data_summary = line
            return {prefix: data_shape}

        ret = {}

        pkl = load_pkl(pkl_path)
        for l in pkl:
            shape = _read_line(l)
            if shape:
                ret.update(shape)
        return ret

    def _get_name_map_list(
        origin_name_list, target_name_list, origin_shape_map, target_shape_map
    ):
        origin_shape_list = [
            origin_shape_map[origin_name[:-4]] for origin_name in origin_name_list
        ]
        target_shape_list = [
            target_shape_map[target_name[:-4]] for target_name in target_name_list
        ]
        _, index_map_list = min_edit_distance(origin_shape_list, target_shape_list)

        name_map_list = [
            (
                origin_name_list[a] if a is not None else None,
                target_name_list[b] if b is not None else None,
            )
            for a, b in index_map_list
        ]
        return name_map_list

    def _get_npy_map(
        origin_npy_list,
        target_npy_list,
        origin_shape_map,
        target_shape_map,
        ignore_unmatched,
    ):
        if len(origin_npy_list) == 0:
            if len(target_npy_list) == 0:
                return []
            else:
                if ignore_unmatched:
                    return []
                return [(None, i) for i in target_npy_list]
        else:
            if len(target_npy_list) == 0:
                if ignore_unmatched:
                    return []
                return [(i, None) for i in origin_npy_list]
            else:
                ret = _get_name_map_list(
                    origin_npy_list, target_npy_list, origin_shape_map, target_shape_map
                )
                if ignore_unmatched:
                    ret = [(i, j) for i, j in ret if i is not None and j is not None]
                return ret

    forward_list = []
    backward_list = []
    origin_shape_map = _get_npy_shape_map(origin_pkl_path)
    target_shape_map = _get_npy_shape_map(target_pkl_path)

    origin_npy_files = defaultdict(list)
    target_npy_files = defaultdict(list)
    for name in Path(origin_npy_dir).glob('*.npy'):
        name = name.name
        key = name[0 : name.rfind('_', 0, name.rfind('_'))]
        origin_npy_files[key].append(name)
    for name in Path(target_npy_dir).glob('*.npy'):
        name = name.name
        key = name[0 : name.rfind('_', 0, name.rfind('_'))]
        target_npy_files[key].append(name)

    for origin_apis, target_apis in apis_map:
        if len(origin_apis) != 0:
            origin_input_forward, origin_input_backward = _get_npy_list(
                origin_apis[0], 'input', origin_npy_files
            )
            origin_output_forward, origin_output_backward = _get_npy_list(
                origin_apis[-1], 'output', origin_npy_files
            )
        else:
            origin_input_forward = []
            origin_input_backward = []
            origin_output_forward = []
            origin_output_backward = []
        if len(target_apis) != 0:
            target_input_forward, target_input_backward = _get_npy_list(
                target_apis[0], 'input', target_npy_files
            )
            target_output_forward, target_output_backward = _get_npy_list(
                target_apis[-1], 'output', target_npy_files
            )
        else:
            target_input_forward = []
            target_input_backward = []
            target_output_forward = []
            target_output_backward = []

        input_forward_map = _get_npy_map(
            origin_input_forward,
            target_input_forward,
            origin_shape_map,
            target_shape_map,
            ignore_unmatched,
        )
        forward_list += input_forward_map
        if not ignore_backward:
            input_backward_map = _get_npy_map(
                origin_input_backward,
                target_input_backward,
                origin_shape_map,
                target_shape_map,
                ignore_unmatched,
            )
            backward_list += input_backward_map

        if not ignore_unmatched:
            if origin_apis[1:-1]:
                origin_inter_forward, origin_inter_backward = zip(
                    # [([a_forward_output.0.npy, a_forward_output.1.npy], [a_backward_output.1.npy]),
                    #  ([b_forward_output.npy], [b_backward_output.npy])]
                    # -> ([a_forward_output.0.npy, a_forward_output.1.npy], [b_forward_output.npy])
                    # -> ([a_backward_output.1.npy], [b_backward_output.npy])
                    *[_get_npy_list(api, '', origin_npy_files) for api in origin_apis[1:-1]]
                )
                origin_inter_forward_map = [
                    (origin, None) for origins in origin_inter_forward for origin in origins
                ]

                forward_list.extend(origin_inter_forward_map)
                if not ignore_backward:
                    origin_inter_backward_map = [
                        (origin, None) for origins in origin_inter_backward for origin in origins
                    ]
                    backward_list.extend(origin_inter_backward_map)
            if target_apis[1:-1]:
                target_inter_forward, target_inter_backward = zip(
                    *[_get_npy_list(api, '', target_npy_files) for api in target_apis[1:-1]]
                )
                target_inter_forward_map = [
                    (None, target) for targets in target_inter_forward for target in targets
                ]

                forward_list.extend(target_inter_forward_map)
                if not ignore_backward:
                    target_inter_backward_map = [
                        (None, target) for targets in target_inter_backward for target in targets
                    ]
                    backward_list.extend(target_inter_backward_map)

        output_forward_map = _get_npy_map(
            origin_output_forward,
            target_output_forward,
            origin_shape_map,
            target_shape_map,
            ignore_unmatched,
        )
        forward_list += output_forward_map
        if not ignore_backward:
            output_backward_map = _get_npy_map(
                origin_output_backward,
                target_output_backward,
                origin_shape_map,
                target_shape_map,
                ignore_unmatched,
            )
            backward_list += output_backward_map

    return forward_list, backward_list


def get_dump_path(root_path):
    if not os.path.exists(os.path.join(root_path, 'rank0')):
        return None

    if os.path.exists(
        os.path.join(root_path, 'rank0', 'mindspore_api_dump')
    ) and os.path.exists(
        os.path.join(root_path, 'rank0', 'mindspore_api_dump_info.pkl')
    ):
        return (
            os.path.join(root_path, 'rank0', 'mindspore_api_dump'),
            os.path.join(root_path, 'rank0', 'mindspore_api_dump_info.pkl'),
            'mindspore',
        )
    elif os.path.exists(
        os.path.join(root_path, 'rank0', 'torch_api_dump')
    ) and os.path.exists(os.path.join(root_path, 'rank0', 'torch_api_dump_info.pkl')):
        return (
            os.path.join(root_path, 'rank0', 'torch_api_dump'),
            os.path.join(root_path, 'rank0', 'torch_api_dump_info.pkl'),
            'pytorch',
        )
    else:
        return None


def api_dump_compare(origin_path: str, target_path: str, output_path: Optional[str] = None,
                     *, rtol: float = 1e-4, atol: float = 1e-4, equal_nan: bool = False,
                     ignore_unmatched: bool = False, **kwargs):
    """compare each api's output by the npy files dumped by the network.

    Args:
        origin_path (str): the directory of original network's dump files.
        target_path (str): the directory of target network's dump files.
        output_path (Optional[str], optional): the directory to save the compare result. Defaults to None.
        ignore_backward (bool, optional): whether to ignore the backward npy files. Defaults to False.
        ignore_unmatched (bool, optional): whether to ignore the unmatched npy files. Defaults to False.
    """
    ignore_backward = kwargs.get('ignore_backward', False)
    convinced_match_method = kwargs.get('convinced_match_method', 'recursion')

    origin_ret = get_dump_path(origin_path)
    if origin_ret is None:
        raise ValueError("origin_path is not a valid dump path")
    origin_npy_path, origin_pkl_path, origin_framework = origin_ret
    target_ret = get_dump_path(target_path)
    if target_ret is None:
        raise ValueError("target_path is not a valid dump path")
    target_npy_path, target_pkl_path, target_framework = target_ret
    field_names = [
        f"ORIGIN NET ({origin_framework})",
        f"TARGET NET ({target_framework})",
    ]
    diff_field_names = ["result of allclose", "ratio of allclose",
                        "cosine similarity", "mean & max diffs"]
    if output_path is None:
        save_map_path = None
        save_forward_path = None
        save_backward_path = None
    else:
        os.makedirs(output_path, mode=0o700, exist_ok=True)
        save_map_path = os.path.join(output_path, 'ts_api_mapping.csv')
        save_forward_path = os.path.join(output_path, 'ts_api_forward_compare.csv')
        save_backward_path = os.path.join(output_path, 'ts_api_backward_compare.csv')
    origin_pkl_list = APIList(origin_framework)
    target_pkl_list = APIList(target_framework)
    origin_pkl_list.construct(origin_pkl_path)
    target_pkl_list.construct(target_pkl_path)
    apis_map = flow_match(
        origin_pkl_list,
        target_pkl_list,
        err_threshold=1.0,
        ignore_shape=False,
        convinced_match_method=convinced_match_method
    )
    _print_apis_map_result(apis_map, output_file=save_map_path, field_names=field_names)
    npy_forward_list, npy_backward_list = get_npy_map_list(
        apis_map,
        origin_npy_path,
        target_npy_path,
        origin_pkl_path,
        target_pkl_path,
        ignore_backward=ignore_backward,
        ignore_unmatched=ignore_unmatched,
    )
    compare_npy_dir(
        origin_npy_path,
        target_npy_path,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        name_map_list=npy_forward_list,
        compare_shape=True,
        output_file=save_forward_path,
        title='The forward comparison results',
        field_names=field_names + diff_field_names
    )
    if not ignore_backward:
        npy_backward_list.reverse()
        compare_npy_dir(
            origin_npy_path,
            target_npy_path,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            name_map_list=npy_backward_list,
            compare_shape=True,
            output_file=save_backward_path,
            title='The backward comparison results',
            field_names=field_names + diff_field_names
        )
