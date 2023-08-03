import glob
import os

from typing import List, Optional
import numpy as np
from troubleshooter.migrator.diff_handler import compare_npy_dir, min_edit_distance
from .apis_match import (
    APIList,
    flow_match,
    _print_apis_map_result,
)

__all__ = ['api_dump_compare']


def get_npy_map_list(
    apis_map: List,
    orig_npy_dir: str,
    target_npy_dir: str,
    ignore_backward: bool = False,
    ignore_unmatched: bool = False,
):
    """covert apis_map to npy_map_list

    Args:
        apis_map (List): 通过toolkit中的api_match生成的api映射表
        orig_npy_dir (str): 原网络dump出的npy文件目录
        target_npy_dir (str): 目标网络dump出的npy文件目录
        ignore_backward (bool, optional): 是否忽略反向npy的比对. Defaults to False.

    Returns:
        List: npy文件映射表
    """

    def _get_npy_list(apis, io, npy_dir):
        npy_list = [
            os.path.basename(i)
            for i in glob.glob(os.path.join(npy_dir, f"{apis}_*{io}*"))
        ]
        forward_npy_list = []
        backward_npy_list = []
        for i in npy_list:
            if "forward" in i:
                forward_npy_list.append(i)
            elif "backward" in i:
                backward_npy_list.append(i)
        forward_npy_list = sorted(forward_npy_list)
        backward_npy_list = sorted(backward_npy_list)
        return forward_npy_list, backward_npy_list

    def _get_name_map_list(orig_name_list, target_name_list):
        orig_shape_list = [
            np.load(os.path.join(orig_npy_dir, orig_name)).shape
            for orig_name in orig_name_list
        ]
        target_shape_list = [
            np.load(os.path.join(target_npy_dir, target_name)).shape
            for target_name in target_name_list
        ]
        _, index_map_list = min_edit_distance(orig_shape_list, target_shape_list)

        name_map_list = [
            (
                orig_name_list[a] if a is not None else None,
                target_name_list[b] if b is not None else None,
            )
            for a, b in index_map_list
        ]
        return name_map_list

    def _get_npy_map(orig_npy_list, target_npy_list, ignore_unmatched):
        if len(orig_npy_list) == 0:
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
                return [(i, None) for i in orig_npy_list]
            else:
                ret = _get_name_map_list(orig_npy_list, target_npy_list)
                if ignore_unmatched:
                    ret = [(i, j) for i, j in ret if i is not None and j is not None]
                return ret

    forward_list = []
    backward_list = []
    for orig_apis, target_apis in apis_map:
        if len(orig_apis) != 0:
            orig_input_forward, orig_input_backward = _get_npy_list(
                orig_apis[0], 'input', orig_npy_dir
            )
            orig_output_forward, orig_output_backward = _get_npy_list(
                orig_apis[-1], 'output', orig_npy_dir
            )
        else:
            orig_input_forward = []
            orig_input_backward = []
            orig_output_forward = []
            orig_output_backward = []
        if len(target_apis) != 0:
            target_input_forward, target_input_backward = _get_npy_list(
                target_apis[0], 'input', target_npy_dir
            )
            target_output_forward, target_output_backward = _get_npy_list(
                target_apis[-1], 'output', target_npy_dir
            )
        else:
            target_input_forward = []
            target_input_backward = []
            target_output_forward = []
            target_output_backward = []

        input_forward_map = _get_npy_map(
            orig_input_forward, target_input_forward, ignore_unmatched
        )
        forward_list += input_forward_map
        if not ignore_backward:
            input_backward_map = _get_npy_map(
                orig_input_backward, target_input_backward, ignore_unmatched
            )
            backward_list += input_backward_map

        if not ignore_unmatched:
            if orig_apis[1:-1]:
                orig_inter_forward, orig_inter_backward = zip(
                    *[_get_npy_list(i, '', orig_npy_dir) for i in orig_apis[1:-1]]
                )
                orig_inter_forward = [(i, None) for j in orig_inter_forward for i in j]
                orig_inter_backward = [
                    (i, None) for j in orig_inter_backward for i in j
                ]
                forward_list += orig_inter_forward
                if not ignore_backward:
                    backward_list += orig_inter_backward
            if target_apis[1:-1]:
                target_inter_forward, target_inter_backward = zip(
                    *[_get_npy_list(i, '', target_npy_dir) for i in target_apis[1:-1]]
                )
                target_inter_forward = [
                    (None, i) for j in target_inter_forward for i in j
                ]
                target_inter_backward = [
                    (None, i) for j in target_inter_backward for i in j
                ]
                forward_list += target_inter_forward
                if not ignore_backward:
                    backward_list += target_inter_backward

        output_forward_map = _get_npy_map(
            orig_output_forward, target_output_forward, ignore_unmatched
        )
        forward_list += output_forward_map
        if not ignore_backward:
            output_backward_map = _get_npy_map(
                orig_output_backward, target_output_backward, ignore_unmatched
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


def api_dump_compare(
    orign_path: str, target_path: str, output_path: Optional[str] = None, **kwargs
):
    """compare each api's output by the npy files dumped by the network.

    Args:
        orign_path (str): the directory of original network's dump files.
        target_path (str): the directory of target network's dump files.
        output_path (Optional[str], optional): the directory to save the compare result. Defaults to None.
        ignore_backward (bool, optional): whether to ignore the backward npy files. Defaults to False.
        ignore_unmatched (bool, optional): whether to ignore the unmatched npy files. Defaults to False.
    """
    ignore_backward = kwargs.get('ignore_backward', False)
    ignore_unmatched = kwargs.get('ignore_unmatched', False)

    orign_ret = get_dump_path(orign_path)
    if orign_ret is None:
        raise ValueError("orign_path is not a valid dump path")
    orign_npy_path, orign_pkl_path, orign_framework = orign_ret
    target_ret = get_dump_path(target_path)
    if target_ret is None:
        raise ValueError("target_path is not a valid dump path")
    target_npy_path, target_pkl_path, target_framework = target_ret
    if output_path is None:
        save_map_path = None
        save_forward_path = None
        save_backward_path = None
        field_names = None
    else:
        os.makedirs(output_path, exist_ok=True)
        save_map_path = os.path.join(output_path, 'ts_api_mapping.csv')
        save_forward_path = os.path.join(output_path, 'ts_api_forward_compare.csv')
        save_backward_path = os.path.join(output_path, 'ts_api_backward_compare.csv')
        field_names = [
            f"ORIG NET ({orign_framework})",
            f"TARGET NET ({target_framework})",
        ]
    orig_pkl_list = APIList(orign_framework)
    target_pkl_list = APIList(target_framework)
    orig_pkl_list.Construct(orign_pkl_path)
    target_pkl_list.Construct(target_pkl_path)
    apis_map = flow_match(orig_pkl_list, target_pkl_list, 1.0)
    _print_apis_map_result(apis_map, output_file=save_map_path, field_names=field_names)
    npy_forward_list, npy_backward_list = get_npy_map_list(
        apis_map,
        orign_npy_path,
        target_npy_path,
        ignore_backward=ignore_backward,
        ignore_unmatched=ignore_unmatched,
    )
    compare_npy_dir(
        orign_npy_path,
        target_npy_path,
        name_map_list=npy_forward_list,
        compare_shape=True,
        output_file=save_forward_path,
    )
    if not ignore_backward:
        compare_npy_dir(
            orign_npy_path,
            target_npy_path,
            name_map_list=npy_backward_list,
            compare_shape=True,
            output_file=save_backward_path,
        )
