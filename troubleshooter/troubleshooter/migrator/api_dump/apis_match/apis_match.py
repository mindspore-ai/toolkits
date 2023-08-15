import functools
import json
import os
import re
from collections import OrderedDict
from typing import Any, List, Optional

import numpy as np
from prettytable import ALL, PrettyTable

from .api_io_dict import pt_io_dict
from .api_name_dict import pt_name_dict
from .download_api_map import get_pt_api_dict

__all__ = ['flow_match', 'APIList', '_print_apis_map_result', 'load_pkl']


def _print_apis_map_result(
    result_list, title=None, print_level=1, output_file=None, **kwargs
):
    # 0 Do not print
    # Print All
    # print False
    if print_level == 0:
        return
    if not result_list:
        return
    field_names = kwargs.get('field_names', ["ORIGIN NET", "TARGET NET"])
    x = PrettyTable(hrules=ALL)
    if title is None:
        x.title = 'The APIs mapping results of the two frameworks'
    else:
        x.title = title

    x.field_names = field_names
    for _origin, _target in result_list:
        origin = [str(name) for name in _origin]
        target = [str(name) for name in _target]
        origin_str = "\n".join(origin) if origin else ""
        target_str = "\n".join(target) if target else ""
        x.add_row([origin_str, target_str])
    print(x.get_string())
    if output_file:
        if not os.path.exists(os.path.dirname(output_file)):
            raise ValueError(f"output_file {output_file} not exist")
        with open(output_file, "w") as f:
            f.write(x.get_csv_string() + os.linesep)


def load_pkl(path: str):
    ret = []
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if line.strip() == b"":
                break
            ret.append(json.loads(line))
    return ret


class APIDataNode:
    def __init__(self, shape, dtype, summary) -> None:
        # 此处大小需要作为key，需要转换为tuple
        self.shape = tuple(shape)
        self.dtype = dtype
        self.summary = np.array(summary)


class APINode:
    """单个API的对象，包含该api的基本信息，如名字、形状等。
    注意该对象名字、形状相同则认为相等，所以一个list里面可能有多个相等的对象但id并不相同。
    """

    api_dump_num = 0

    def __init__(
        self,
        framework: str,
        dump_type: str,
        api_name: str,
        api_id: int,
        index: Optional[int] = None,
    ):
        self.dump_type = dump_type
        self.api_name = api_name
        self.api_id = api_id
        self.uni_name = get_uni_name(framework, dump_type, api_name)
        self.dump_id = APINode.api_dump_num
        APINode.api_dump_num += 1

        self.forward_input_data = OrderedDict()
        self.forward_output_data = OrderedDict()

        self.l_index = index

    # def __hash__(self) -> int:
    #     return hash((self.uni_name, self.input_shape, self.output_shape))
    # NOTE 不要用list的index查找索引
    def __eq__(self, __value: object) -> bool:
        def _eq_shape(a, b):
            if len(a) != len(b):
                return False
            for i, j in zip(a.values(), b.values()):
                if i.shape != j.shape:
                    return False
            return True

        return (
            self.uni_name == __value.uni_name
            and _eq_shape(self.forward_input_data, __value.forward_input_data)
            and _eq_shape(self.forward_output_data, __value.forward_output_data)
        )

    def __str__(self):
        return f"{self.dump_type}_{self.api_name}_{self.api_id}"

    def __repr__(self) -> str:
        return f"{self.dump_type}_{self.api_name}_{self.api_id}"

    def update_api(
        self, dump_type, api_name, api_id, direction, data_io, data_id, api_data
    ) -> bool:
        assert direction in [
            "forward",
            "backward",
        ], "direction should be forward or backward."
        assert data_io in ["input", "output"], "data_io should be input or output."
        if (
            dump_type == self.dump_type
            and api_name == self.api_name
            and api_id == self.api_id
        ):
            if direction == "forward":
                if data_io == "input":
                    self.forward_input_data[data_id] = api_data
                else:
                    self.forward_output_data[data_id] = api_data
            else:
                raise NotImplementedError("backward not implemented.")
        else:
            return False
        return True


class APIList:
    """用于根据pkl文件构造一个api数据流。"""

    def __init__(self, framework: str):
        self.api_list = []
        self.framework = framework

    def Construct(self, pkl_path: str) -> Any:
        pkl = load_pkl(pkl_path)
        for l in pkl:
            ret = self._read_line(l)
            if not ret:
                break
        GetUniIO(self.api_list, self.framework)

    def _read_line(self, line):
        prefix, dump_step, _, data_type, data_shape, data_summary = line
        api_data = APIDataNode(data_shape, data_type, data_summary)

        def _read_prefix(prefix):
            pattern = re.compile(
                r"^(\w+?)_(\w+)_(\d+)+_(\w+)_([io][nu]t?put)\.?(\d+)?\.?(\d+)?$"
            )
            prefix_con = re.findall(pattern, prefix)
            if len(prefix_con) == 0:
                print(f"ignore {prefix}")
                return None
            prefix_con = prefix_con[0]
            if prefix_con[5] == "":
                data_id = '0'
            elif prefix_con[6] == "":
                data_id = prefix_con[5]
            else:
                data_id = prefix_con[5] + '.' + prefix_con[6]

            return (
                prefix_con[0],
                prefix_con[1],
                int(prefix_con[2]),
                prefix_con[3],
                prefix_con[4],
                data_id,
            )

        prefix_con = _read_prefix(prefix)
        if prefix_con:
            dump_type, api_name, api_id, direction, data_io, data_id = prefix_con

            if direction == "forward":
                if len(self.api_list) == 0 or not self.api_list[-1].update_api(
                    dump_type, api_name, api_id, direction, data_io, data_id, api_data
                ):
                    self.api_list.append(
                        APINode(
                            self.framework,
                            dump_type,
                            api_name,
                            api_id,
                            index=len(self.api_list),
                        )
                    )
                    self.api_list[-1].update_api(
                        dump_type,
                        api_name,
                        api_id,
                        direction,
                        data_io,
                        data_id,
                        api_data,
                    )

            # 忽略backward及其后面的内容
            elif direction == "backward":
                return False
        return True

    def __iter__(self):
        return iter(self.api_list)

    def __next__(self):
        return next(self.api_list)

    def __getitem__(self, index):
        return self.api_list[index]

    def __len__(self):
        return len(self.api_list)


class APIInterNode:
    """API数据流之间的中间对象"""

    def __init__(
        self,
        output_api_name,
        input_api_name,
        forward_output_data,
        forward_input_data,
        index=[],
    ):
        self.output_api_name = output_api_name
        self.input_api_name = input_api_name

        self.forward_output_data = forward_output_data
        self.forward_input_data = forward_input_data

        self.l_index = index

    def __eq__(self, __value: object) -> bool:
        def _eq_shape(a, b):
            if len(a) != len(b):
                return False
            for i, j in zip(a.values(), b.values()):
                if i.shape != j.shape:
                    return False
            return True

        return _eq_shape(
            self.forward_input_data, __value.forward_input_data
        ) and _eq_shape(self.forward_output_data, __value.forward_output_data)


def GetUniIO(api_list: List, framework) -> Any:
    if framework == 'mindspore':
        return
    elif framework == 'pytorch':
        api_io_dict = pt_io_dict
    else:
        raise NotImplementedError(f'Not support {framework} now.')
    for api in api_list:
        if (api.dump_type, api.api_name) in api_io_dict:
            if len(api_io_dict[(api.dump_type, api.api_name)][0]) != 0:
                forward_input_data = OrderedDict()
                for k, v in api_io_dict[(api.dump_type, api.api_name)][0].items():
                    forward_input_data[k] = api.forward_input_data[v]
                api.forward_input_data = forward_input_data
            if len(api_io_dict[(api.dump_type, api.api_name)][1]) != 0:
                forward_output_data = OrderedDict()
                for k, v in api_io_dict[(api.dump_type, api.api_name)][1].items():
                    forward_output_data[k] = api.forward_output_data[v]
                api.forward_output_data = forward_output_data


@functools.lru_cache()
def get_uni_name_map():
    # 在线获取pytorch字典
    def _get_pt_api_dict():
        apis_dict = get_pt_api_dict()
        if apis_dict is None:
            print('load local pytorch name map.')
            return pt_name_dict
        ret = {}
        for k, v in apis_dict.items():
            pt_api = k.split(".")[-2:]
            ms_api = v.split(".")[-2:]

            if ms_api[0] not in ["nn", "ops", "Tensor"]:
                continue
            if pt_api[0] not in ["torch", "functional", "Tensor", "Module", "nn"]:
                continue

            pt_api_name, ms_api_name = pt_api[-1], ms_api[-1]
            pt_api_type, ms_api_type = pt_api[0].lower(), ms_api[0].lower()
            ms_api_type = 'functional' if ms_api_type == 'ops' else ms_api_type
            ret.update({(pt_api_type, pt_api_name): (ms_api_type, ms_api_name)})
        return ret

    return {"pytorch": pt_name_dict, "mindspore": {}}


def get_uni_name(framework: str, dump_type: str, name: str) -> str:
    """获取统一的算子名称

    Args:
        framework (str): 框架
        dump_type (str): 算子类型
        name (str): 算子名称

    Returns:
        str: 统一的算子名称
    """
    if framework not in [
        "pytorch",
        "mindspore",
    ]:
        raise NotImplementedError(f"not support {framework} now.")
    uni_name_map = get_uni_name_map()
    if (dump_type.lower(), name) in uni_name_map[framework]:
        return '_'.join(uni_name_map[framework][(dump_type.lower(), name)])
    return f'{dump_type.lower()}_{name}'


class APIsInterMatch:
    """核心匹配算法"""

    def __call__(self, orig_list: List, target_list: List, err_threshold: float = 1.0):
        orig_inter, target_inter = self._get_vertex(orig_list, target_list)
        map_inter = self._inter_match(
            orig_inter, target_inter, err_threshold=err_threshold
        )
        ret = self._get_map(orig_list, target_list, map_inter)
        if len(ret[0][0]) == 0 and len(ret[0][1]) == 0:
            ret = ret[1:]
        if len(ret[-1][0]) == 0 and len(ret[-1][1]) == 0:
            ret = ret[:-1]
        return ret

    def _get_map(self, origin_list, target_list, map_inter):
        ret = []
        origin_ptr = 0
        target_ptr = 0
        for origin_idx, target_idx in map_inter:
            ret.append(
                (origin_list[origin_ptr:origin_idx], target_list[target_ptr:target_idx])
            )
            origin_ptr, target_ptr = origin_idx, target_idx
        ret.append((origin_list[origin_ptr:], target_list[target_ptr:]))
        return ret

    def _get_vertex(self, origin_list, target_list):
        origin_inter = [
            APIInterNode(
                "",
                origin_list[0].api_name,
                OrderedDict(),
                origin_list[0].forward_input_data,
                index=0,
            )
        ]
        for i in range(len(origin_list) - 1):
            origin_inter.append(
                APIInterNode(
                    origin_list[i].api_name,
                    origin_list[i + 1].api_name,
                    origin_list[i].forward_output_data,
                    origin_list[i + 1].forward_input_data,
                    index=i + 1,
                )
            )
        origin_inter.append(
            APIInterNode(
                origin_list[-1].api_name,
                "",
                origin_list[-1].forward_output_data,
                OrderedDict(),
                index=len(origin_list),
            )
        )
        target_inter = [
            APIInterNode(
                "",
                target_list[0].api_name,
                OrderedDict(),
                target_list[0].forward_input_data,
                index=0,
            )
        ]
        for i in range(len(target_list) - 1):
            target_inter.append(
                APIInterNode(
                    target_list[i].api_name,
                    target_list[i + 1].api_name,
                    target_list[i].forward_output_data,
                    target_list[i + 1].forward_input_data,
                    index=i + 1,
                )
            )
        target_inter.append(
            APIInterNode(
                target_list[-1].api_name,
                "",
                target_list[-1].forward_output_data,
                OrderedDict(),
                index=len(target_list),
            )
        )
        return origin_inter, target_inter

    def _api_summary_sim(self, origin_inter, target_inter, err_threshold):
        input_summary_diff = np.abs(
            np.array(
                [
                    i.summary - j.summary
                    for i, j in zip(
                        origin_inter.forward_input_data.values(),
                        target_inter.forward_input_data.values(),
                    )
                ]
            )
        )
        input_summary_sum = np.array(
            [
                np.abs(i.summary) + np.abs(j.summary)
                for i, j in zip(
                    origin_inter.forward_input_data.values(),
                    target_inter.forward_input_data.values(),
                )
            ]
        )
        output_summary_diff = np.abs(
            np.array(
                [
                    i.summary - j.summary
                    for i, j in zip(
                        origin_inter.forward_output_data.values(),
                        target_inter.forward_output_data.values(),
                    )
                ]
            )
        )
        output_summary_sum = np.array(
            [
                np.abs(i.summary) + np.abs(j.summary)
                for i, j in zip(
                    origin_inter.forward_output_data.values(),
                    target_inter.forward_output_data.values(),
                )
            ]
        )
        if (
            np.any(input_summary_diff / (input_summary_sum + 1e-6)) <= err_threshold
            and np.any(output_summary_diff / (output_summary_sum + 1e-6))
            <= err_threshold
        ):
            dist = (
                np.arctan(
                    np.linalg.norm(input_summary_diff)
                    + np.linalg.norm(output_summary_diff)
                )
                * 2
                / np.pi
            )
        else:
            dist = 1
        return dist

    def _api_name_sim(self, origin_inter, target_inter):
        def _min_distance(word1: str, word2: str) -> int:
            @functools.lru_cache(None)
            def helper(i, j):
                if i == len(word1) or j == len(word2):
                    return len(word1) - i + len(word2) - j
                if word1[i] == word2[j]:
                    return helper(i + 1, j + 1)
                else:
                    inserted = helper(i, j + 1)
                    deleted = helper(i + 1, j)
                    replaced = helper(i + 1, j + 1)
                    return min(inserted, deleted, replaced) + 1

            return helper(0, 0)

        input_dist = (
            float(
                _min_distance(origin_inter.input_api_name, target_inter.input_api_name)
            )
            / max(len(origin_inter.input_api_name), len(target_inter.input_api_name))
            if len(origin_inter.input_api_name) != 0
            or len(target_inter.input_api_name) != 0
            else 1.0
        )
        output_dist = (
            float(
                _min_distance(
                    origin_inter.output_api_name, target_inter.output_api_name
                )
            )
            / max(len(origin_inter.output_api_name), len(target_inter.output_api_name))
            if len(origin_inter.output_api_name) != 0
            or len(target_inter.output_api_name) != 0
            else 1.0
        )
        # input_dist = 1. if input_dist > 0.5 else input_dist
        # output_dist = 1. if output_dist > 0.5 else output_dist
        return (input_dist + output_dist) / 2

    def _inter_match(self, origin_inter, target_inter, err_threshold):
        dp = [
            [0 for _ in range(len(target_inter) + 1)]
            for _ in range(len(origin_inter) + 1)
        ]
        path = [
            [0 for _ in range(len(target_inter) + 1)]
            for _ in range(len(origin_inter) + 1)
        ]

        dp[0] = [i for i in range(len(target_inter) + 1)]
        for i in range(len(origin_inter) + 1):
            dp[i][0] = i

        for i in range(1, len(origin_inter) + 1):
            for j in range(1, len(target_inter) + 1):
                dist = 1
                if origin_inter[i - 1] == target_inter[j - 1]:
                    if err_threshold < 1:
                        dist = self._api_summary_sim(
                            origin_inter[i - 1], target_inter[j - 1], err_threshold
                        )
                    else:
                        dist = 0

                    if dist < 1:
                        name_dist = self._api_name_sim(
                            origin_inter[i - 1], target_inter[j - 1]
                        )
                        dist = (dist + name_dist) / 20

                match_dist = (dp[i - 1][j - 1] + dist, 1 if dist < 1 else 2)
                unmatch_i_dist = (dp[i - 1][j] + 1, 3)
                unmatch_j_dist = (dp[i][j - 1] + 1, 4)
                path_min = min(
                    match_dist, unmatch_i_dist, unmatch_j_dist, key=lambda x: x[0]
                )
                dp[i][j] = path_min[0]
                path[i][j] = path_min[1]

        ret = []
        i, j = len(origin_inter), len(target_inter)
        while i > 0 and j > 0:
            if path[i][j] == 1:
                ret.append(
                    (
                        origin_inter[i - 1].l_index,
                        target_inter[j - 1].l_index,
                    )
                )
                i -= 1
                j -= 1
            elif path[i][j] == 2:
                i -= 1
                j -= 1
            elif path[i][j] == 3:
                i -= 1
            elif path[i][j] == 4:
                j -= 1
        return ret[::-1]


apis_match = APIsInterMatch()


class FlowMatch:
    """用户匹配接口"""

    def __call__(
        self,
        origin_list: APIList,
        target_list: APIList,
        err_threshold: float = 1.0,
        ignore_shape=False,
    ):
        assert (
            err_threshold >= 0 and err_threshold <= 1
        ), "err_threshold must be in [0., 1.]"

        match = self._convinced_match(origin_list, target_list, ignore_shape)
        unmatch = self._get_unmatch_api(origin_list, target_list, match)
        unmatch_split = []
        for i in unmatch:
            if len(i[0]) == 0 or len(i[1]) == 0:
                unmatch_split.append(i)
                continue
            unmatch_split.append(apis_match(*i, err_threshold=err_threshold))
        match = [([i[0]], [i[1]]) for i in match]
        if (
            isinstance(unmatch_split[0], tuple)
            and len(unmatch_split[0][0]) == 0
            and len(unmatch_split[0][1]) == 0
        ):
            ret = []
        else:
            ret = (
                [unmatch_split[0]]
                if isinstance(unmatch_split[0], tuple)
                else unmatch_split[0][:]
            )

        for m, u in zip(match, unmatch_split[1:]):
            ret.append(m)
            if isinstance(u, tuple):
                if len(u[0]) != 0 or len(u[1]) != 0:
                    ret.append(u)
            else:
                ret += u
        return ret

    def _convinced_match(self, origin_list, target_list, ignore_shape=False):
        def _get_api_attr(api):
            if not ignore_shape:
                forward_input_shape = tuple(
                    [i.shape for i in api.forward_input_data.values()]
                )
                forward_output_shape = tuple(
                    [i.shape for i in api.forward_output_data.values()]
                )
                return (
                    api.uni_name,
                    forward_input_shape,
                    forward_output_shape,
                )
            else:
                return api.uni_name

        api_count = {}
        api_match = []
        for api in origin_list:
            if api_count.get(_get_api_attr(api)):
                api_count[_get_api_attr(api)] += 1
            else:
                api_count[_get_api_attr(api)] = 1
        for api in target_list:
            if api_count.get(_get_api_attr(api)):
                api_count[_get_api_attr(api)] -= 1
        for k, v in api_count.items():
            if v == 0:
                origin_api = [api for api in origin_list if _get_api_attr(api) == k]
                target_api = [api for api in target_list if _get_api_attr(api) == k]
                api_match += list(zip(origin_api, target_api))
        api_match = sorted(api_match, key=lambda x: x[0].l_index)
        return api_match

    def _get_unmatch_api(self, origin_list, target_list, conv_map):
        last_origin = 0
        last_target = 0
        unmatch_origin = []
        unmatch_target = []
        for origin_api, target_api in conv_map:
            p_origin = origin_api.l_index
            unmatch_origin.append(origin_list[last_origin:p_origin])
            last_origin = p_origin + 1

            p_target = target_api.l_index
            unmatch_target.append(target_list[last_target:p_target])
            last_target = p_target + 1

        unmatch_origin.append(origin_list[last_origin:])
        unmatch_target.append(target_list[last_target:])

        return list(zip(unmatch_origin, unmatch_target))


flow_match = FlowMatch()
