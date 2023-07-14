import functools
import json
import re
from typing import Any, List, Optional

import numpy as np
from prettytable import ALL as ALL
from prettytable import PrettyTable

from troubleshooter.toolbox.apis_match.api_io_dict import pt_io_dict
from troubleshooter.toolbox.apis_match.download_api_map import get_pt_api_dict


__all__ = ['FlowMatch', 'APIList', 'print_apis_map_result']


def print_apis_map_result(result_list, title=None, print_level=1, **kwargs):
    # 0 Do not print
    # Print All
    # print False
    if print_level == 0:
        return
    if not result_list:
        return
    field_names = kwargs.get('field_names', ["ORIG NET", "TARGET NET"])
    x = PrettyTable(hrules=ALL)
    if title is None:
        x.title = 'The APIs mapping results of the two frameworks'
    else:
        x.title = title

    x.field_names = field_names
    for _orig, _target in result_list:
        orig = [str(i) for i in _orig]
        target = [str(i) for i in _target]
        orig_str = "\n".join(orig) if orig else ""
        target_str = "\n".join(target) if target else ""
        x.add_row([orig_str, target_str])
    print(x.get_string())


def load_pkl(path: str):
    ret = []
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if line.strip() == b"":
                break
            ret.append(json.loads(line))
    return ret


class GetUniIO:
    def __init__(self, framework: str = 'pytorch') -> None:
        # 前面是输入映射，后面是输出映射
        self.framework = framework
        if framework == 'pytorch':
            self.api_io_dict = pt_io_dict
        else:
            raise NotImplementedError(f'Not support {framework} now.')

    def __call__(self, api_list: List) -> Any:
        for api in api_list:
            if (api.dump_type, api.api_name) in self.api_io_dict:
                if len(self.api_io_dict[(api.dump_type, api.api_name)][0]) != 0:
                    forward_input_shape = []
                    forward_input_type = []
                    forward_input_summery = []
                    for i in self.api_io_dict[(api.dump_type, api.api_name)][0]:
                        forward_input_shape.append(api.forward_input_shape[i])
                        forward_input_type.append(api.forward_input_type[i])
                        forward_input_summery.append(api.forward_input_summery[i])
                    api.forward_input_shape = forward_input_shape
                    api.forward_input_type = forward_input_type
                    api.forward_input_summery = forward_input_summery
                if len(self.api_io_dict[(api.dump_type, api.api_name)][1]) != 0:
                    forward_output_shape = []
                    forward_output_type = []
                    forward_output_summery = []
                    for i in self.api_io_dict[(api.dump_type, api.api_name)][1]:
                        forward_output_shape.append(api.forward_output_shape[i])
                        forward_output_type.append(api.forward_output_type[i])
                        forward_output_summery.append(api.forward_output_summery[i])

                    api.forward_output_shape = forward_output_shape
                    api.forward_output_type = forward_output_type
                    api.forward_output_summery = forward_output_summery


class GetMSUniName(object):
    def __init__(self) -> None:
        _pt_dict = self._get_pt_api_dict()
        self._uni_name_map = {"pytorch": _pt_dict, "mindspore": {}}

    def _get_pt_api_dict(self):
        apis_dict = get_pt_api_dict()
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
            ret.update({(pt_api_type, pt_api_name): f"{ms_api_type}_{ms_api_name}"})
        return ret

    def __call__(self, framework: str, dump_type: str, name: str):
        assert framework in [
            "pytorch",
            "mindspore",
        ], "framework should be pytorch or mindspore."
        if (dump_type.lower(), name) in self._uni_name_map[framework]:
            return self._uni_name_map[framework][(dump_type.lower(), name)]
        return f'{dump_type.lower()}_{name}'


class APIDump:
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
        self.uni_name = GetMSUniName()(framework, dump_type, api_name)
        self.dump_id = APIDump.api_dump_num
        APIDump.api_dump_num += 1

        # 此处大小需要作为key，需要转换为tuple
        self.forward_input_shape = []
        self.forward_output_shape = []
        self.forward_input_type = []
        self.forward_output_type = []
        self.forward_input_summery = []
        self.forward_output_summery = []

        self.dump_list_index = index

    # def __hash__(self) -> int:
    #     return hash((self.uni_name, self.input_shape, self.output_shape))
    # NOTE 不要用list的index查找索引
    def __eq__(self, __value: object) -> bool:
        def _eq_list(a, b):
            if len(a) != len(b):
                return False
            for i, j in zip(a, b):
                if i != j:
                    return False
            return True

        return (
            self.uni_name == __value.uni_name
            and _eq_list(self.forward_input_shape, __value.forward_input_shape)
            and _eq_list(self.forward_output_shape, __value.forward_output_shape)
        )

    def __str__(self):
        return f"{self.dump_type}_{self.api_name}_{self.api_id}"

    def __repr__(self) -> str:
        return f"{self.dump_type}_{self.api_name}_{self.api_id}"

    def update_api(
        self,
        dump_type,
        api_name,
        api_id,
        direction,
        data_io,
        data_id,
        data_type,
        data_shape,
        data_summery,
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
                    if len(self.forward_input_type) <= data_id:
                        self.forward_input_type.append(data_type)
                        self.forward_input_shape.append(tuple(data_shape))
                        self.forward_input_summery.append(np.array(data_summery))
                    else:
                        self.forward_input_type[data_id] = data_type
                        self.forward_input_shape[data_id] = tuple(data_shape)
                        self.forward_input_summery[data_id] = np.array(data_summery)
                else:
                    if len(self.forward_output_type) <= data_id:
                        self.forward_output_type.append(data_type)
                        self.forward_output_shape.append(tuple(data_shape))
                        self.forward_output_summery.append(np.array(data_summery))
                    else:
                        self.forward_output_type[data_id] = data_type
                        self.forward_output_shape[data_id] = tuple(data_shape)
                        self.forward_output_summery[data_id] = np.array(data_summery)
            else:
                raise NotImplementedError("backward not implemented.")
        else:
            return False
        return True


class APIList:
    def __init__(self, framework: str):
        self.api_list = []
        self.framework = framework

    def Construct(self, pkl_path: str) -> Any:
        pkl = load_pkl(pkl_path)
        for l in pkl:
            ret = self._read_line(l)
            if not ret:
                break

    def _read_line(self, line):
        prefix, dump_step, _, data_type, data_shape, data_summery = line

        def _read_prefix(prefix):
            pattern = re.compile(
                r"^(\w+?)_(\w+)_(\d+)+_(\w+)_([io][nu]t?put)\.?(\d+)?$"
            )
            prefix_con = re.findall(pattern, prefix)
            if len(prefix_con) == 0:
                print(f"ignore {prefix}")
                return None
            prefix_con = prefix_con[0]
            if prefix_con[5] == "":
                data_id = 0
            else:
                data_id = prefix_con[5]

            return (
                prefix_con[0],
                prefix_con[1],
                int(prefix_con[2]),
                prefix_con[3],
                prefix_con[4],
                int(data_id),
            )

        prefix_con = _read_prefix(prefix)
        if prefix_con:
            dump_type, api_name, api_id, direction, data_io, data_id = prefix_con

            if direction == "forward":
                if len(self.api_list) == 0 or not self.api_list[-1].update_api(
                    dump_type,
                    api_name,
                    api_id,
                    direction,
                    data_io,
                    data_id,
                    data_type,
                    data_shape,
                    data_summery,
                ):
                    self.api_list.append(
                        APIDump(self.framework, dump_type, api_name, api_id)
                    )
                    self.api_list[-1].update_api(
                        dump_type,
                        api_name,
                        api_id,
                        direction,
                        data_io,
                        data_id,
                        data_type,
                        data_shape,
                        data_summery,
                    )

            # 忽略backward及其后面的内容
            elif direction == "backward":
                return False
        return True


class APIInter:
    def __init__(
        self,
        output_api_name,
        input_api_name,
        forward_output_shape,
        forward_output_type,
        forward_output_summery,
        forward_input_shape,
        forward_input_type,
        forward_input_summery,
        index=[],
    ):
        self.output_api_name = output_api_name
        self.input_api_name = input_api_name
        self.forward_input_shape = forward_input_shape
        self.forward_output_shape = forward_output_shape
        self.forward_input_type = forward_input_type
        self.forward_output_type = forward_output_type
        self.forward_input_summery = forward_input_summery
        self.forward_output_summery = forward_output_summery

        self.inter_list_index = index

    def __repr__(self) -> str:
        return f"{self.forward_output_shape}_{self.forward_input_shape}"

    def __eq__(self, __value: object) -> bool:
        def _eq_list(a, b):
            if len(a) != len(b):
                return False
            for i, j in zip(a, b):
                if i != j:
                    return False
            return True

        return _eq_list(
            self.forward_input_shape, __value.forward_input_shape
        ) and _eq_list(self.forward_output_shape, __value.forward_output_shape)


class APIsInterMatch:
    def __init__(self) -> None:
        pass

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

    def _get_map(self, orig_list, target_list, map_inter):
        ret = []
        orig_ptr = 0
        target_ptr = 0
        for orig_idx, target_idx in map_inter:
            ret.append(
                (orig_list[orig_ptr:orig_idx], target_list[target_ptr:target_idx])
            )
            orig_ptr, target_ptr = orig_idx, target_idx
        ret.append((orig_list[orig_ptr:], target_list[target_ptr:]))
        return ret

    def _get_vertex(self, orig_list, target_list):
        orig_inter = [
            APIInter(
                "",
                orig_list[0].api_name,
                [],
                [],
                [],
                orig_list[0].forward_input_shape,
                orig_list[0].forward_input_type,
                orig_list[0].forward_input_summery,
                index=0,
            )
        ]
        for i in range(len(orig_list) - 1):
            orig_inter.append(
                APIInter(
                    orig_list[i].api_name,
                    orig_list[i + 1].api_name,
                    orig_list[i].forward_output_shape,
                    orig_list[i].forward_output_type,
                    orig_list[i].forward_output_summery,
                    orig_list[i + 1].forward_input_shape,
                    orig_list[i + 1].forward_input_type,
                    orig_list[i + 1].forward_input_summery,
                    index=i + 1,
                )
            )
        orig_inter.append(
            APIInter(
                orig_list[-1].api_name,
                "",
                orig_list[-1].forward_output_shape,
                orig_list[-1].forward_output_type,
                orig_list[-1].forward_output_summery,
                [],
                [],
                [],
                index=len(orig_list),
            )
        )
        target_inter = [
            APIInter(
                "",
                target_list[0].api_name,
                [],
                [],
                [],
                target_list[0].forward_input_shape,
                target_list[0].forward_input_type,
                target_list[0].forward_input_summery,
                index=0,
            )
        ]
        for i in range(len(target_list) - 1):
            target_inter.append(
                APIInter(
                    target_list[i].api_name,
                    target_list[i + 1].api_name,
                    target_list[i].forward_output_shape,
                    target_list[i].forward_output_type,
                    target_list[i].forward_output_summery,
                    target_list[i + 1].forward_input_shape,
                    target_list[i + 1].forward_input_type,
                    target_list[i + 1].forward_input_summery,
                    index=i + 1,
                )
            )
        target_inter.append(
            APIInter(
                target_list[-1].api_name,
                "",
                target_list[-1].forward_output_shape,
                target_list[-1].forward_output_type,
                target_list[-1].forward_output_summery,
                [],
                [],
                [],
                index=len(target_list),
            )
        )
        return orig_inter, target_inter

    def _api_summery_sim(self, orig_inter, target_inter, err_threshold):
        input_summery_diff = np.abs(
            np.array(
                [
                    i - j
                    for i, j in zip(
                        orig_inter.forward_input_summery,
                        target_inter.forward_input_summery,
                    )
                ]
            )
        )
        input_summery_sum = np.array(
            [
                np.abs(i) + np.abs(j)
                for i, j in zip(
                    orig_inter.forward_input_summery,
                    target_inter.forward_input_summery,
                )
            ]
        )
        output_summery_diff = np.abs(
            np.array(
                [
                    i - j
                    for i, j in zip(
                        orig_inter.forward_output_summery,
                        target_inter.forward_output_summery,
                    )
                ]
            )
        )
        output_summery_sum = np.array(
            [
                np.abs(i) + np.abs(j)
                for i, j in zip(
                    orig_inter.forward_output_summery,
                    target_inter.forward_output_summery,
                )
            ]
        )
        if (
            np.any(input_summery_diff / (input_summery_sum + 1e-6)) <= err_threshold
            and np.any(output_summery_diff / (output_summery_sum + 1e-6))
            <= err_threshold
        ):
            dist = (
                np.arctan(
                    np.linalg.norm(input_summery_diff)
                    + np.linalg.norm(output_summery_diff)
                )
                * 2
                / np.pi
            )
        else:
            dist = 1
        return dist

    def _api_name_sim(self, orig_inter, target_inter):
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
            float(_min_distance(orig_inter.input_api_name, target_inter.input_api_name))
            / max(len(orig_inter.input_api_name), len(target_inter.input_api_name))
            if len(orig_inter.input_api_name) != 0
            or len(target_inter.input_api_name) != 0
            else 1.0
        )
        output_dist = (
            float(
                _min_distance(orig_inter.output_api_name, target_inter.output_api_name)
            )
            / max(len(orig_inter.output_api_name), len(target_inter.output_api_name))
            if len(orig_inter.output_api_name) != 0
            or len(target_inter.output_api_name) != 0
            else 1.0
        )
        # input_dist = 1. if input_dist > 0.5 else input_dist
        # output_dist = 1. if output_dist > 0.5 else output_dist
        return (input_dist + output_dist) / 2

    def _inter_match(self, orig_inter, target_inter, err_threshold):
        dp = [
            [0 for _ in range(len(target_inter) + 1)]
            for _ in range(len(orig_inter) + 1)
        ]
        path = [
            [0 for _ in range(len(target_inter) + 1)]
            for _ in range(len(orig_inter) + 1)
        ]

        dp[0] = [i for i in range(len(target_inter) + 1)]
        for i in range(len(orig_inter) + 1):
            dp[i][0] = i

        for i in range(1, len(orig_inter) + 1):
            for j in range(1, len(target_inter) + 1):
                dist = 1
                if orig_inter[i - 1] == target_inter[j - 1]:
                    if err_threshold < 1:
                        dist = self._api_summery_sim(
                            orig_inter[i - 1], target_inter[j - 1], err_threshold
                        )
                    else:
                        dist = 0

                    if dist < 1:
                        name_dist = self._api_name_sim(
                            orig_inter[i - 1], target_inter[j - 1]
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
        i, j = len(orig_inter), len(target_inter)
        while i > 0 and j > 0:
            if path[i][j] == 1:
                ret.append(
                    (
                        orig_inter[i - 1].inter_list_index,
                        target_inter[j - 1].inter_list_index,
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


class FlowMatch:
    def __init__(self) -> None:
        pass

    def _set_api_index(self, orig_list, target_list):
        for i, api in enumerate(orig_list):
            api.index = i
        for i, api in enumerate(target_list):
            api.index = i

    def __call__(
        self, orig_list: APIList, target_list: APIList, err_threshold: float = 1.0
    ):
        assert (
            err_threshold >= 0 and err_threshold <= 1
        ), "err_threshold must be in [0., 1.]"
        orig_list = orig_list.api_list
        target_list = target_list.api_list
        self._set_api_index(orig_list, target_list)
        match = self._convinced_match(orig_list, target_list)
        unmatch = self._get_unmatch_api(orig_list, target_list, match)
        unmatch_split = []
        for i in unmatch:
            if len(i[0]) == 0 or len(i[1]) == 0:
                unmatch_split.append(i)
                continue
            apis_match_fun = APIsInterMatch()
            unmatch_split.append(apis_match_fun(*i, err_threshold=err_threshold))
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

    def _convinced_match(self, orig_list, target_list):
        def _get_api_attr(api):
            return (
                api.uni_name,
                tuple(api.forward_input_shape),
                tuple(api.forward_output_shape),
            )

        api_count = {}
        api_match = []
        for api in orig_list:
            if api_count.get(_get_api_attr(api)):
                api_count[_get_api_attr(api)] += 1
            else:
                api_count[_get_api_attr(api)] = 1
        for api in target_list:
            if api_count.get(_get_api_attr(api)):
                api_count[_get_api_attr(api)] -= 1
        for k, v in api_count.items():
            if v == 0:
                orig_api = [api for api in orig_list if _get_api_attr(api) == k]
                target_api = [api for api in target_list if _get_api_attr(api) == k]
                api_match += list(zip(orig_api, target_api))
        api_match = sorted(api_match, key=lambda x: x[0].index)
        return api_match

    def _get_unmatch_api(self, orig_list, target_list, conv_map):
        last_orig = 0
        last_target = 0
        unmatch_orig = []
        unmatch_target = []
        for orig_api, target_api in conv_map:
            if not isinstance(orig_api, list) and not isinstance(orig_api, tuple):
                p_orig = orig_api.index
                unmatch_orig.append(orig_list[last_orig:p_orig])
                last_orig = p_orig + 1
            else:
                p_orig = orig_api[0].index
                unmatch_orig.append(orig_list[last_orig:p_orig])
                last_orig = orig_api[-1].index + 1
            if not isinstance(target_api, list) and not isinstance(target_api, tuple):
                p_target = target_api.index
                unmatch_target.append(target_list[last_target:p_target])
                last_target = p_target + 1
            else:
                p_target = target_api[0].index
                unmatch_target.append(orig_list[last_target:p_target])
                last_target = target_api[-1].index + 1
        unmatch_orig.append(orig_list[last_orig:])
        unmatch_target.append(target_list[last_target:])

        return list(zip(unmatch_orig, unmatch_target))
