import functools
import json
import re
from typing import Any

import numpy as np
from download_api_map import get_ops_dict


def print_apis_map(apis_map):
    for pt, ms in apis_map:
        if len(pt) > len(ms):
            ms += [""] * (len(pt) - len(ms))
        elif len(pt) < len(ms):
            pt += [""] * (len(ms) - len(pt))
        for pt_ops, ms_ops in zip(pt, ms):
            print(pt_ops, "\t", ms_ops)
        print("---------------------------------------------------------")


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
    def __init__(self) -> None:
        # 前面是输入映射，后面是输出映射
        self.pt_io_dict = {
            ('Functional','batch_norm'): ([0], [0]),
                           }

    def __call__(self, ops_list) -> Any:
        for ops in ops_list:
            if (ops.dump_type,ops.ops_name) in self.pt_io_dict:
                if len(self.pt_io_dict[(ops.dump_type,ops.ops_name)][0]) != 0:
                    forward_input_shape = []
                    forward_input_type = []
                    forward_input_summery = []
                    for i in self.pt_io_dict[(ops.dump_type,ops.ops_name)][0]:
                        forward_input_shape.append(ops.forward_input_shape[i])
                        forward_input_type.append(ops.forward_input_type[i])
                        forward_input_summery.append(ops.forward_input_summery[i])
                    ops.forward_input_shape = forward_input_shape
                    ops.forward_input_type = forward_input_type
                    ops.forward_input_summery = forward_input_summery
                if len(self.pt_io_dict[(ops.dump_type,ops.ops_name)][1]) != 0:
                    forward_output_shape = []
                    forward_output_type = []
                    forward_output_summery = []
                    for i in self.pt_io_dict[(ops.dump_type,ops.ops_name)][1]:
                        forward_output_shape.append(ops.forward_output_shape[i])
                        forward_output_type.append(ops.forward_output_type[i])
                        forward_output_summery.append(ops.forward_output_summery[i])

                    ops.forward_output_shape = forward_output_shape
                    ops.forward_output_type = forward_output_type
                    ops.forward_output_summery = forward_output_summery


class GetUniName:
    def __init__(self) -> None:
        pt_dict, ms_dict = self._get_ops_dict()
        self.uni_name_map = {"pytorch": pt_dict, "mindspore": ms_dict}

    def _get_ops_dict(self):
        ops_dict = get_ops_dict()
        ms_dict = {}
        pt_dict = {}
        for k, v in ops_dict.items():
            pt_ops = k.split(".")[-2:]
            ms_ops = v.split(".")[-2:]

            if ms_ops[0] not in ["nn", "ops", "Tensor"]:
                continue
            if pt_ops[0] not in ["torch", "functional", "Tensor", "Module", "nn"]:
                continue

            pt_ops_name, ms_ops_name = pt_ops[-1], ms_ops[-1]
            pt_ops_type, ms_ops_type = pt_ops[0].lower(), ms_ops[0].lower()
            ms_dict.update({(ms_ops_type, ms_ops_name): f"{ms_ops_type}_{ms_ops_name}"})
            pt_dict.update({(pt_ops_type, pt_ops_name): f"{ms_ops_type}_{ms_ops_name}"})
        return pt_dict, ms_dict

    def __call__(self, framework: str, dump_type: str, name: str):
        assert framework in [
            "pytorch",
            "mindspore",
        ], "framework should be pytorch or mindspore."
        if (dump_type.lower(), name) in self.uni_name_map[framework]:
            return self.uni_name_map[framework][(dump_type.lower(), name)]
        return f'{dump_type.lower()}_{name}'


class OPSDump:
    ops_dump_num = 0
    get_uni_name = GetUniName()

    def __init__(self, framework, dump_type, ops_name, ops_id, index=None):
        self.dump_type = dump_type
        self.ops_name = ops_name
        self.ops_id = ops_id
        self.uni_name = OPSDump.get_uni_name(framework, dump_type, ops_name)
        self.dump_id = OPSDump.ops_dump_num
        OPSDump.ops_dump_num += 1

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
        return f"{self.dump_type}_{self.ops_name}_{self.ops_id}"

    def __repr__(self) -> str:
        return f"{self.dump_type}_{self.ops_name}_{self.ops_id}"

    def update_ops(
        self,
        dump_type,
        ops_name,
        ops_id,
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
            and ops_name == self.ops_name
            and ops_id == self.ops_id
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


class OPSList:
    def __init__(self, framework):
        self.ops_list = []
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
            dump_type, ops_name, ops_id, direction, data_io, data_id = prefix_con

            if direction == "forward":
                if len(self.ops_list) == 0 or not self.ops_list[-1].update_ops(
                    dump_type,
                    ops_name,
                    ops_id,
                    direction,
                    data_io,
                    data_id,
                    data_type,
                    data_shape,
                    data_summery,
                ):
                    self.ops_list.append(
                        OPSDump(self.framework, dump_type, ops_name, ops_id)
                    )
                    self.ops_list[-1].update_ops(
                        dump_type,
                        ops_name,
                        ops_id,
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


class OPSInter:
    def __init__(
        self,
        output_ops_name,
        input_ops_name,
        forward_output_shape,
        forward_output_type,
        forward_output_summery,
        forward_input_shape,
        forward_input_type,
        forward_input_summery,
        index=[],
    ):
        self.output_ops_name = output_ops_name
        self.input_ops_name = input_ops_name
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

    def __call__(self, pt_list, ms_list, err_threshold=1):
        pt_inter, ms_inter = self._get_vertex(pt_list, ms_list)
        map_inter = self._inter_match(pt_inter, ms_inter, err_threshold=err_threshold)
        ret = self._get_map(pt_list, ms_list, map_inter)
        if len(ret[0][0]) == 0 and len(ret[0][1]) == 0:
            ret = ret[1:]
        if len(ret[-1][0]) == 0 and len(ret[-1][1]) == 0:
            ret = ret[:-1]
        return ret

    def _get_map(self, pt_list, ms_list, map_inter):
        ret = []
        pt_ptr = 0
        ms_ptr = 0
        for pt_idx, ms_idx in map_inter:
            ret.append((pt_list[pt_ptr:pt_idx], ms_list[ms_ptr:ms_idx]))
            pt_ptr, ms_ptr = pt_idx, ms_idx
        ret.append((pt_list[pt_ptr:], ms_list[ms_ptr:]))
        return ret

    def _get_vertex(self, pt_list, ms_list):
        pt_inter = [
            OPSInter(
                "",
                pt_list[0].ops_name,
                [],
                [],
                [],
                pt_list[0].forward_input_shape,
                pt_list[0].forward_input_type,
                pt_list[0].forward_input_summery,
                index=0,
            )
        ]
        for i in range(len(pt_list) - 1):
            pt_inter.append(
                OPSInter(
                    pt_list[i].ops_name,
                    pt_list[i + 1].ops_name,
                    pt_list[i].forward_output_shape,
                    pt_list[i].forward_output_type,
                    pt_list[i].forward_output_summery,
                    pt_list[i + 1].forward_input_shape,
                    pt_list[i + 1].forward_input_type,
                    pt_list[i + 1].forward_input_summery,
                    index=i + 1,
                )
            )
        pt_inter.append(
            OPSInter(
                pt_list[-1].ops_name,
                "",
                pt_list[-1].forward_output_shape,
                pt_list[-1].forward_output_type,
                pt_list[-1].forward_output_summery,
                [],
                [],
                [],
                index=len(pt_list),
            )
        )
        ms_inter = [
            OPSInter(
                "",
                ms_list[0].ops_name,
                [],
                [],
                [],
                ms_list[0].forward_input_shape,
                ms_list[0].forward_input_type,
                ms_list[0].forward_input_summery,
                index=0,
            )
        ]
        for i in range(len(ms_list) - 1):
            ms_inter.append(
                OPSInter(
                    ms_list[i].ops_name,
                    ms_list[i + 1].ops_name,
                    ms_list[i].forward_output_shape,
                    ms_list[i].forward_output_type,
                    ms_list[i].forward_output_summery,
                    ms_list[i + 1].forward_input_shape,
                    ms_list[i + 1].forward_input_type,
                    ms_list[i + 1].forward_input_summery,
                    index=i + 1,
                )
            )
        ms_inter.append(
            OPSInter(
                ms_list[-1].ops_name,
                "",
                ms_list[-1].forward_output_shape,
                ms_list[-1].forward_output_type,
                ms_list[-1].forward_output_summery,
                [],
                [],
                [],
                index=len(ms_list),
            )
        )
        return pt_inter, ms_inter

    def _ops_summery_sim(self, pt_inter, ms_inter,err_threshold):
        input_summery_diff = np.abs(
            np.array(
                [
                    i - j
                    for i, j in zip(
                        pt_inter.forward_input_summery,
                        ms_inter.forward_input_summery,
                    )
                ]
            )
        )
        input_summery_sum = np.array(
            [
                np.abs(i) + np.abs(j)
                for i, j in zip(
                    pt_inter.forward_input_summery,
                    ms_inter.forward_input_summery,
                )
            ]
        )
        output_summery_diff = np.abs(
            np.array(
                [
                    i - j
                    for i, j in zip(
                        pt_inter.forward_output_summery,
                        ms_inter.forward_output_summery,
                    )
                ]
            )
        )
        output_summery_sum = np.array(
            [
                np.abs(i) + np.abs(j)
                for i, j in zip(
                    pt_inter.forward_output_summery,
                    ms_inter.forward_output_summery,
                )
            ]
        )
        if (
            np.any(input_summery_diff / (input_summery_sum + 1e-6))
            <= err_threshold
            and np.any(
                output_summery_diff / (output_summery_sum + 1e-6)
            )
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

    def _ops_name_sim(self, pt_inter, ms_inter):
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
            float(_min_distance(pt_inter.input_ops_name, ms_inter.input_ops_name))
            / max(len(pt_inter.input_ops_name), len(ms_inter.input_ops_name))
            if len(pt_inter.input_ops_name) != 0 or len(ms_inter.input_ops_name) != 0
            else 1.
        )
        output_dist = (
            float(_min_distance(pt_inter.output_ops_name, ms_inter.output_ops_name))
            / max(len(pt_inter.output_ops_name), len(ms_inter.output_ops_name))
            if len(pt_inter.output_ops_name) != 0 or len(ms_inter.output_ops_name) != 0
            else 1.
        )
        # input_dist = 1. if input_dist > 0.5 else input_dist
        # output_dist = 1. if output_dist > 0.5 else output_dist
        return (input_dist + output_dist) / 2

    def _inter_match(self, pt_inter, ms_inter, err_threshold):
        dp = [[0 for _ in range(len(ms_inter) + 1)] for _ in range(len(pt_inter) + 1)]
        path = [[0 for _ in range(len(ms_inter) + 1)] for _ in range(len(pt_inter) + 1)]

        dp[0] = [i for i in range(len(ms_inter) + 1)]
        for i in range(len(pt_inter) + 1):
            dp[i][0] = i

        for i in range(1, len(pt_inter) + 1):
            for j in range(1, len(ms_inter) + 1):
                dist = 1
                if pt_inter[i - 1] == ms_inter[j - 1]:
                    if err_threshold < 1:
                        dist = self._ops_summery_sim(pt_inter[i - 1], ms_inter[j - 1], err_threshold)
                    else:
                        dist = 0

                    if dist < 1:
                        name_dist = self._ops_name_sim(pt_inter[i - 1], ms_inter[j - 1])
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
        i, j = len(pt_inter), len(ms_inter)
        while i > 0 and j > 0:
            if path[i][j] == 1:
                ret.append(
                    (pt_inter[i - 1].inter_list_index, ms_inter[j - 1].inter_list_index)
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

    def _set_ops_index(self, pt_list, ms_list):
        for i, ops in enumerate(pt_list):
            ops.index = i
        for i, ops in enumerate(ms_list):
            ops.index = i

    def __call__(self, pt_list, ms_list, err_threshold=1):
        pt_list = pt_list.ops_list
        ms_list = ms_list.ops_list
        self._set_ops_index(pt_list, ms_list)
        match = self._convinced_match(pt_list, ms_list)
        unmatch = self._get_unmatch_ops(pt_list, ms_list, match)
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

    def _convinced_match(self, pt_list, ms_list):
        def _get_ops_attr(ops):
            return (
                ops.uni_name,
                tuple(ops.forward_input_shape),
                tuple(ops.forward_output_shape),
            )

        ops_count = {}
        ops_match = []
        for ops in pt_list:
            if ops_count.get(_get_ops_attr(ops)):
                ops_count[_get_ops_attr(ops)] += 1
            else:
                ops_count[_get_ops_attr(ops)] = 1
        for ops in ms_list:
            if ops_count.get(_get_ops_attr(ops)):
                ops_count[_get_ops_attr(ops)] -= 1
        for k, v in ops_count.items():
            if v == 0:
                pt_ops = [ops for ops in pt_list if _get_ops_attr(ops) == k]
                ms_ops = [ops for ops in ms_list if _get_ops_attr(ops) == k]
                ops_match += list(zip(pt_ops, ms_ops))
        ops_match = sorted(ops_match, key=lambda x: x[0].index)
        return ops_match

    def _get_unmatch_ops(self, pt_list, ms_list, conv_map):
        last_pt = 0
        last_ms = 0
        unmatch_pt = []
        unmatch_ms = []
        for pt_ops, ms_ops in conv_map:
            if not isinstance(pt_ops, list) and not isinstance(pt_ops, tuple):
                p_pt = pt_ops.index
                unmatch_pt.append(pt_list[last_pt:p_pt])
                last_pt = p_pt + 1
            else:
                p_pt = pt_ops[0].index
                unmatch_pt.append(pt_list[last_pt:p_pt])
                last_pt = pt_ops[-1].index + 1
            if not isinstance(ms_ops, list) and not isinstance(ms_ops, tuple):
                p_ms = ms_ops.index
                unmatch_ms.append(ms_list[last_ms:p_ms])
                last_ms = p_ms + 1
            else:
                p_ms = ms_ops[0].index
                unmatch_ms.append(pt_list[last_ms:p_ms])
                last_ms = ms_ops[-1].index + 1
        unmatch_pt.append(pt_list[last_pt:])
        unmatch_ms.append(ms_list[last_ms:])

        return list(zip(unmatch_pt, unmatch_ms))
