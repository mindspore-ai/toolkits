import pandas as pd
from tqdm import tqdm
from decimal import Decimal

from recompute_profile.ir_graph_analy import extract_recompute_ops_from_ir_floder
from recompute_profile.kernel_details_analy_for_dict import analy_kernel_details, analy_recompute_result

import seaborn as sns
from recompute_profile.static_value import *
import argparse

def to_decimal(x):
    return Decimal(x).quantize(Decimal('0.000'))


def is_overlapping(interval1, interval2):
    """Check overlapping time intervals"""
    return (interval1[0] >= interval2[0] and interval1[0] < interval2[1]) or \
           (interval1[0] <= interval2[0] and interval1[1] > interval2[0])


def find_overlapping_intervals(list_of_intervals, target_interval):
    '''Binary search for overlapping time'''
    overlapping_indices = set()
    left, right = 0, len(list_of_intervals) - 1
    while left <= right:
        mid = (left + right) // 2
        if is_overlapping(list_of_intervals[mid], target_interval):
            overlapping_indices.add(mid)
        if list_of_intervals[mid][1] <= target_interval[0]:
            left = mid + 1
        else:
            right = mid - 1
    for i in range(left, len(list_of_intervals)):
        if is_overlapping(list_of_intervals[i], target_interval):
            overlapping_indices.add(i)
        else:
            break
    for i in range(right, -1, -1):
        if is_overlapping(list_of_intervals[i], target_interval):
            overlapping_indices.add(i)
        else:
            break
    return sorted(overlapping_indices)


class Overlap_Analyzer:
    """
    Analyzes performance overlap between kernel execution and data transfer.
    This class processes kernel_details files and IR files.
    """
    def __init__(self, kernel_details, save_path="./"):
        self.path = save_path

        details_df = pd.read_csv(kernel_details, converters={"Start Time(us)": to_decimal, "Duration(us)": to_decimal})
        if 'aic_mac_ratio' in details_df.columns:
            details_df = details_df[["Name", "Accelerator Core", "Start Time(us)", "Duration(us)", "Type", "aic_mac_ratio"]]
            self.aic_mac_ratio = True
        else:
            details_df = details_df[["Name", "Accelerator Core", "Start Time(us)", "Duration(us)", "Type"]]
            self.aic_mac_ratio = False
        start_time = details_df["Start Time(us)"][0]
        details_df["Type"] = details_df["Type"]
        details_df["Start Time(us)"] = details_df["Start Time(us)"] - start_time
        details_df["end_time"] = details_df["Start Time(us)"] + details_df["Duration(us)"]
        details_df.rename(columns={'Start Time(us)': 'start_time'}, inplace=True)
        details_df = details_df[details_df["Accelerator Core"] != "AI_CPU"]

        self.op_type_priority_template = OVERLAP_PRIORITY
        self.communication_priority_re = COMMUNICATION_OVERLAP_PRIORITY_RE
        def compute_value(value):
            if value["Accelerator Core"] != "COMMUNICATION":
                if "FlashAttentionScoreGrad" in value["Name"]:
                    return "fa_back"
                elif "FlashAttentionScore" in value["Name"]:
                    return "fa_forward"
                elif "matmul" in value["Name"].lower() and "memset" not in value["Name"].lower():
                    return "matmul"
                else:
                    return "vector"
            else:
                for communication_op_type in self.communication_priority_re:
                    if communication_op_type in value["Name"]:
                        return communication_op_type
                return GLOBAL_COMMUNICATION_OTHER
        details_df["op_complex_type"] = details_df.apply(compute_value, axis=1)
        details_df[OVERLAP_TIME] = Decimal(0)
        details_df[NOT_OVERLAP_TIME] = details_df["Duration(us)"]
        details_df["recompute"] = "no-match"
        details_df["is_dp_allgather"] = ""
        details_df["unique_id"] = ""
        details_df["forward_unique_id"] = ""
        details_df["micro"] = ""

        self.comunication_op_type = details_df[details_df['Accelerator Core'] == 'COMMUNICATION']['op_complex_type'].drop_duplicates().tolist()
        self.op_type = details_df['op_complex_type'].drop_duplicates().tolist()
        self.op_dict = details_df.to_dict(orient='records')
        details_df = None
        import gc
        gc.collect()
        self.op_dict.sort(key=lambda x: (x["start_time"], x["end_time"]))

        def get_priority(data):
            for key in self.op_type_priority_template.keys():
                if key in data:
                    return self.op_type_priority_template[key]
            return self.op_type_priority_template[GLOBAL_OP_TYPE_OTHER]
        self.op_type.sort(key=lambda x: get_priority(x))

        self.op_type_priority = {key: value for value, key in enumerate(self.op_type)}
        self.op_type_priority_keys = list(self.op_type_priority.keys())
        self.op_analy_result = {
            item: {TOTAL_TIME: to_decimal(0), OVERLAP_TIME: to_decimal(0), OVERLAP_RATIO: 0, NOT_OVERLAP_TIME: to_decimal(0),
                   RATIO_OF_TOTAL_TIME:0, OP_COUNT: 0, RECOMPUTE_COUNT: 0, RECOMPUTE_TIME: 0, RECOMPUTE_OVERLAP_TIME: to_decimal(0),
                   RECOMPUTE_NOT_OVERLAP_TIME: to_decimal(0)} for
            item in self.op_type}
        self.free_time = Decimal(0)
        self.total_step_time = self.get_total_step_time()

        self.dp_allgather = {TOTAL_TIME: to_decimal(0), OVERLAP_TIME: to_decimal(0), OVERLAP_RATIO: 0, NOT_OVERLAP_TIME: to_decimal(0),
                   RATIO_OF_TOTAL_TIME:0, OP_COUNT: 0, RECOMPUTE_COUNT: 0, RECOMPUTE_TIME: 0, RECOMPUTE_OVERLAP_TIME: to_decimal(0),
                   RECOMPUTE_NOT_OVERLAP_TIME: to_decimal(0)}

    def is_valid_overlap_time(self, start_time_input, end_time_input):
        return end_time_input > start_time_input

    def get_total_step_time(self):
        return self.op_dict[-1]["end_time"] - self.op_dict[0]["start_time"]

    def process_overlap_time(self, overlap_start_time, overlap_end_time, op_complex_type,
                             index, overlap_time_pipeline):
        overlap_time = max(overlap_end_time - overlap_start_time, 0)
        is_valid = self.is_valid_overlap_time(overlap_start_time, overlap_end_time)
        if is_valid:
            overlap_time_pipeline.append({OVERLAP_TIME: [overlap_start_time, overlap_end_time],
                                          "op_complex_type": op_complex_type,
                                          "index": index})

    def process_overlap_time_pipeline(self, overlap_time_pipeline):
        if len(overlap_time_pipeline) == 0:
            return
        overlap_time_pipeline.sort(key=lambda x: (self.op_type_priority[x["op_complex_type"]],
                                                  x[OVERLAP_TIME][0],
                                                  x[OVERLAP_TIME][1]))

        def merge_intervals(intervals):
            """Merge overlapping time intervals"""
            if not intervals:
                return []
            intervals.sort(key=lambda x: (x[0], x[1]))
            merged = [intervals[0]]
            for current in intervals[1:]:
                last_merged = merged[-1]
                if current[0] <= last_merged[1]:
                    merged[-1] = [last_merged[0], max(last_merged[1], current[1])]
                else:
                    merged.append(current)
            return merged

        overlap_time_merge = []
        for item_index in range(len(overlap_time_pipeline)):
            s1 = overlap_time_pipeline[item_index][OVERLAP_TIME][0]
            e1 = overlap_time_pipeline[item_index][OVERLAP_TIME][1]
            overlapping_interval_index = find_overlapping_intervals(overlap_time_merge,
                                                                    overlap_time_pipeline[item_index][OVERLAP_TIME])
            if len(overlapping_interval_index) == 0:
                overlap_time_merge.append([s1, e1])
            else:
                s2 = overlap_time_merge[overlapping_interval_index[0]][0]
                e2 = overlap_time_merge[overlapping_interval_index[-1]][1]
                interval_time = Decimal(0)
                for interval_index in range(1, len(overlapping_interval_index)):
                    interval_time += overlap_time_merge[overlapping_interval_index[interval_index]][0] - \
                                     overlap_time_merge[overlapping_interval_index[interval_index - 1]][1]
                if s2 >= s1 and e2 <= e1:
                    # case1
                    self.op_dict[overlap_time_pipeline[item_index]["index"]][OVERLAP_TIME] += e2 - s2 - interval_time
                    self.op_dict[overlap_time_pipeline[item_index]["index"]][
                        NOT_OVERLAP_TIME] -= e2 - s2 - interval_time
                    self.op_analy_result[overlap_time_pipeline[item_index]["op_complex_type"]][
                        OVERLAP_TIME] += e2 - s2 - interval_time
                elif s2 < s1 and e2 <= e1:
                    # case2
                    self.op_dict[overlap_time_pipeline[item_index]["index"]][OVERLAP_TIME] += e2 - s1 - interval_time
                    self.op_dict[overlap_time_pipeline[item_index]["index"]][
                        NOT_OVERLAP_TIME] -= e2 - s1 - interval_time
                    self.op_analy_result[overlap_time_pipeline[item_index]["op_complex_type"]][
                        OVERLAP_TIME] += e2 - s1 - interval_time
                elif s2 >= s1 and e2 > e1:
                    # case3
                    self.op_dict[overlap_time_pipeline[item_index]["index"]][OVERLAP_TIME] += e1 - s2 - interval_time
                    self.op_dict[overlap_time_pipeline[item_index]["index"]][
                        NOT_OVERLAP_TIME] -= e1 - s2 - interval_time
                    self.op_analy_result[overlap_time_pipeline[item_index]["op_complex_type"]][
                        OVERLAP_TIME] += e1 - s2 - interval_time
                else:
                    # case4
                    self.op_dict[overlap_time_pipeline[item_index]["index"]][OVERLAP_TIME] += e1 - s1 - interval_time
                    self.op_dict[overlap_time_pipeline[item_index]["index"]][NOT_OVERLAP_TIME] -= e1 - s1 - interval_time
                    self.op_analy_result[overlap_time_pipeline[item_index]["op_complex_type"]][
                        OVERLAP_TIME] += e1 - s1 - interval_time
                overlap_time_merge.append([s1, e1])
                # Duration(us)
            overlap_time_merge = merge_intervals(overlap_time_merge)

    def get_overlap_max_time(self, name, overlap_max_time, s2):
        if name not in overlap_max_time:
            overlap_max_time[name] = 0
            overlap_start_time = s2
        else:
            overlap_start_time = max(s2, overlap_max_time[name])
        return overlap_start_time

    def get_overlap_analyse_result(self):
        '''Get the analysis result of overlaps.'''
        overlap_max_time = {}
        overlap_time_pipeline = []
        if len(self.op_dict) <= 0:
            raise Exception("len(hccl_data) <= 0, so overlap time array and total time array is empty!")

        left_value_index = 0
        self.op_analy_result[self.op_dict[left_value_index]["op_complex_type"]][TOTAL_TIME] += \
        self.op_dict[left_value_index]["end_time"] - \
        self.op_dict[left_value_index]["start_time"]
        self.op_analy_result[self.op_dict[left_value_index]["op_complex_type"]][OP_COUNT] += 1
        for index in range(1, len(self.op_dict)):
            s1 = self.op_dict[left_value_index]["start_time"]
            e1 = self.op_dict[left_value_index]["end_time"]
            s2 = self.op_dict[index]["start_time"]
            e2 = self.op_dict[index]["end_time"]
            op_complex_type_left = self.op_dict[left_value_index]["op_complex_type"]
            op_complex_type_current = self.op_dict[index]["op_complex_type"]
            name_left = self.op_dict[left_value_index]["Name"]
            name_current = self.op_dict[index]["Name"]
            if e1 <= s2:
                left_value_index = index
                self.process_overlap_time_pipeline(overlap_time_pipeline)
                overlap_time_pipeline = []
            elif e1 > s2 and e1 <= e2:
                overlap_start_time = self.get_overlap_max_time(name_left, overlap_max_time, s2)
                overlap_end_time = e1
                self.process_overlap_time(overlap_start_time, overlap_end_time, op_complex_type_left,
                                          left_value_index, overlap_time_pipeline)
                overlap_max_time[name_left] = max(max(s1, e1), overlap_max_time[name_left])

                overlap_start_time = self.get_overlap_max_time(name_current, overlap_max_time, s2)
                overlap_end_time = e1
                self.process_overlap_time(overlap_start_time, overlap_end_time, op_complex_type_current,
                                          index, overlap_time_pipeline)
                overlap_max_time[name_current] = max(max(s2, e1), overlap_max_time[name_current])
                overlap_max_time.pop(name_left)
                left_value_index = index
            elif e1 > s2 and e1 > e2:
                overlap_start_time = self.get_overlap_max_time(name_left, overlap_max_time, s2)
                overlap_end_time = e2
                self.process_overlap_time(overlap_start_time, overlap_end_time, op_complex_type_left,
                                          left_value_index, overlap_time_pipeline)
                overlap_max_time[name_left] = max(max(s1, e2), overlap_max_time[name_left])

                overlap_start_time = self.get_overlap_max_time(name_current, overlap_max_time, s2)
                overlap_end_time = e2
                self.process_overlap_time(overlap_start_time, overlap_end_time, op_complex_type_current,
                                          index, overlap_time_pipeline)
                overlap_max_time[name_current] = max(max(s2, e2), overlap_max_time[name_current])
                overlap_max_time.pop(name_current)
            self.op_analy_result[op_complex_type_current][TOTAL_TIME] += e2 - s2
            self.op_analy_result[op_complex_type_current][OP_COUNT] += 1
        self.process_overlap_time_pipeline(overlap_time_pipeline)

    def analyse(self):
        self.get_overlap_analyse_result()
        self.analy_result()

    def analy_result(self):
        '''Summary and analysis of detailed data'''
        total_not_overlap_time = Decimal(0)
        total_compute_time = Decimal(0)
        total_compute_overlap_time = Decimal(0)
        total_compute_not_overlap_time = Decimal(0)
        total_comunication_time = Decimal(0)
        total_comunication_overlap_time = Decimal(0)
        total_comunication_not_overlap_time = Decimal(0)
        totoal_op_count = 0
        total_compute_count = 0
        total_comunication_count = 0
        for op_type in self.op_type:
            self.op_analy_result[op_type][NOT_OVERLAP_TIME] = self.op_analy_result[op_type][TOTAL_TIME] - \
                                                                self.op_analy_result[op_type][OVERLAP_TIME]
            total_not_overlap_time += self.op_analy_result[op_type][NOT_OVERLAP_TIME]
            totoal_op_count += self.op_analy_result[op_type][OP_COUNT]
            if op_type in self.comunication_op_type:
                total_comunication_not_overlap_time += self.op_analy_result[op_type][NOT_OVERLAP_TIME]
                total_comunication_time += self.op_analy_result[op_type][TOTAL_TIME]
                total_comunication_overlap_time += self.op_analy_result[op_type][OVERLAP_TIME]
                total_comunication_count += self.op_analy_result[op_type][OP_COUNT]
            else:
                total_compute_not_overlap_time += self.op_analy_result[op_type][NOT_OVERLAP_TIME]
                total_compute_time += self.op_analy_result[op_type][TOTAL_TIME]
                total_compute_overlap_time += self.op_analy_result[op_type][OVERLAP_TIME]
                total_compute_count += self.op_analy_result[op_type][OP_COUNT]

            self.op_analy_result[op_type][RATIO_OF_TOTAL_TIME] = self.op_analy_result[op_type][NOT_OVERLAP_TIME] / self.total_step_time
            self.op_analy_result[op_type][OVERLAP_RATIO] = self.op_analy_result[op_type][OVERLAP_TIME] / self.op_analy_result[op_type][TOTAL_TIME]

            total_time_float = float(self.op_analy_result[op_type][TOTAL_TIME]) / US_TO_MS
            overlap_time = float(self.op_analy_result[op_type][OVERLAP_TIME]) / US_TO_MS
            overlap_ratio = float(self.op_analy_result[op_type][OVERLAP_RATIO])
            not_overlap_time = float(self.op_analy_result[op_type][NOT_OVERLAP_TIME]) / US_TO_MS
            op_count = self.op_analy_result[op_type][OP_COUNT]
            print(f"{op_type}:    "
                  f"total time: {total_time_float} ms,"
                  f"overlap time: {overlap_time} ms,"
                  f"overlap_ratio: {overlap_ratio},"
                  f"not overlap time: {not_overlap_time} ms,"
                  f"op_count: {op_count}")
        self.free_time = self.total_step_time - total_not_overlap_time
        self.op_analy_result["compute_time"] = {
                        TOTAL_TIME: total_compute_time,
                        OVERLAP_TIME: total_compute_overlap_time,
                        OVERLAP_RATIO: total_compute_overlap_time/total_compute_time,
                        NOT_OVERLAP_TIME: total_compute_not_overlap_time,
                        RATIO_OF_TOTAL_TIME:total_compute_not_overlap_time / self.total_step_time,
                        OP_COUNT: total_compute_count,
                        RECOMPUTE_COUNT: 0,
                        RECOMPUTE_OVERLAP_TIME: 0,
                        RECOMPUTE_NOT_OVERLAP_TIME: 0,
                        RECOMPUTE_TIME: 0,
        }
        self.op_analy_result["comunication_time"] = {
                                    TOTAL_TIME: total_comunication_time,
                                    OVERLAP_TIME: total_comunication_overlap_time,
                                    OVERLAP_RATIO: total_comunication_overlap_time / total_comunication_time,
                                    NOT_OVERLAP_TIME: total_comunication_not_overlap_time,
                                     RATIO_OF_TOTAL_TIME: total_comunication_not_overlap_time / self.total_step_time,
                                    OP_COUNT: total_comunication_count,
                                    RECOMPUTE_COUNT: 0,
                                    RECOMPUTE_OVERLAP_TIME: 0,
                                    RECOMPUTE_NOT_OVERLAP_TIME: 0,
                                    RECOMPUTE_TIME: 0,
        }
        self.op_analy_result["free_time"] = {
            TOTAL_TIME: self.free_time,
            OVERLAP_TIME: 0,
            OVERLAP_RATIO: 0,
            NOT_OVERLAP_TIME: self.free_time,
            RATIO_OF_TOTAL_TIME: self.free_time / self.total_step_time,
            OP_COUNT: 0,
            RECOMPUTE_COUNT: 0,
            RECOMPUTE_OVERLAP_TIME: 0,
            RECOMPUTE_NOT_OVERLAP_TIME: 0,
            RECOMPUTE_TIME: 0,
        }
        self.op_analy_result[TOTAL_TIME] = {
            TOTAL_TIME: total_compute_time + total_comunication_time + self.free_time,
            OVERLAP_TIME: total_compute_overlap_time + total_comunication_overlap_time,
            OVERLAP_RATIO: (total_compute_overlap_time + total_comunication_overlap_time) / (total_compute_time + total_comunication_time + self.free_time),
            NOT_OVERLAP_TIME: self.total_step_time,
            RATIO_OF_TOTAL_TIME: 1,
            OP_COUNT: totoal_op_count,
            RECOMPUTE_COUNT: 0,
            RECOMPUTE_OVERLAP_TIME: 0,
            RECOMPUTE_NOT_OVERLAP_TIME: 0,
            RECOMPUTE_TIME: 0,
        }
        return self.op_analy_result

    def analy_recompute(self, ir_file_path=""):
        '''recompute analysis'''
        if ir_file_path == "":
            return
        summary = extract_recompute_ops_from_ir_floder(ir_file_path)
        detail_df, summary = analy_kernel_details(self.op_dict, summary)

        for result_index in detail_df:
            op_complex_type = result_index["op_complex_type"]
            if result_index["recompute"] == "recompute":
                self.op_analy_result[op_complex_type][RECOMPUTE_COUNT] += 1
                self.op_analy_result[op_complex_type][RECOMPUTE_OVERLAP_TIME] += result_index[OVERLAP_TIME]
                self.op_analy_result[op_complex_type][RECOMPUTE_NOT_OVERLAP_TIME] += result_index[NOT_OVERLAP_TIME]
                self.op_analy_result[op_complex_type][RECOMPUTE_TIME] += result_index["Duration(us)"]
            if result_index["is_dp_allgather"] == "dp":
                self.dp_allgather[TOTAL_TIME] += result_index["Duration(us)"]
                self.dp_allgather[OVERLAP_TIME] += result_index[OVERLAP_TIME]
                self.dp_allgather[NOT_OVERLAP_TIME] += result_index[NOT_OVERLAP_TIME]
                self.dp_allgather[OP_COUNT] += 1
                if result_index["recompute"] == "recompute":
                    self.dp_allgather[RECOMPUTE_COUNT] += 1
                    self.dp_allgather[RECOMPUTE_OVERLAP_TIME] += result_index[RECOMPUTE_OVERLAP_TIME]
                    self.dp_allgather[RECOMPUTE_NOT_OVERLAP_TIME] += result_index[RECOMPUTE_NOT_OVERLAP_TIME]
                    self.dp_allgather[RECOMPUTE_TIME] += result_index[RECOMPUTE_TIME]

        for op_type in self.op_type:
            if op_type in self.comunication_op_type:
                self.op_analy_result["comunication_time"][RECOMPUTE_COUNT] +=self.op_analy_result[op_type][RECOMPUTE_COUNT]
                self.op_analy_result["comunication_time"][RECOMPUTE_OVERLAP_TIME] += self.op_analy_result[op_type][
                    RECOMPUTE_OVERLAP_TIME]
                self.op_analy_result["comunication_time"][RECOMPUTE_NOT_OVERLAP_TIME] += self.op_analy_result[op_type][
                    RECOMPUTE_NOT_OVERLAP_TIME]
                self.op_analy_result["comunication_time"][RECOMPUTE_TIME] += \
                self.op_analy_result[op_type][
                    RECOMPUTE_TIME]
            else:
                self.op_analy_result["compute_time"][RECOMPUTE_COUNT] += self.op_analy_result[op_type][
                    RECOMPUTE_COUNT]
                self.op_analy_result["compute_time"][RECOMPUTE_OVERLAP_TIME] += self.op_analy_result[op_type][
                    RECOMPUTE_OVERLAP_TIME]
                self.op_analy_result["compute_time"][RECOMPUTE_NOT_OVERLAP_TIME] += \
                self.op_analy_result[op_type][RECOMPUTE_NOT_OVERLAP_TIME]
                self.op_analy_result["compute_time"][RECOMPUTE_TIME] += \
                    self.op_analy_result[op_type][RECOMPUTE_TIME]
            self.op_analy_result[TOTAL_TIME][RECOMPUTE_COUNT] += self.op_analy_result[op_type][
                RECOMPUTE_COUNT]
            self.op_analy_result[TOTAL_TIME][RECOMPUTE_OVERLAP_TIME] += self.op_analy_result[op_type][
                RECOMPUTE_OVERLAP_TIME]
            self.op_analy_result[TOTAL_TIME][RECOMPUTE_NOT_OVERLAP_TIME] += \
                self.op_analy_result[op_type][RECOMPUTE_NOT_OVERLAP_TIME]
            self.op_analy_result[TOTAL_TIME][RECOMPUTE_TIME] += \
                self.op_analy_result[op_type][RECOMPUTE_TIME]

    def analy_parallel(self):
        '''Parallel reanalysis'''
        self.op_analy_result["----->parallel summary:"] = {}
        fa_time = self.op_analy_result["fa_forward"][NOT_OVERLAP_TIME] + self.op_analy_result["fa_back"][NOT_OVERLAP_TIME]
        self.op_analy_result["fa"] = {TOTAL_TIME: self.op_analy_result["fa_forward"][TOTAL_TIME] + self.op_analy_result["fa_back"][TOTAL_TIME],
                                              OVERLAP_TIME: self.op_analy_result["fa_forward"][OVERLAP_TIME] + self.op_analy_result["fa_back"][OVERLAP_TIME],
                                               NOT_OVERLAP_TIME: fa_time,
                                              RATIO_OF_TOTAL_TIME: fa_time / self.op_analy_result[TOTAL_TIME][NOT_OVERLAP_TIME],
                                              OP_COUNT: self.op_analy_result["fa_forward"][OP_COUNT] + self.op_analy_result["fa_back"][OP_COUNT],
                                              RECOMPUTE_COUNT: self.op_analy_result["fa_forward"][RECOMPUTE_COUNT] + self.op_analy_result["fa_back"][RECOMPUTE_COUNT],
                                              RECOMPUTE_OVERLAP_TIME: self.op_analy_result["fa_forward"][RECOMPUTE_OVERLAP_TIME] + self.op_analy_result["fa_back"][RECOMPUTE_OVERLAP_TIME],
                                              RECOMPUTE_NOT_OVERLAP_TIME: self.op_analy_result["fa_forward"][RECOMPUTE_NOT_OVERLAP_TIME] + self.op_analy_result["fa_back"][RECOMPUTE_NOT_OVERLAP_TIME],
                                              RECOMPUTE_TIME: self.op_analy_result["fa_forward"][RECOMPUTE_TIME] + self.op_analy_result["fa_back"][RECOMPUTE_TIME]
                                              }
        if "grouped_matmul" in self.op_analy_result:
            cube_time = self.op_analy_result["grouped_matmul"][NOT_OVERLAP_TIME] + self.op_analy_result["matmul"][NOT_OVERLAP_TIME]
            self.op_analy_result["cube(cube-fa)"] = {
                TOTAL_TIME: self.op_analy_result["grouped_matmul"][TOTAL_TIME] + self.op_analy_result["matmul"][
                    TOTAL_TIME],
                OVERLAP_TIME: self.op_analy_result["grouped_matmul"][OVERLAP_TIME] + self.op_analy_result["matmul"][
                    OVERLAP_TIME],
                NOT_OVERLAP_TIME: cube_time,
                RATIO_OF_TOTAL_TIME: cube_time / self.op_analy_result[TOTAL_TIME][NOT_OVERLAP_TIME],
                OP_COUNT: self.op_analy_result["grouped_matmul"][OP_COUNT] + self.op_analy_result["matmul"][
                    OP_COUNT],
                RECOMPUTE_COUNT: self.op_analy_result["grouped_matmul"][RECOMPUTE_COUNT] +
                                   self.op_analy_result["matmul"][
                                       RECOMPUTE_COUNT],
                RECOMPUTE_OVERLAP_TIME: self.op_analy_result["grouped_matmul"][RECOMPUTE_OVERLAP_TIME] +
                                          self.op_analy_result["matmul"][RECOMPUTE_OVERLAP_TIME],
                RECOMPUTE_NOT_OVERLAP_TIME: self.op_analy_result["grouped_matmul"][RECOMPUTE_NOT_OVERLAP_TIME] +
                                              self.op_analy_result["matmul"][RECOMPUTE_NOT_OVERLAP_TIME],
                RECOMPUTE_TIME: self.op_analy_result["grouped_matmul"][RECOMPUTE_TIME] +
                                  self.op_analy_result["matmul"][
                                      RECOMPUTE_TIME]
            }
        else:
            cube_time = self.op_analy_result["matmul"][NOT_OVERLAP_TIME]
            self.op_analy_result["cube(cube-fa)"] = {
                TOTAL_TIME: self.op_analy_result["matmul"][TOTAL_TIME],
                OVERLAP_TIME: self.op_analy_result["matmul"][OVERLAP_TIME],
                NOT_OVERLAP_TIME: cube_time,
                RATIO_OF_TOTAL_TIME: cube_time / self.op_analy_result[TOTAL_TIME][NOT_OVERLAP_TIME],
                OP_COUNT: self.op_analy_result["matmul"][OP_COUNT],
                RECOMPUTE_COUNT: self.op_analy_result["matmul"][
                    RECOMPUTE_COUNT],
                RECOMPUTE_OVERLAP_TIME: self.op_analy_result["matmul"][RECOMPUTE_OVERLAP_TIME],
                RECOMPUTE_NOT_OVERLAP_TIME: self.op_analy_result["matmul"][RECOMPUTE_NOT_OVERLAP_TIME],
                RECOMPUTE_TIME: self.op_analy_result["matmul"][
                    RECOMPUTE_TIME]
            }
        self.op_analy_result["vector op"] = {
            TOTAL_TIME: self.op_analy_result["vector"][TOTAL_TIME],
            OVERLAP_TIME: self.op_analy_result["vector"][OVERLAP_TIME],
            NOT_OVERLAP_TIME: self.op_analy_result["vector"][NOT_OVERLAP_TIME],
            RATIO_OF_TOTAL_TIME: self.op_analy_result["vector"][RATIO_OF_TOTAL_TIME],
            OP_COUNT: self.op_analy_result["vector"][OP_COUNT],
            RECOMPUTE_COUNT: self.op_analy_result["vector"][RECOMPUTE_COUNT],
            RECOMPUTE_OVERLAP_TIME: self.op_analy_result["vector"][RECOMPUTE_OVERLAP_TIME],
            RECOMPUTE_NOT_OVERLAP_TIME: self.op_analy_result["vector"][RECOMPUTE_NOT_OVERLAP_TIME] ,
            RECOMPUTE_TIME: self.op_analy_result["vector"][RECOMPUTE_TIME]}
        self.op_analy_result["compute"] = {
            TOTAL_TIME: self.op_analy_result["compute_time"][TOTAL_TIME],
            OVERLAP_TIME: self.op_analy_result["compute_time"][OVERLAP_TIME],
            NOT_OVERLAP_TIME: self.op_analy_result["compute_time"][NOT_OVERLAP_TIME],
            RATIO_OF_TOTAL_TIME: self.op_analy_result["compute_time"][RATIO_OF_TOTAL_TIME],
            OP_COUNT: self.op_analy_result["compute_time"][OP_COUNT],
            RECOMPUTE_COUNT: self.op_analy_result["compute_time"][RECOMPUTE_COUNT],
            RECOMPUTE_OVERLAP_TIME: self.op_analy_result["compute_time"][RECOMPUTE_OVERLAP_TIME],
            RECOMPUTE_NOT_OVERLAP_TIME: self.op_analy_result["compute_time"][RECOMPUTE_NOT_OVERLAP_TIME],
            RECOMPUTE_TIME: self.op_analy_result["compute_time"][RECOMPUTE_TIME]
        }
        def get_sumarry(op_key_list, summary_key):
            time = Decimal(0)
            time_overlap = Decimal(0)
            time_total = Decimal(0)
            op_count = 0
            recompute_count = Decimal(0)
            recompute_overlap_time = Decimal(0)
            recompute_not_overlap_time = Decimal(0)
            recompute_time = Decimal(0)
            for key in self.op_analy_result.keys():
                for op_key in op_key_list:
                    if op_key in key:
                        time += self.op_analy_result[key][NOT_OVERLAP_TIME]
                        time_overlap += self.op_analy_result[key][OVERLAP_TIME]
                        time_total += self.op_analy_result[key][TOTAL_TIME]
                        op_count += self.op_analy_result[key][OP_COUNT]
                        recompute_count += self.op_analy_result[key][RECOMPUTE_COUNT]
                        recompute_overlap_time += self.op_analy_result[key][RECOMPUTE_OVERLAP_TIME]
                        recompute_not_overlap_time += self.op_analy_result[key][RECOMPUTE_NOT_OVERLAP_TIME]
                        recompute_time += self.op_analy_result[key][RECOMPUTE_TIME]
            if summary_key == "tp":
                time -= self.dp_allgather[NOT_OVERLAP_TIME]
                time_overlap -= self.dp_allgather[OVERLAP_TIME]
                time_total -= self.dp_allgather[TOTAL_TIME]
                op_count -= self.dp_allgather[OP_COUNT]
                recompute_count -= self.dp_allgather[RECOMPUTE_COUNT]
                recompute_overlap_time -= self.dp_allgather[RECOMPUTE_OVERLAP_TIME]
                recompute_not_overlap_time -= self.dp_allgather[RECOMPUTE_NOT_OVERLAP_TIME]
                recompute_time -= self.dp_allgather[RECOMPUTE_TIME]
            elif summary_key == "dp":
                time += self.dp_allgather[NOT_OVERLAP_TIME]
                time_overlap += self.dp_allgather[OVERLAP_TIME]
                time_total += self.dp_allgather[TOTAL_TIME]
                op_count += self.dp_allgather[OP_COUNT]
                recompute_count += self.dp_allgather[RECOMPUTE_COUNT]
                recompute_overlap_time += self.dp_allgather[RECOMPUTE_OVERLAP_TIME]
                recompute_not_overlap_time += self.dp_allgather[RECOMPUTE_NOT_OVERLAP_TIME]
                recompute_time += self.dp_allgather[RECOMPUTE_TIME]
            if time_total == 0:
                self.op_analy_result[summary_key] = {
                    TOTAL_TIME: time_total,
                    OVERLAP_TIME: 0,
                    OVERLAP_RATIO: 0,
                    NOT_OVERLAP_TIME: 0,
                    RATIO_OF_TOTAL_TIME: 0,
                    OP_COUNT: 0,
                    RECOMPUTE_COUNT: 0,
                    RECOMPUTE_OVERLAP_TIME: 0,
                    RECOMPUTE_NOT_OVERLAP_TIME: 0,
                    RECOMPUTE_TIME: 0
                }
                return
            ratio = time / self.op_analy_result[TOTAL_TIME][NOT_OVERLAP_TIME]
            overlap_ratio = time_overlap / time_total
            self.op_analy_result[summary_key] = {
                TOTAL_TIME: time_total,
                OVERLAP_TIME: time_total - time,
                OVERLAP_RATIO: overlap_ratio,
                NOT_OVERLAP_TIME: time,
                RATIO_OF_TOTAL_TIME: ratio,
                OP_COUNT: op_count,
                RECOMPUTE_COUNT: recompute_count,
                RECOMPUTE_OVERLAP_TIME: recompute_overlap_time,
                RECOMPUTE_NOT_OVERLAP_TIME: recompute_not_overlap_time,
                RECOMPUTE_TIME: recompute_time
            }

        for key, value in COMMUNICATION_PARALLEL.items():
            get_sumarry(value, key)
        get_sumarry(["comunication_time"], "comunication")
        get_sumarry(["free_time"], "free")
        get_sumarry([TOTAL_TIME], "total")

        for op_type in self.op_analy_result.keys():
            if "free" in op_type or "----->" in op_type or self.op_analy_result[op_type][NOT_OVERLAP_TIME] == 0:
                continue
            self.op_analy_result[op_type][RECOMPUTE_RATIO] = \
                self.op_analy_result[op_type][RECOMPUTE_NOT_OVERLAP_TIME] / self.op_analy_result[op_type][NOT_OVERLAP_TIME]
        self.op_analy_result.pop("compute_time")
        self.op_analy_result.pop("comunication_time")
        self.op_analy_result.pop("free_time")
        self.op_analy_result.pop(TOTAL_TIME)

    def us_to_ms(self):
        '''Build a new dictionary in the specified order.'''
        for result_index in self.op_analy_result:
            for result_key in self.op_analy_result[result_index].keys():
                if "ratio" in result_key:
                    self.op_analy_result[result_index][result_key] = round(
                        self.op_analy_result[result_index][result_key], 3)
                elif "time" in result_key:
                    self.op_analy_result[result_index][result_key] = round(
                        self.op_analy_result[result_index][result_key] / MS_TO_S, 3)

    def save_result(self):
        df1 = pd.DataFrame(self.op_analy_result).transpose()
        df2 = pd.DataFrame(self.op_dict)

        with pd.ExcelWriter(self.path) as writer:
            df1.to_excel(writer, sheet_name='Sheet1', index=True)
            df2.to_excel(writer, sheet_name='Sheet2', index=True)

    def plot_main_graph(self, axs):
        '''plot an end-to-end time analysis chart.'''
        data_dict = {}
        start_flag = False
        for key in self.op_analy_result.keys():
            if "total" in key:
                continue
            if "----->parallel summary:" in key:
                start_flag = True
                continue
            if start_flag:
                data_dict[key] = self.op_analy_result[key]
        name_labels = []
        for name, metrics in data_dict.items():
            name_labels.append(name)
            total_time_ratio = float(metrics[RATIO_OF_TOTAL_TIME])
            if RECOMPUTE_RATIO not in metrics:
                recompute_ratio = 0
            else:
                recompute_ratio = float(metrics[RECOMPUTE_RATIO])
            recompute_time = round(total_time_ratio * recompute_ratio, 2)
            total_bar = axs[0,0].bar(name, total_time_ratio, color='#B03A2E',
                               label='Time Ratio' if name == list(data_dict.keys())[0] else "")
            recompute_bar = axs[0,0].bar(name, recompute_time, color='#DDA15E',
                                   label='Recompute Ratio' if name == list(data_dict.keys())[0] else "")
            # the number '#B03A2E' means dark red color
            axs[0,0].text(name, total_time_ratio + (total_time_ratio)/40, f'{total_time_ratio:.2f}', ha='center', color='#B03A2E')

            if recompute_time > 1e-8:
                # the number '#DDA15E' means dark yellow color
                axs[0,0].text(name, recompute_time + (recompute_time)/8, f'{recompute_time:.2f}', ha='center', color='#DDA15E')
        axs[0, 0].set_xticklabels(labels=name_labels, rotation=45, ha='right')
        axs[0,0].legend()
        axs[0,0].set_ylabel('Ratio')
        axs[0,0].set_title('Time Ratio vs Recompute Ratio')

    def plot_overlap_graph(self, axs):
        '''Plot the overlap situation diagram.'''
        data_dict = {}
        for key in self.op_analy_result.keys():
            if "ep" in key or "tp" in key or "dp" in key or "pp" in key:
                data_dict[key] = self.op_analy_result[key]
        name_labels = []
        for name, metrics in data_dict.items():
            name_labels.append(name)
            total_time = float(metrics[TOTAL_TIME])
            overlap_time = float(metrics[OVERLAP_TIME])
            overlap_ratio = float(metrics[OVERLAP_RATIO])

            total_bar = axs[0,1].bar(name, total_time, color='#B03A2E',
                               label='Total Time' if name == list(data_dict.keys())[0] else "")
            overlap_bar = axs[0,1].bar(name, overlap_time, color='#DDA15E',
                                   label='Overlap Time' if name == list(data_dict.keys())[0] else "")
            axs[0,1].text(name, overlap_time + (overlap_time)/10, f'overlap_ratio: {overlap_ratio:.2f}%', ha='center', color='#DDA15E')
        axs[0,1].set_xticklabels(labels=name_labels) #, rotation=45, ha='right')
        axs[0,1].legend()
        axs[0,1].set_ylabel('Time(ms)')
        axs[0,1].set_title('Total time vs Overlap time')

    def plot_aicmacratio_graph(self, axs):
        '''Plot the aic_mac_ratio diagram.'''
        fa_aicmatio = []
        cube_aicmatio = []
        vector_time = []
        for data in self.op_dict:
            if self.aic_mac_ratio:
                if "fa" in data["op_complex_type"]:
                    fa_aicmatio.append(data["aic_mac_ratio"])
                if "matmul" in data["op_complex_type"]:
                    cube_aicmatio.append(data["aic_mac_ratio"])
            if "vector" in data["op_complex_type"]:
                vector_time.append(float(data["Duration(us)"]/US_TO_MS))

        sns.histplot(fa_aicmatio, kde=True, label="fa aic mac ratio", color="blue", alpha=0.3, ax=axs[1,0])
        sns.histplot(cube_aicmatio, kde=True, label="cube aic mac ratio", color="red", alpha=0.3, ax=axs[1,0])

        axs[1,0].set_xlabel("Value")
        axs[1,0].set_ylabel("Density/Frequency")
        axs[1,0].set_title("Histogram and KDE of the FA and Cube operators")
        axs[1,0].legend()
        axs[1,0].grid(True)
        counts, bins, _ = axs[1,1].hist(
            vector_time,
            bins=50,
            color='lightgreen',
            edgecolor='black',
            alpha=0.5,
            label='Histogram'
        )
        bin_centers = (bins[:-1] + bins[1:]) / 2
        axs[1,1].plot(bin_centers, counts, 'b-', linewidth=2, marker='o', markersize=5, label='Line Plot')
        axs[1,1].set_xlabel('time(ms)')
        axs[1,1].set_ylabel('number')
        axs[1,1].set_title('Histogram of Vector Operator Execution Times')
        axs[1,1].legend()
        axs[1,1].grid(True, linestyle='--', alpha=0.5)

    def plot_figure(self, axs):
        '''The entry method of plot.'''
        self.plot_main_graph(axs)
        self.plot_overlap_graph(axs)
        self.plot_aicmacratio_graph(axs)

    def get_op_analy_result(self):
        data_dict = {}
        start_flag = False
        for key in self.op_analy_result.keys():
            if "----->parallel summary:" in key:
                start_flag = True
                continue
            if start_flag:
                data_dict[key] = self.op_analy_result[key]
        return data_dict


def get_op_analy_result(kernel_details, ir_file_path="", save_path="./output.xlsx", title="DeepInsight"):
    analyzer = Overlap_Analyzer(kernel_details, save_path=save_path)
    overlap_statistic = analyzer.analyse(title=title)
    analyzer.analy_recompute(ir_file_path)
    analyzer.analy_parallel()
    analyzer.us_to_ms()
    analyzer.save_result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_details", type=str, required=True, help="kernel_details path")
    parser.add_argument("--ir_file_path", type=str, required=True, help="ir floder path")
    parser.add_argument("--title", type=str, required=True, help="Identifier of model (e.g. mixtral_130b_128k)")
    parser.add_argument("--save_path", type=str, default="./", help="save path")
    args = parser.parse_args()

    get_op_analy_result(kernel_details = args.kernel_details,
                        ir_file_path = args.ir_file_path,
                        save_path = args.save_path,
                        title=args.title)
