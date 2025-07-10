import pandas as pd
from decimal import Decimal
import seaborn as sns
import matplotlib.pyplot as plt
import time

from utils.ir_graph_analy import IR_Analyzer
from utils.kernel_details_analy import Kernel_Analyzer
from utils.static_value import *

def is_overlapping(interval1, interval2):
    return (interval1[0] >= interval2[0] and interval1[0] < interval2[1]) or \
           (interval1[0] <= interval2[0] and interval1[1] > interval2[0])

def find_overlapping_intervals(list_of_intervals, target_interval):
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

class Analyzer:
    def __init__(self, kernel_details_file, parallel_config = None, ir_file_path="", save_path="./"):
        self.path = save_path

        self.kernel_analyzer = Kernel_Analyzer()
        self.kernel_analyzer.analy_kernel_details(kernel_details_file)

        self.ir_analyzer = IR_Analyzer()
        self.ir_analyzer.analy_ir_file(ir_file_path)

        self.op_type_priority = {key: value for value, key in enumerate(self.kernel_analyzer.op_type_list)}

        self.free_time = Decimal(0)
        self.total_step_time = self.get_total_step_time()

        self.di_op_summary = {
            item: {DI_TOTAL_TIME: self.to_decimal(0), DI_OVERLAP_TIME: self.to_decimal(0), DI_OVERLAP_RATIO: 0, DI_NOT_OVERLAP_TIME: self.to_decimal(0), DI_NON_OVERLAP_RATIO:0, DI_OP_COUNT: 0, DI_RECOMPUTE_COUNT: 0, DI_RECOMPUTE_TIME: 0, DI_RECOMPUTE_OVERLAP_TIME: self.to_decimal(0), DI_RECOMPUTE_NOT_OVERLAP_TIME: self.to_decimal(0)} for item in self.kernel_analyzer.op_type_list}
        
        self.top10_vec = None

        if parallel_config == None:
            self.parallel_config = {PARALLEL_TP: None, PARALLEL_PP: None, PARALLEL_EP: None, PARALLEL_CP: None, PARALLEL_DP: None}
        else:
            self.parallel_config = parallel_config

    def to_decimal(self, x):
        return Decimal(x).quantize(Decimal('0.000'))

    def get_comm_op_type(self, op_name):
        for comm_type in COMM_OP_LIST:
            if comm_type in op_name:
                return comm_type
        return None

    def get_total_step_time(self):
        return self.kernel_analyzer.op_list[-1][DI_END_TIME] - self.kernel_analyzer.op_list[0][KERNEL_START_TIME]
    
    def process_overlap_time(self, overlap_start_time, overlap_end_time, op_type,
                             index, overlap_time_pipeline):
        is_valid_overlap_time = overlap_end_time > overlap_start_time
        if is_valid_overlap_time:
            overlap_time_pipeline.append({DI_OVERLAP_TIME: [overlap_start_time, overlap_end_time],
                                          DI_OP_TYPE: op_type,
                                          "index": index})

    def process_overlap_time_pipeline(self, overlap_time_pipeline):
        if len(overlap_time_pipeline) == 0:
            return
        overlap_time_pipeline.sort(key=lambda x: (self.op_type_priority[x[DI_OP_TYPE]],
                                                  x[DI_OVERLAP_TIME][0],
                                                  x[DI_OVERLAP_TIME][1]))

        def merge_intervals(intervals):
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
            s1 = overlap_time_pipeline[item_index][DI_OVERLAP_TIME][0]
            e1 = overlap_time_pipeline[item_index][DI_OVERLAP_TIME][1]
            overlapping_interval_index = find_overlapping_intervals(overlap_time_merge,
                                                                    overlap_time_pipeline[item_index][DI_OVERLAP_TIME])
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
                    self.kernel_analyzer.op_list[overlap_time_pipeline[item_index]["index"]][DI_OVERLAP_TIME] += e2 - s2 - interval_time
                    self.kernel_analyzer.op_list[overlap_time_pipeline[item_index]["index"]][
                        DI_NOT_OVERLAP_TIME] -= e2 - s2 - interval_time
                    self.di_op_summary[overlap_time_pipeline[item_index][DI_OP_TYPE]][
                        DI_OVERLAP_TIME] += e2 - s2 - interval_time
                elif s2 < s1 and e2 <= e1:
                    # case2
                    self.kernel_analyzer.op_list[overlap_time_pipeline[item_index]["index"]][DI_OVERLAP_TIME] += e2 - s1 - interval_time
                    self.kernel_analyzer.op_list[overlap_time_pipeline[item_index]["index"]][
                        DI_NOT_OVERLAP_TIME] -= e2 - s1 - interval_time
                    self.di_op_summary[overlap_time_pipeline[item_index][DI_OP_TYPE]][
                        DI_OVERLAP_TIME] += e2 - s1 - interval_time
                elif s2 >= s1 and e2 > e1:
                    # case3
                    self.kernel_analyzer.op_list[overlap_time_pipeline[item_index]["index"]][DI_OVERLAP_TIME] += e1 - s2 - interval_time
                    self.kernel_analyzer.op_list[overlap_time_pipeline[item_index]["index"]][
                        DI_NOT_OVERLAP_TIME] -= e1 - s2 - interval_time
                    self.di_op_summary[overlap_time_pipeline[item_index][DI_OP_TYPE]][
                        DI_OVERLAP_TIME] += e1 - s2 - interval_time
                else:
                    # case4
                    self.kernel_analyzer.op_list[overlap_time_pipeline[item_index]["index"]][DI_OVERLAP_TIME] += e1 - s1 - interval_time
                    self.kernel_analyzer.op_list[overlap_time_pipeline[item_index]["index"]][DI_NOT_OVERLAP_TIME] -= e1 - s1 - interval_time
                    self.di_op_summary[overlap_time_pipeline[item_index][DI_OP_TYPE]][
                        DI_OVERLAP_TIME] += e1 - s1 - interval_time
                overlap_time_merge.append([s1, e1])
            overlap_time_merge = merge_intervals(overlap_time_merge)

    def get_overlap_max_time(self, name, overlap_max_time, s2):
        if name not in overlap_max_time:
            overlap_max_time[name] = 0
            overlap_start_time = s2
        else:
            overlap_start_time = max(s2, overlap_max_time[name])
        return overlap_start_time

    def update_overlap_summary(self):
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

        for op_type in self.kernel_analyzer.op_type_list:
            self.di_op_summary[op_type][DI_NOT_OVERLAP_TIME] = self.di_op_summary[op_type][DI_TOTAL_TIME] - \
                                                                self.di_op_summary[op_type][DI_OVERLAP_TIME]
            total_not_overlap_time += self.di_op_summary[op_type][DI_NOT_OVERLAP_TIME]
            totoal_op_count += self.di_op_summary[op_type][DI_OP_COUNT]
            if op_type in COMM_OP_LIST:
                total_comunication_not_overlap_time += self.di_op_summary[op_type][DI_NOT_OVERLAP_TIME]
                total_comunication_time += self.di_op_summary[op_type][DI_TOTAL_TIME]
                total_comunication_overlap_time += self.di_op_summary[op_type][DI_OVERLAP_TIME]
                total_comunication_count += self.di_op_summary[op_type][DI_OP_COUNT]
            else:
                total_compute_not_overlap_time += self.di_op_summary[op_type][DI_NOT_OVERLAP_TIME]
                total_compute_time += self.di_op_summary[op_type][DI_TOTAL_TIME]
                total_compute_overlap_time += self.di_op_summary[op_type][DI_OVERLAP_TIME]
                total_compute_count += self.di_op_summary[op_type][DI_OP_COUNT]

            self.di_op_summary[op_type][DI_NON_OVERLAP_RATIO] = self.di_op_summary[op_type][DI_NOT_OVERLAP_TIME] / self.total_step_time
            self.di_op_summary[op_type][DI_OVERLAP_RATIO] = self.di_op_summary[op_type][DI_OVERLAP_TIME] / self.di_op_summary[op_type][DI_TOTAL_TIME]
        
        self.free_time = self.total_step_time - total_not_overlap_time

        self.di_op_summary[DI_FREE] = {
            DI_TOTAL_TIME: self.free_time,
            DI_OVERLAP_TIME: 0,
            DI_OVERLAP_RATIO: 0,
            DI_NOT_OVERLAP_TIME: self.free_time,
            DI_NON_OVERLAP_RATIO: self.free_time / self.total_step_time,
            DI_OP_COUNT: 0,
            DI_RECOMPUTE_COUNT: 0,
            DI_RECOMPUTE_OVERLAP_TIME: 0,
            DI_RECOMPUTE_NOT_OVERLAP_TIME: 0,
            DI_RECOMPUTE_TIME: 0,
        }

        self.di_op_summary[DI_TOTAL] = {
            DI_TOTAL_TIME: total_compute_time + total_comunication_time + self.free_time,
            DI_OVERLAP_TIME: total_compute_overlap_time + total_comunication_overlap_time,
            DI_OVERLAP_RATIO: (total_compute_overlap_time + total_comunication_overlap_time) / (total_compute_time + total_comunication_time + self.free_time),
            DI_NOT_OVERLAP_TIME: self.total_step_time,
            DI_NON_OVERLAP_RATIO: 1,
            DI_OP_COUNT: totoal_op_count,
            DI_RECOMPUTE_COUNT: 0,
            DI_RECOMPUTE_OVERLAP_TIME: 0,
            DI_RECOMPUTE_NOT_OVERLAP_TIME: 0,
            DI_RECOMPUTE_TIME: 0,
        }
        return self.di_op_summary

    def analy_overlap(self):
        overlap_max_time = {}
        overlap_time_pipeline = []
        if len(self.kernel_analyzer.op_list) <= 0:
            raise Exception("len(kernel_op_list) <= 0, so overlap time array and total time array is empty!")

        left_value_index = 0
        self.di_op_summary[self.kernel_analyzer.op_list[left_value_index][DI_OP_TYPE]][DI_TOTAL_TIME] += \
            self.kernel_analyzer.op_list[left_value_index][DI_END_TIME] - \
            self.kernel_analyzer.op_list[left_value_index][KERNEL_START_TIME]
        self.di_op_summary[self.kernel_analyzer.op_list[left_value_index][DI_OP_TYPE]][DI_OP_COUNT] += 1
        for index in range(1, len(self.kernel_analyzer.op_list)):
            s1 = self.kernel_analyzer.op_list[left_value_index][KERNEL_START_TIME]
            e1 = self.kernel_analyzer.op_list[left_value_index][DI_END_TIME]
            s2 = self.kernel_analyzer.op_list[index][KERNEL_START_TIME]
            e2 = self.kernel_analyzer.op_list[index][DI_END_TIME]
            op_type_left = self.kernel_analyzer.op_list[left_value_index][DI_OP_TYPE]
            op_type_current = self.kernel_analyzer.op_list[index][DI_OP_TYPE]
            name_left = self.kernel_analyzer.op_list[left_value_index][KERNEL_OP_FULL_NAME]
            name_current = self.kernel_analyzer.op_list[index][KERNEL_OP_FULL_NAME]
            if e1 <= s2:
                left_value_index = index
                self.process_overlap_time_pipeline(overlap_time_pipeline)
                overlap_time_pipeline = []
            elif e1 > s2 and e1 <= e2:
                overlap_start_time = self.get_overlap_max_time(name_left, overlap_max_time, s2)
                overlap_end_time = e1
                self.process_overlap_time(overlap_start_time, overlap_end_time, op_type_left,
                                          left_value_index, overlap_time_pipeline)
                overlap_max_time[name_left] = max(max(s1, e1), overlap_max_time[name_left])

                overlap_start_time = self.get_overlap_max_time(name_current, overlap_max_time, s2)
                overlap_end_time = e1
                self.process_overlap_time(overlap_start_time, overlap_end_time, op_type_current,
                                          index, overlap_time_pipeline)
                overlap_max_time[name_current] = max(max(s2, e1), overlap_max_time[name_current])
                overlap_max_time.pop(name_left)
                left_value_index = index
            elif e1 > s2 and e1 > e2:
                overlap_start_time = self.get_overlap_max_time(name_left, overlap_max_time, s2)
                overlap_end_time = e2
                self.process_overlap_time(overlap_start_time, overlap_end_time, op_type_left,
                                          left_value_index, overlap_time_pipeline)
                overlap_max_time[name_left] = max(max(s1, e2), overlap_max_time[name_left])

                overlap_start_time = self.get_overlap_max_time(name_current, overlap_max_time, s2)
                overlap_end_time = e2
                self.process_overlap_time(overlap_start_time, overlap_end_time, op_type_current,
                                          index, overlap_time_pipeline)
                overlap_max_time[name_current] = max(max(s2, e2), overlap_max_time[name_current])
                overlap_max_time.pop(name_current)

            if len(overlap_time_pipeline) > 300:
                overlap_time_pipeline_new = self.split_temp_data(overlap_time_pipeline, e2, s2)
                self.process_overlap_time_pipeline(overlap_time_pipeline_new)
                overlap_time_pipeline_new = []
                overlap_time_pipeline = self.clear_temp_data(overlap_time_pipeline, e2, s2)
            self.di_op_summary[op_type_current][DI_TOTAL_TIME] += e2 - s2
            self.di_op_summary[op_type_current][DI_OP_COUNT] += 1
        self.process_overlap_time_pipeline(overlap_time_pipeline)

        self.update_overlap_summary()        

    def clear_temp_data(self, overlap_time_pipeline, e2, s2):
        for i in range(len(overlap_time_pipeline) - 1, -1, -1):
            if overlap_time_pipeline[i][DI_OVERLAP_TIME][1] < s2:
                del overlap_time_pipeline[i]
        return overlap_time_pipeline

    def split_temp_data(self, overlap_time_pipeline, e2, s2):
        import copy
        overlap_time_pipeline_new = copy.deepcopy(overlap_time_pipeline)
        for i in range(len(overlap_time_pipeline)):
            if overlap_time_pipeline[i][DI_OVERLAP_TIME][1] <= s2:
                continue
            if overlap_time_pipeline[i][DI_OVERLAP_TIME][0] >= s2:
                continue
            overlap_time_pipeline_new[i][DI_OVERLAP_TIME][1] = s2
            overlap_time_pipeline[i][DI_OVERLAP_TIME][0] = s2
        return overlap_time_pipeline_new

    def update_recompute_summary(self):
        if self.ir_analyzer.ir_op_dict is None:
            return
        
        self.kernel_analyzer.combine_ir_into_kernel(self.ir_analyzer.ir_op_dict)

        for op_info in self.kernel_analyzer.op_list:
            op_type = op_info[DI_OP_TYPE]
            if op_info[DI_RECOMPUTE] == 1:
                self.di_op_summary[op_type][DI_RECOMPUTE_COUNT] += 1
                self.di_op_summary[op_type][DI_RECOMPUTE_OVERLAP_TIME] += op_info[DI_OVERLAP_TIME]
                self.di_op_summary[op_type][DI_RECOMPUTE_NOT_OVERLAP_TIME] += op_info[DI_NOT_OVERLAP_TIME]
                self.di_op_summary[op_type][DI_RECOMPUTE_TIME] += op_info[KERNEL_DURATION]
                self.di_op_summary[op_type][DI_RECOMPUTE_RATIO] = \
                    self.di_op_summary[op_type][DI_RECOMPUTE_NOT_OVERLAP_TIME] / self.di_op_summary[op_type][DI_NOT_OVERLAP_TIME]

        for op_type in self.kernel_analyzer.op_type_list:
            self.di_op_summary[DI_TOTAL][DI_RECOMPUTE_COUNT] += self.di_op_summary[op_type][
                DI_RECOMPUTE_COUNT]
            self.di_op_summary[DI_TOTAL][DI_RECOMPUTE_OVERLAP_TIME] += self.di_op_summary[op_type][
                DI_RECOMPUTE_OVERLAP_TIME]
            self.di_op_summary[DI_TOTAL][DI_RECOMPUTE_NOT_OVERLAP_TIME] += \
                self.di_op_summary[op_type][DI_RECOMPUTE_NOT_OVERLAP_TIME]
            self.di_op_summary[DI_TOTAL][DI_RECOMPUTE_TIME] += \
                self.di_op_summary[op_type][DI_RECOMPUTE_TIME]
            self.di_op_summary[DI_TOTAL][DI_RECOMPUTE_RATIO] = \
                self.di_op_summary[op_type][DI_RECOMPUTE_NOT_OVERLAP_TIME] / self.di_op_summary[op_type][DI_NOT_OVERLAP_TIME]

    def get_coords(self, k, parallel_order, sizes):
        coords = {}
        remainder = k

        for i, name in enumerate(parallel_order):
            if i < len(parallel_order) - 1:
                next_sizes = parallel_order[i+1:]
                weight = 1
                for n in next_sizes:
                    weight *= sizes[n]
                index = remainder // weight
                coords[name] = index % sizes[name]
                remainder = remainder % weight
            else:
                coords[name] = remainder % sizes[name]

        return coords


    def classify_communication_domain(self, device_ids, parallel_order, sizes):
        coords = [self.get_coords(k, parallel_order, sizes) for k in device_ids]

        varying_dims = set()

        for dim in parallel_order:
            dim_values = []
            for d in coords:
                dim_values.append(d[dim])
            if len(set(dim_values)) > 1:
                varying_dims.add(dim)

        if not varying_dims:
            return PARALLEL_SINGE
        elif len(varying_dims) == 1:
            return list(varying_dims)[0]
        else:
            return ""

    def process_ratio(self):
        for key in self.di_op_summary.keys():
            if self.di_op_summary[key][DI_TOTAL_TIME] > 0:
                self.di_op_summary[key][DI_OVERLAP_RATIO] = self.di_op_summary[key][DI_OVERLAP_TIME] / self.di_op_summary[key][DI_TOTAL_TIME]
            self.di_op_summary[key][DI_NON_OVERLAP_RATIO] = self.di_op_summary[key][DI_NOT_OVERLAP_TIME] / self.total_step_time
            if self.di_op_summary[key][DI_NOT_OVERLAP_TIME] > 0:
                self.di_op_summary[key][DI_RECOMPUTE_RATIO] = self.di_op_summary[key][DI_RECOMPUTE_NOT_OVERLAP_TIME] / self.di_op_summary[key][DI_NOT_OVERLAP_TIME]

    def is_input_parallel_config(self):
        return self.parallel_config[PARALLEL_CP] is None or self.parallel_config[PARALLEL_TP] is None or \
               self.parallel_config[PARALLEL_EP] is None or self.parallel_config[PARALLEL_PP] is None or \
               self.parallel_config[PARALLEL_DP] is None

    def analy_parallel(self):
        import copy
        if self.ir_analyzer.ir_op_dict is None or self.is_input_parallel_config():
            for key in PARALLEL_MAPPING.keys():
                if key in self.di_op_summary.keys():
                    if PARALLEL_MAPPING[key] not in self.di_op_summary:
                        self.di_op_summary[PARALLEL_MAPPING[key]] = copy.deepcopy(self.di_op_summary[key])
                    else:
                        for key_data in self.di_op_summary[PARALLEL_MAPPING[key]]:
                            self.di_op_summary[PARALLEL_MAPPING[key]][key_data] += self.di_op_summary[key][key_data]
            self.process_ratio()
            return
        for index in range(len(self.kernel_analyzer.op_list)):
            group_rank_ids = self.kernel_analyzer.op_list[index][IR_GROUP_RANK_IDS]
            if group_rank_ids:
                group_rank_ids = group_rank_ids.strip("\"")
                if ',' not in group_rank_ids:
                    group_rank_ids = group_rank_ids[:-1] + ',' + group_rank_ids[-1]

                group_rank_ids=eval(group_rank_ids)

                parallel_order = [PARALLEL_TP, PARALLEL_CP, PARALLEL_DP, PARALLEL_PP]
                parallel_order.reverse()
                sizes = {PARALLEL_TP: self.parallel_config[PARALLEL_TP], PARALLEL_CP: self.parallel_config[PARALLEL_CP],
                         PARALLEL_DP: self.parallel_config[PARALLEL_DP], PARALLEL_PP: self.parallel_config[PARALLEL_PP]}
                parallel_type = self.classify_communication_domain(group_rank_ids, parallel_order, sizes)
                self.kernel_analyzer.op_list[index][DI_PARALLEL_TYPE] = parallel_type
            else:
                self.kernel_analyzer.op_list[index][DI_PARALLEL_TYPE] = ""

        for key in PARALLEL_MAPPING.keys():
            if key.lower() not in [COMM_AG, COMM_RS] and key.lower() in self.di_op_summary.keys():
                if PARALLEL_MAPPING[key] not in self.di_op_summary:
                    self.di_op_summary[PARALLEL_MAPPING[key]] = copy.deepcopy(self.di_op_summary[key])
                else:
                    for key_data in self.di_op_summary[PARALLEL_MAPPING[key]]:
                        self.di_op_summary[PARALLEL_MAPPING[key]][key_data] += self.di_op_summary[key][key_data]
            self.process_ratio()
        
        accurate_key_list = [PARALLEL_TP, PARALLEL_CP]
        for op_type in accurate_key_list:
            if op_type not in self.di_op_summary:
                self.di_op_summary[op_type] =  {DI_TOTAL_TIME: self.to_decimal(0), DI_OVERLAP_TIME: self.to_decimal(0), DI_OVERLAP_RATIO: 0, DI_NOT_OVERLAP_TIME: self.to_decimal(0), DI_NON_OVERLAP_RATIO:0, DI_OP_COUNT: 0, DI_RECOMPUTE_COUNT: 0, DI_RECOMPUTE_TIME: 0, DI_RECOMPUTE_OVERLAP_TIME: self.to_decimal(0), DI_RECOMPUTE_NOT_OVERLAP_TIME: self.to_decimal(0)}
            for index in range(len(self.kernel_analyzer.op_list)):
                comm_type = self.kernel_analyzer.op_list[index][DI_OP_TYPE]
                if comm_type.lower() not in [COMM_AG, COMM_RS]:
                    continue
                parallel_type = self.kernel_analyzer.op_list[index][DI_PARALLEL_TYPE]
                if parallel_type == op_type:
                    self.di_op_summary[op_type][DI_TOTAL_TIME] += self.kernel_analyzer.op_list[index][DI_END_TIME] - \
            self.kernel_analyzer.op_list[index][KERNEL_START_TIME]
                    self.di_op_summary[op_type][DI_OVERLAP_TIME] += self.kernel_analyzer.op_list[index][DI_OVERLAP_TIME]
                    self.di_op_summary[op_type][DI_NOT_OVERLAP_TIME] = self.di_op_summary[op_type][DI_TOTAL_TIME] - \
                                                                self.di_op_summary[op_type][DI_OVERLAP_TIME]
                    self.di_op_summary[op_type][DI_OP_COUNT] += 1
                    if self.kernel_analyzer.op_list[index][DI_RECOMPUTE] == 1:
                        self.di_op_summary[op_type][DI_RECOMPUTE_COUNT] += 1
                        self.di_op_summary[op_type][DI_RECOMPUTE_TIME] += self.kernel_analyzer.op_list[index][KERNEL_DURATION]
                        self.di_op_summary[op_type][DI_RECOMPUTE_OVERLAP_TIME] += self.kernel_analyzer.op_list[index][DI_OVERLAP_TIME]
                        self.di_op_summary[op_type][DI_RECOMPUTE_NOT_OVERLAP_TIME] += self.kernel_analyzer.op_list[index][DI_NOT_OVERLAP_TIME]

            self.di_op_summary[op_type][DI_NON_OVERLAP_RATIO] = self.di_op_summary[op_type][DI_NOT_OVERLAP_TIME] / self.total_step_time
            if self.di_op_summary[op_type][DI_TOTAL_TIME] > 0:
                self.di_op_summary[op_type][DI_OVERLAP_RATIO] = self.di_op_summary[op_type][DI_OVERLAP_TIME] / self.di_op_summary[op_type][DI_TOTAL_TIME]
        
        op_type = PARALLEL_DP
        if op_type not in self.di_op_summary:
            self.di_op_summary[op_type] =  {DI_TOTAL_TIME: self.to_decimal(0), DI_OVERLAP_TIME: self.to_decimal(0), DI_OVERLAP_RATIO: 0, DI_NOT_OVERLAP_TIME: self.to_decimal(0), DI_NON_OVERLAP_RATIO:0, DI_OP_COUNT: 0, DI_RECOMPUTE_COUNT: 0, DI_RECOMPUTE_TIME: 0, DI_RECOMPUTE_OVERLAP_TIME: self.to_decimal(0), DI_RECOMPUTE_NOT_OVERLAP_TIME: self.to_decimal(0)}
        for index in range(len(self.kernel_analyzer.op_list)):
            parallel_type = self.kernel_analyzer.op_list[index][DI_PARALLEL_TYPE]
            comm_type = self.kernel_analyzer.op_list[index][DI_OP_TYPE]
            if COMM_AG in comm_type.lower() or COMM_RS in comm_type.lower():
                if parallel_type != PARALLEL_CP and parallel_type != PARALLEL_TP:
                    self.di_op_summary[op_type][DI_TOTAL_TIME] += self.kernel_analyzer.op_list[index][DI_END_TIME] - \
            self.kernel_analyzer.op_list[index][KERNEL_START_TIME]
                    self.di_op_summary[op_type][DI_OVERLAP_TIME] += self.kernel_analyzer.op_list[index][DI_OVERLAP_TIME]
                    self.di_op_summary[op_type][DI_NOT_OVERLAP_TIME] = self.di_op_summary[op_type][DI_TOTAL_TIME] - \
                                                                self.di_op_summary[op_type][DI_OVERLAP_TIME]
                    self.di_op_summary[op_type][DI_OP_COUNT] += 1
                    if self.kernel_analyzer.op_list[index][DI_RECOMPUTE] == 1:
                        self.di_op_summary[op_type][DI_RECOMPUTE_COUNT] += 1
                        self.di_op_summary[op_type][DI_RECOMPUTE_TIME] += self.kernel_analyzer.op_list[index][KERNEL_DURATION]
                        self.di_op_summary[op_type][DI_RECOMPUTE_OVERLAP_TIME] += self.kernel_analyzer.op_list[index][DI_OVERLAP_TIME]
                        self.di_op_summary[op_type][DI_RECOMPUTE_NOT_OVERLAP_TIME] += self.kernel_analyzer.op_list[index][DI_NOT_OVERLAP_TIME]

        self.di_op_summary[op_type][DI_NON_OVERLAP_RATIO] = self.di_op_summary[op_type][DI_NOT_OVERLAP_TIME] / self.total_step_time
        if self.di_op_summary[op_type][DI_TOTAL_TIME] > 0:
            self.di_op_summary[op_type][DI_OVERLAP_RATIO] = self.di_op_summary[op_type][DI_OVERLAP_TIME] / self.di_op_summary[op_type][DI_TOTAL_TIME]

    def analy_vector(self):
        vec_dict = {}
        for index in range(len(self.kernel_analyzer.op_list)):
            di_op_type = self.kernel_analyzer.op_list[index][DI_OP_TYPE]
            kernel_op_type = self.kernel_analyzer.op_list[index][KERNEL_OP_TYPE]
            duration = self.kernel_analyzer.op_list[index][KERNEL_DURATION]
            if di_op_type == OP_VEC:
                if kernel_op_type in vec_dict.keys():
                    vec_dict[kernel_op_type] += duration / US_TO_MS
                else:
                    vec_dict[kernel_op_type] = 0

        self.top10_vec = dict(sorted(vec_dict.items(), key=lambda item: item[1], reverse=True)[:10])
    
    def format_result(self):
        for op_info in self.di_op_summary:
            for result_key in self.di_op_summary[op_info].keys():
                if "ratio" in result_key:
                    self.di_op_summary[op_info][result_key] = round(
                        self.di_op_summary[op_info][result_key], 3)
                elif "_ms" in result_key:
                    self.di_op_summary[op_info][result_key] = round(
                        self.di_op_summary[op_info][result_key] / US_TO_MS, 3)

    def save_result(self):
        df1 = pd.DataFrame(self.di_op_summary).transpose()
        df2 = pd.DataFrame(self.kernel_analyzer.op_list)

        base_path = self.path
        path1 = base_path + "\\di_ir_op_dict.csv"
        path2 = base_path + "\\di_op_details.csv"

        df1.to_csv(path1, index=True)
        df2.to_csv(path2, index=True)

    def plot_main_graph(self, axs):
        plot_data_dict = {}

        plot_key_list = []
        compute_keys = [OP_GMM, OP_BMM, OP_MATMUL, OP_CONV2D, OP_FA, OP_VEC]
        plot_compute_keys = ['GMM', 'BMM', 'MM', 'CONV2D', 'FA', 'Vector']
        for i in range(len(compute_keys)):
            if compute_keys[i] in self.di_op_summary.keys():
                plot_key_list.append(plot_compute_keys[i])
                plot_data_dict[plot_compute_keys[i]] = self.di_op_summary[compute_keys[i]]
        
        if self.ir_analyzer.ir_op_dict is None or self.is_input_parallel_config():
            plot_key_list.extend([PARALLEL_DP, PARALLEL_TCP, PARALLEL_EP, PARALLEL_PP, COMM_BROADCAST])
        else:
            plot_key_list.extend([PARALLEL_DP, PARALLEL_TP, PARALLEL_CP, PARALLEL_EP, PARALLEL_PP, COMM_BROADCAST])
        for key in plot_key_list:
            if key in self.di_op_summary.keys():
                plot_data_dict[key] = self.di_op_summary[key]
        plot_data_dict[DI_FREE] = self.di_op_summary[DI_FREE]
        name_labels = []

        communication_ratio = 0
        compute_ratio = 0
        free_ratio = 0
        for name, metrics in plot_data_dict.items():
            name_labels.append(name)
            total_time_ratio = float(metrics[DI_NON_OVERLAP_RATIO])
            if DI_RECOMPUTE_RATIO not in metrics:
                recompute_ratio = 0
            else:
                recompute_ratio = float(metrics[DI_RECOMPUTE_RATIO])
            recompute_time = round(total_time_ratio * recompute_ratio, 2)
            total_bar = axs[0,0].bar(name, total_time_ratio, color='#B03A2E',
                               label='Time Ratio' if name == list(plot_data_dict.keys())[0] else "") # the number '#B03A2E' means dark red color
            recompute_bar = axs[0,0].bar(name, recompute_time, color='#DDA15E',
                               label='Recompute Ratio' if name == list(plot_data_dict.keys())[0] else "") # the number '#DDA15E' means dark yellow color
            
            axs[0,0].text(name, total_time_ratio + (total_time_ratio)/40, f'{total_time_ratio:.2f}', ha='center', color='#B03A2E') # the number '#B03A2E' means dark red color

            if recompute_time > 1e-8:
                axs[0,0].text(name, recompute_time + (recompute_time)/8, f'{recompute_time:.2f}', ha='center', color='#DDA15E') # the number '#DDA15E' means dark yellow color

            if name in plot_compute_keys:
                compute_ratio += total_time_ratio
            elif name == DI_FREE:
                free_ratio += total_time_ratio
            else:
                communication_ratio += total_time_ratio
        axs[0, 0].set_xticklabels(labels=name_labels, rotation=45, ha='right')
        axs[0,0].legend()
        axs[0,0].set_ylabel('Ratio')
        axs[0,0].set_title(f'E2E Step Time Decomposition\ncompute_ratio: {compute_ratio:.2f}, communication_ratio: {communication_ratio:.2f}, free_ratio: {free_ratio:.2f}')

    def plot_overlap_graph(self, axs):
        plot_data_dict = {}
        for key in self.di_op_summary.keys():
            if key in PARALLEL_LIST:
                plot_data_dict[key] = self.di_op_summary[key]
        name_labels = []
        for name, metrics in plot_data_dict.items():
            name_labels.append(name)
            total_time = float(metrics[DI_TOTAL_TIME])
            overlap_time = float(metrics[DI_OVERLAP_TIME])
            overlap_ratio = float(metrics[DI_OVERLAP_RATIO]) * 100

            total_bar = axs[0,1].bar(name, total_time, color='#B03A2E',
                               label='Total Time' if name == list(plot_data_dict.keys())[0] else "")  # the number '#B03A2E' means dark red color
            overlap_bar = axs[0,1].bar(name, overlap_time, color='#DDA15E',
                                   label='Overlap Time' if name == list(plot_data_dict.keys())[0] else "") # the number '#DDA15E' means dark yellow color
            axs[0,1].text(name, overlap_time + (overlap_time)/10, f'overlap_ratio: {overlap_ratio:.2f}%', ha='center', color='#DDA15E')
        axs[0,1].set_xticklabels(labels=name_labels)
        axs[0,1].legend()
        axs[0,1].set_ylabel('Time(ms)')
        axs[0,1].set_title('Total time vs Overlap time')

    def plot_aicmacratio_graph(self, axs):
        fa_aicmatio = []
        cube_aicmatio = []
        vector_time = []
        for data in self.kernel_analyzer.op_list:
            if "fa" in data[DI_OP_TYPE]:
                fa_aicmatio.append(data[KERNEL_MAC_RATIO])
            if "matmul" in data[DI_OP_TYPE] and data[KERNEL_MAC_RATIO] != '' and data[KERNEL_MAC_RATIO] > 0:
                cube_aicmatio.append(data[KERNEL_MAC_RATIO])
            if "vector" in data[DI_OP_TYPE]:
                vector_time.append(float(data[KERNEL_DURATION]/US_TO_MS))

        sns.histplot(fa_aicmatio, kde=True, label="fa aic mac ratio", color="blue", alpha=0.3, ax=axs[1,0])
        sns.histplot(cube_aicmatio, kde=True, label="cube aic mac ratio", color="red", alpha=0.3, ax=axs[1,0])

        axs[1,0].set_xlabel("Value")
        axs[1,0].set_ylabel("Frequency")
        axs[1,0].set_title("aic-mac-ratio of FA and Cube")
        axs[1,0].legend()
        axs[1,0].grid(True)

        if self.top10_vec:
            vec_name = self.top10_vec.keys()
            vec_time = self.top10_vec.values()
        else:
            vec_name = []
            vec_time = []
        total_bar = axs[1,1].bar(vec_name, vec_time, color='#B03A2E', label='Time')
        axs[1,1].set_ylabel('Time (ms)')
        axs[1,1].set_title('Top10 Vector Operator Time')
        axs[1,1].legend()
        axs[1,1].grid(True, linestyle='--', alpha=0.5)

        axs[1,1].set_xticklabels(vec_name, rotation=30, ha='right', fontsize=6)

    def plot_figure(self, axs):
        self.plot_main_graph(axs)
        self.plot_overlap_graph(axs)
        self.plot_aicmacratio_graph(axs)

    def get_op_analy_result(self):
        plot_data_dict = {}
        plot_key_list = []
        compute_keys = [OP_GMM, OP_BMM, OP_MATMUL, OP_CONV2D, OP_FA, OP_VEC]
        plot_compute_keys = ['GMM', 'BMM', 'MM', 'CONV2D', 'FA', 'Vector']
        for i in range(len(compute_keys)):
            if compute_keys[i] in self.di_op_summary.keys():
                plot_key_list.append(plot_compute_keys[i])
                plot_data_dict[plot_compute_keys[i]] = self.di_op_summary[compute_keys[i]]
        if self.ir_analyzer.ir_op_dict is None:
            plot_key_list.extend([PARALLEL_DP, PARALLEL_TCP, PARALLEL_EP, PARALLEL_PP, COMM_BROADCAST])
        else:
            plot_key_list.extend([PARALLEL_DP, PARALLEL_TP, PARALLEL_CP, PARALLEL_EP, PARALLEL_PP, COMM_BROADCAST])
        for key in plot_key_list:
            if key in self.di_op_summary.keys():
                plot_data_dict[key] = self.di_op_summary[key]
        
        return plot_data_dict

def run_di_analyze(kernel_details, parallel_config = None, ir_file_path="", save_path="./di_result.xlsx"):
    start_time = time.time()
    analyzer = Analyzer(kernel_details, parallel_config, ir_file_path, save_path=save_path)
    print(f"1. Analyzer init time = {time.time() - start_time} s.")
    
    start_time = time.time()
    analyzer.analy_overlap()
    print(f"2. analy_overlap time = {time.time() - start_time} s.")

    start_time = time.time()
    analyzer.update_recompute_summary()
    print(f"3. update_recompute_summary time = {time.time() - start_time} s.")
    
    start_time = time.time()
    analyzer.analy_parallel()
    print(f"4. analy_parallel time = {time.time() - start_time} s.")

    analyzer.analy_vector()

    start_time = time.time()
    analyzer.format_result()
    print(f"5. us_to_ms time = {time.time() - start_time} s.")

    start_time = time.time()
    analyzer.save_result()
    print(f"6. save_result time = {time.time() - start_time} s.")

    fig, axs = plt.subplots(2, 2)
    analyzer.plot_figure(axs)
    plt.tight_layout()
    plt.show()

    analyzer.get_op_analy_result()

if __name__ == "__main__":
    kernel_details = r".\example\kernel_details.csv"
    ir_file_path = r".\example\graph"
    save_path = r".\example"
    parallel_config = {PARALLEL_TP: 4, PARALLEL_PP: 2, PARALLEL_EP: 1, PARALLEL_CP: 1, PARALLEL_DP: 2}
    run_di_analyze(kernel_details = kernel_details,
                   parallel_config = parallel_config,
                   ir_file_path = ir_file_path,
                   save_path = save_path)

