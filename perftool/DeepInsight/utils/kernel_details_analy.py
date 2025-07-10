import re
import gc
import csv
import pandas as pd
from decimal import Decimal
from utils.static_value import *

class Kernel_Analyzer:
    def __init__(self):
        self.overlap_priority_dict = {}
        for i in range(len(OVERLAP_PRIORITY_LIST)):
            self.overlap_priority_dict[OVERLAP_PRIORITY_LIST[i]] = i

        self.op_list = None
        self.op_type_list = None

    def to_decimal(self, x):
        return Decimal(x).quantize(Decimal('0.000'))

    def extractLayerIDFromName(self, name):
        name_split = name.split('/')
        for split in name_split:
            pattern = '^(\d)-'
            match = re.search(pattern, split)
            if match:
                extracted_id = match.group(1)
                return extracted_id
        return ""

    def extractGradientFromName(self, name):
        if name.startswith("Kernel::KernelLaunch::Gradients") or name.startswith("Gradients"):
            return 1
        else:
            return 0

    def extractRecomputeFromName(self, name):
        if name.startswith("Kernel::KernelLaunch::recompute_Default") or name.startswith("recompute_Default"):
            return 1
        else:
            return 0

    def analy_kernel_details(self, kernel_details_file):
        details_df = pd.read_csv(kernel_details_file, converters={"Start Time(us)": self.to_decimal, "Duration(us)": self.to_decimal})

        details_df = details_df.reindex(columns=LIST_OF_ORIGIN_OP_DETAILS).fillna('')
        details_df = details_df.reset_index(drop=True)

        for i in range(len(LIST_OF_ORIGIN_OP_DETAILS)):
            details_df.rename(columns={LIST_OF_ORIGIN_OP_DETAILS[i]: LIST_OF_KERNEL_OP_DETAILS[i]}, inplace=True)

        details_df[DI_OP_SHORT_NAME] = details_df[KERNEL_OP_FULL_NAME].str.split('/').str[-1].str.lower()
        details_df = details_df[~(details_df[DI_OP_SHORT_NAME].str.contains("aicpu") & ~details_df[KERNEL_OP_FULL_NAME].str.contains("Pynative"))]

        details_df[DI_END_TIME] = details_df[KERNEL_START_TIME] + details_df[KERNEL_DURATION]
        details_df[DI_OVERLAP_TIME] = Decimal(0)
        details_df[DI_NOT_OVERLAP_TIME] = details_df[KERNEL_DURATION]

        def getLayerId(value):
            return self.extractLayerIDFromName(value[KERNEL_OP_FULL_NAME])
        details_df[DI_LAYER_ID_IN_NAME] = details_df.apply(getLayerId, axis=1)
        def getGradient(value):
            return self.extractGradientFromName(value[KERNEL_OP_FULL_NAME])
        details_df[DI_GRADIENT_IN_NAME] = details_df.apply(getGradient, axis=1)
        def getRecompute(value):
            return self.extractRecomputeFromName(value[KERNEL_OP_FULL_NAME])
        details_df[DI_RECOMPUTE_IN_NAME] = details_df.apply(getRecompute, axis=1)


        details_df[DI_OP_TYPE] = OP_VEC
        matmul_mask = details_df[DI_OP_SHORT_NAME].str.contains("matmul", case=False) & \
                    ~details_df[DI_OP_SHORT_NAME].str.contains("memset", case=False) & \
                    ~details_df[DI_OP_SHORT_NAME].str.contains("transpose", case=False)
        details_df.loc[matmul_mask, DI_OP_TYPE] = OP_MATMUL

        gmm_mask = details_df[DI_OP_SHORT_NAME].str.contains("groupedmatmul", case=False) & \
                    ~details_df[DI_OP_SHORT_NAME].str.contains("memset", case=False)
        details_df.loc[gmm_mask, DI_OP_TYPE] = OP_GMM

        gmm_fusion_mask = details_df[DI_OP_SHORT_NAME].str.contains("gmm", case=False) & \
                    ~details_df[DI_OP_SHORT_NAME].str.contains("memset", case=False)
        details_df.loc[gmm_fusion_mask, DI_OP_TYPE] = OP_GMM

        bmm_mask = details_df[DI_OP_SHORT_NAME].str.contains("batchmatmul", case=False) & \
                    ~details_df[DI_OP_SHORT_NAME].str.contains("memset", case=False)
        details_df.loc[bmm_mask, DI_OP_TYPE] = OP_BMM

        bmm_fusion_mask = details_df[DI_OP_SHORT_NAME].str.contains("bmm", case=False) & \
                    ~details_df[DI_OP_SHORT_NAME].str.contains("memset", case=False)
        details_df.loc[bmm_fusion_mask, DI_OP_TYPE] = OP_BMM

        conv2d_fusion_mask = details_df[DI_OP_SHORT_NAME].str.contains("conv2d", case=False) & \
                    ~details_df[DI_OP_SHORT_NAME].str.contains("memset", case=False)
        details_df.loc[conv2d_fusion_mask, DI_OP_TYPE] = OP_CONV2D

        fa_mask = details_df[DI_OP_SHORT_NAME].str.contains("flashattentionscore", case=False)
        details_df.loc[fa_mask, DI_OP_TYPE] = OP_FA

        for comm_op_type in COMM_OP_LIST:
            comm_mask = details_df[DI_OP_SHORT_NAME].str.contains(comm_op_type, case=False) & \
                    details_df[DI_OP_SHORT_NAME].str.contains("hcom_", case=False)
            details_df.loc[comm_mask, DI_OP_TYPE] = comm_op_type

        self.op_type_list = details_df[DI_OP_TYPE].drop_duplicates().tolist()
        def get_priority(data):
            for key in self.overlap_priority_dict.keys():
                if key in data:
                    return self.overlap_priority_dict[key]
            return self.overlap_priority_dict[OP_OTHER]
        self.op_type_list.sort(key=lambda x: get_priority(x))

        self.op_list = details_df.to_dict(orient='records')
        self.op_list.sort(key=lambda x: (x[KERNEL_START_TIME], x[DI_END_TIME]))

        details_df = None
        gc.collect()

    def contains_two_defaults(self, s, prefix_default):
        if s.count(prefix_default) > 2:
            print("Count Default > 2: ", s)
        return s.count(prefix_default) == 2

    def prefix_split(self, ops_name, prefix):
        if prefix in ops_name:
            ops_name = ops_name[len(prefix):]
        return ops_name

    def prefix_default_add_value(self, name, prefix, prefix_default):
        ops_name = name
        if "/" in ops_name:
            ops_name = ops_name.rsplit("/", 1)[0]
            if ops_name.endswith(prefix_default):
                ops_name = ops_name.rsplit("/", 1)[0]
        if "Default" == ops_name:
            return name
        ops_name = self.prefix_split(ops_name, prefix)
        return ops_name

    def updateCommTypeAndShape(self, index, item, ops_name, ir_op_dict):
        if item == IR_COMM_INPUT_TYPE and ir_op_dict[ops_name][item] != "" and self.op_list[index][KERNEL_INPUT_TYPE] == "":
            self.op_list[index][KERNEL_INPUT_TYPE] = ir_op_dict[ops_name][item]
        if item == IR_COMM_INPUT_SHAPE and ir_op_dict[ops_name][item] != "" and self.op_list[index][KERNEL_INPUT_SHAPE] == "":
            self.op_list[index][KERNEL_INPUT_SHAPE] = ir_op_dict[ops_name][item]
        if item == IR_COMM_OUTPUT_TYPE and ir_op_dict[ops_name][item] != "" and self.op_list[index][KERNEL_OUTPUT_TYPE] == "":
            self.op_list[index][KERNEL_OUTPUT_TYPE] = ir_op_dict[ops_name][item]
        if item == IR_COMM_OUTPUT_SHAPE and ir_op_dict[ops_name][item] != "" and self.op_list[index][KERNEL_OUTPUT_SHAPE] == "":
            self.op_list[index][KERNEL_OUTPUT_SHAPE] = ir_op_dict[ops_name][item]

    def combine_ir_into_kernel(self, ir_op_dict):
        prefix = "Kernel::KernelLaunch::"
        iter_index = 0
        prefix_default = "Default"
        op_flag = "-op"
        op_flag_move_value = {}

        kernel_key_map = {}
        for index, row in enumerate(self.op_list):
            ops_name = row[KERNEL_OP_FULL_NAME]
            if prefix in ops_name:
                iter_index = index
            prefix_default_ok = False
            if ops_name.startswith(prefix_default):
                ops_name_temp = ops_name.split("Default/", 1)[1]
                iter_index_temp = iter_index
                if ops_name_temp in self.op_list[iter_index_temp][KERNEL_OP_FULL_NAME]:
                    kernel_key_map[ops_name] = kernel_key_map[self.op_list[iter_index_temp][KERNEL_OP_FULL_NAME]]
                else:
                    kernel_key_map[ops_name] = self.prefix_default_add_value(ops_name, prefix, prefix_default)
                prefix_default_ok = True

            prefix_default_two_ok = False
            if self.contains_two_defaults(ops_name, prefix_default) and (not prefix_default_ok):
                kernel_key_map[ops_name] = self.prefix_default_add_value(ops_name, prefix, prefix_default)
                prefix_default_two_ok = True

            op_flag_ok = False
            if op_flag in ops_name and (not prefix_default_ok) and (not prefix_default_two_ok):
                ops_name_temp_first = ops_name.rsplit(op_flag, 1)[0]
                ops_name_temp_scend = ops_name.rsplit(op_flag, 1)[1]
                ops_name_temp_scend = ops_name_temp_scend if "/" not in ops_name_temp_scend else ops_name_temp_scend.split("/", 1)[0]
                ops_name_temp = ops_name_temp_first + op_flag + ops_name_temp_scend
                ops_name_temp = self.prefix_split(ops_name_temp, prefix)
                kernel_key_map[ops_name] = ops_name_temp
                op_flag_ok = True
            if not op_flag_ok and not prefix_default_ok and not prefix_default_two_ok:
                print(ops_name)
                raise Exception("Ops name detect error!")
            
            if op_flag in ops_name:
                ops_name_temp = kernel_key_map[ops_name]
                ops_name_temp_first = ops_name_temp.rsplit(op_flag, 1)[0]
                ops_name_temp_scend = ops_name_temp.rsplit(op_flag, 1)[1]
                ops_name_temp_scend = ops_name_temp_scend if "/" not in ops_name_temp_scend else ops_name_temp_scend.split("/", 1)[0]
                if ops_name_temp_first not in op_flag_move_value:
                    op_flag_move_value[ops_name_temp_first] = int(ops_name_temp_scend)
                else:
                    op_flag_move_value[ops_name_temp_first] = min(int(ops_name_temp_scend), op_flag_move_value[ops_name_temp_first])
        
        for key, value in kernel_key_map.items():
            ops_name_temp_first = value.rsplit(op_flag, 1)[0]
            if ops_name_temp_first not in op_flag_move_value:
                continue
            ops_name_temp_scend = value.rsplit(op_flag, 1)[1]
            ops_name_temp_scend = ops_name_temp_scend if "/" not in ops_name_temp_scend else ops_name_temp_scend.split("/", 1)[0]
            ops_name_temp_scend = (int)(ops_name_temp_scend) - op_flag_move_value[ops_name_temp_first]
            new_name = ops_name_temp_first + op_flag + str(ops_name_temp_scend)
            kernel_key_map[key] = new_name

        with open('ir_op_dict.csv', 'w', newline='', encoding='utf-8') as f:
            all_fields = set()
            for sub_dict in ir_op_dict.values():
                all_fields.update(sub_dict.keys())
            
            fieldnames = ['Name'] + sorted(all_fields)

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for name, sub_dict in ir_op_dict.items():
                row = {'Name': name}
                row.update(sub_dict)
                writer.writerow(row)

        with open('kernel_key_map.csv', 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['Key', 'Value']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for key, value in kernel_key_map.items():
                writer.writerow({'Key': key, 'Value': value})

        kernel_op_set = set(list(kernel_key_map.values()))
        ir_op_set = set(list(ir_op_dict.keys()))

        diff_kernel_only = kernel_op_set - ir_op_set
        diff_ir_only = ir_op_set - kernel_op_set

        with open('diff_kernel_only.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['only_in_kernel'])
            for item in diff_kernel_only:
                writer.writerow([item])

        with open('diff_ir_only.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['only_in_ir'])
            for item in diff_ir_only:
                writer.writerow([item])

        for index, row in enumerate(self.op_list):
            ops_name = kernel_key_map[row[KERNEL_OP_FULL_NAME]]
            if ops_name in ir_op_dict:
                self.op_list[index]["di_is_kernel_op_in_ir"] = 1
                for item in IR_ITEM_LIST:
                    if item in IR_COMM_TYPE_SHAPE_LIST:
                        self.updateCommTypeAndShape(index, item, ops_name, ir_op_dict)
                    else:
                        self.op_list[index][item] = ir_op_dict[ops_name][item]
            else:
                self.op_list[index]["di_is_kernel_op_in_ir"] = 0
                for item in IR_ITEM_LIST:
                    self.op_list[index][item] = ""
