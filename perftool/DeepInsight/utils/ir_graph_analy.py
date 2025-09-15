import re
import os
import glob
from utils.static_value import *

global micro_max_value

class IR_Analyzer:
    def __init__(self):
        
        self.ir_op_dict = None

    def get_ir_file_list(self, ir_file_path_dir):
        files = glob.glob(os.path.join(ir_file_path_dir, 'graph_build*'))
        files.sort()
        return files

    def contains_dp_flag(self, line):
        pattern = r'= (AllGather|ReduceScatter).*parallel_optimizer'
        return bool(re.search(pattern, line))

    def extractOpName(self, line):
        pattern = r"^\s*%\d+[^=]*=\s*([^(]+)\("
        match = re.search(pattern, line)
        if match:
            return str(match.group(1))
        else:
            return None

    def extractInt64Value(self, line, attr_prefix):
        pattern = attr_prefix + r'\s*I64\((\d+)\)'
        match = re.search(pattern, line)
        if match:
            return str(match.group(1))
        else:
            return None

    def extractBoolValue(self, line, attr_prefix):
        pattern = attr_prefix + r'\s*Bool\((\d+)\)'
        match = re.search(pattern, line)
        if match:
            return str(match.group(1))
        else:
            return None

    def extractIdValue(self, line, attr_prefix):
        pattern = attr_prefix + r'\s*"(\d+)"'
        match = re.search(pattern, line)
        if match:
            return str(match.group(1))
        else:
            return None

    def extractTupleValue(self, line, attr_prefix):
        pattern = attr_prefix + r'\s*(\([^)]+\))'
        match = re.search(pattern, line)
        if match:
            extracted_str = match.group(1)
            return '\"' + extracted_str + '\"'
        else:
            return None

    def extractListValue(self, line, attr_prefix):
        pattern = attr_prefix + r'\s*\[([^)]+)\]'
        match = re.search(pattern, line)
        if match:
            extracted_str = match.group(1)
            return extracted_str
        else:
            return None

    def extractStringValue(self, line, attr_prefix):
        pattern = attr_prefix + r'\s*\"([^\"]+)\"'
        match = re.search(pattern, line)
        if match:
            extracted_str = match.group(1)
            return extracted_str
        else:
            return None

    def is_valid_op_info_line(self, line):
        pattern = r"^\s*%\d+"
        op_info_match = re.match(pattern, line)
        return True if op_info_match else False

    def get_ir_line_info(self, line, next_line):
        dp_flag = self.contains_dp_flag(line)
        micro = self.extractInt64Value(line, "micro:")
        unique_id = self.extractIdValue(line, "{unique_id:")
        forward_unique_id = self.extractIdValue(line, "forward_unique_id:")
        duplicated_flag = self.extractBoolValue(line, "duplicated:")
        group_rank_ids = self.extractTupleValue(line, "group_rank_ids:")
        group = self.extractStringValue(line, "group:")
        need_cse_after_recompute = self.extractBoolValue(line, "need_cse_after_recompute")
        comm = self.extractInt64Value(line, "comm:")
        mirror_user_id = self.extractStringValue(line, "mirror_user_id:")
        ir_short_name = self.extractOpName(line)

        if ir_short_name.lower() in COMM_OP_LIST:
            ir_comm_input_type = self.extractListValue(next_line, r"<Tensor")
            ir_comm_input_shape = self.extractTupleValue(next_line, rf"\(<Tensor\[{ir_comm_input_type}\],")
            ir_comm_output_type = self.extractListValue(next_line, r"-> \(<Tensor")
            ir_comm_output_shape = self.extractTupleValue(next_line, rf"-> \(<Tensor\[{ir_comm_output_type}\],")
        else:
            ir_comm_input_type = ""
            ir_comm_input_shape = ""
            ir_comm_output_type = ""
            ir_comm_output_shape = ""

        op_attrs = {}
        op_attrs[IR_DUPLICATED] = 1 if duplicated_flag else 0
        op_attrs[IR_DP] = 1 if dp_flag else 0
        op_attrs[IR_MICRO] = micro if micro else ""
        op_attrs[IR_UNIQUE_ID] = unique_id if unique_id else ""
        op_attrs[IR_FORWARD_UNIQUE_ID] = forward_unique_id if forward_unique_id else ""
        op_attrs[IR_GROUP_RANK_IDS] = group_rank_ids if group_rank_ids else ""
        op_attrs[IR_GROUP] = group if group else ""
        op_attrs[IR_NEED_CSE_AFTER_RECOMPUTE] = 1 if need_cse_after_recompute else 0
        op_attrs[IR_COMM_ID] = comm if comm else ""
        op_attrs[IR_MIRROR_ID] = mirror_user_id if mirror_user_id else ""
        op_attrs[IR_OP_SHORT_NAME] = ir_short_name if ir_short_name else ""
        op_attrs[IR_COMM_INPUT_TYPE] = ir_comm_input_type
        op_attrs[IR_COMM_INPUT_SHAPE] = ir_comm_input_shape
        op_attrs[IR_COMM_OUTPUT_TYPE] = ir_comm_output_type
        op_attrs[IR_COMM_OUTPUT_SHAPE] = ir_comm_output_shape
        op_attrs[DI_RECOMPUTE] = op_attrs[IR_DUPLICATED]
        return op_attrs

    def extract_ops_info_from_ir(self, ir_file_path):
        op_attrs = {}
        flag_get_op_info = False
        ir_file_path = os.path.realpath(ir_file_path)
        if not ir_file_path.endswith(".ir"):
            return
        with open(ir_file_path, "r", encoding="utf-8") as file:
            line = file.readline()
            while line:
                if not flag_get_op_info:
                    if self.is_valid_op_info_line(line):
                        next_line = file.readline()
                        op_attrs = self.get_ir_line_info(line, next_line)
                        line = next_line
                        flag_get_op_info = True

                fullname = line
                if flag_get_op_info and "Fullname with scope" in fullname:
                    fullname = re.search(r"Fullname with scope: \((.*)\)", fullname).group(1)
                    if fullname in self.ir_op_dict.keys():
                        raise Exception("Extract ir graph information error: found repeat op name!")
                    self.ir_op_dict[fullname] = op_attrs
                    op_attrs = {}
                    flag_get_op_info = False

                line = file.readline()

    def analy_recompute_by_unique_id(self):
        forward_unique_id_dict = {}
        micro_max_value = -1
        for ir_op_full_name, ir_op_attrs in self.ir_op_dict.items():
            micro = ir_op_attrs[IR_MICRO]
            if micro:
                micro_max_value = max(micro_max_value, int(micro))
            
            forward_unique_id = ir_op_attrs[IR_FORWARD_UNIQUE_ID]
            if forward_unique_id:
                forward_unique_id_dict[forward_unique_id] = []
        micro_max_value += 1

        for ir_op_full_name, ir_op_attrs in self.ir_op_dict.items():
            current_op_unique_id = ir_op_attrs[IR_UNIQUE_ID]
            if current_op_unique_id == "":
                continue
            if current_op_unique_id in forward_unique_id_dict:
                forward_unique_id_dict[current_op_unique_id].append(ir_op_full_name)

        count = 0
        for op_unique_id, op_full_name in forward_unique_id_dict.items():
            if len(op_full_name) != 2 * micro_max_value:
                continue
            for index in range(1, len(op_full_name), 2):
                self.ir_op_dict[op_full_name[index]][DI_RECOMPUTE] = 1
                count += 1

    def analy_ir_file(self, ir_file_path_dir):
        if ir_file_path_dir == "":
            return
        self.ir_op_dict = {}
        ir_file_list = self.get_ir_file_list(ir_file_path_dir)
        for ir_file in ir_file_list:
            self.extract_ops_info_from_ir(ir_file)

        self.analy_recompute_by_unique_id()
