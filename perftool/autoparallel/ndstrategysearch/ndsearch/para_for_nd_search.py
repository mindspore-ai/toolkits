from pipeline_conductor.pp_util import parse_shell, parse_shell_config
from utils.common import cal_model_layers_num
from utils.input_config import InputConfig


class ParaForNd:
    def __init__(self, para):
        self.mf_args = None
        self.dp = 1
        self.pp = 1
        self.tp = 1
        self.cp = 1
        self.num_layers = 1
        self.mbn = 1
        self.max_mem = 58 * 1024
        self.expert_num = 0
        self.get_args_from_file(para)

    def get_args_from_file(self, para):
        if para.YAML_PATH:
            input_args = InputConfig(para.YAML_PATH)
            self.dp = input_args.parallel_config.data_parallel
            self.tp = input_args.parallel_config.model_parallel
            self.pp = input_args.parallel_config.pipeline_stage
            self.num_layers = cal_model_layers_num(input_args)
            self.mbn = input_args.parallel_config.micro_batch_num
            self.max_mem = input_args.context.max_device_memory
            self.expert_num = input_args.moe_config.expert_num
            self.mf_args = input_args

        elif para.SHELL_PATH:
            # todo: 填入shell解析
            input_args, unparses = parse_shell(para.SHELL_PATH)
            self.dp = input_args.get('DP')
            self.pp = input_args.get('PP')
            self.tp = input_args.get('TP')
            self.num_layers = input_args.get('NUM_LAYERS')
            if 'EXPERT_NUM' in input_args:
                self.expert_num = input_args['EXPERT_NUM']
            # TODO: parser  readme要求用户显示化定义
            self.mbn = input_args.get('GBS') // input_args.get('MBS') // self.dp
        else:
            RuntimeError("pls input valid config file")


