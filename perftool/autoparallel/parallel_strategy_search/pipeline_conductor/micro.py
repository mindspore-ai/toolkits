from pipeline_conductor.memory import Memory


class Micro:
    part = int
    vpp = int
    state = 'f'
    micro_id = int
    split = int

    def __init__(self, part, vpp, state, micro_id, split):
        self.part = part
        self.vpp = vpp
        self.state = state
        self.micro_id = micro_id
        self.split = split


class SortMicro:
    parts = int
    num_vpp = int
    num_stage = int
    distribution = []
    low_mem = bool
    seq_split = int
    is_f_then_b = False

    def __init__(self, parts, num_vpp, num_stage, distribution, low_mem, seq_split):
        self.forward = []
        self.backward = []
        self.warmup_num = []
        self.final_orders = []
        self.parts = parts
        self.num_vpp = num_vpp
        self.num_stage = num_stage
        self.distribution = distribution
        self.low_mem = low_mem
        self.seq_split = seq_split
        self.build_f_b_sort()
        self.set_warmup_num()
        self.set_micro_sort()

    def build_f_b_sort(self):
        for part in range(self.parts):
            for vpp in range(self.num_vpp):
                for micro_id in range(self.distribution[part]):
                    for split in range(self.seq_split):
                        micro = Micro(part, vpp, 'f', micro_id, split)
                        self.forward.append(micro)
            for vpp in range(self.num_vpp - 1, -1, -1):
                for micro_id in range(self.distribution[part]):
                    for split in range(self.seq_split - 1, -1, -1):
                        micro = Micro(part, vpp, 'b', micro_id, split)
                        self.backward.append(micro)

    def set_warmup_num(self):
        for stage in range(self.num_stage):
            if self.low_mem:
                warmup = min(((self.num_vpp - 1) * self.distribution[0] + (self.num_stage - stage - 1)) *
                             self.seq_split, len(self.forward))
            else:
                warmup = min(((self.num_vpp - 1) * self.distribution[0] + (self.num_stage - stage - 1) * 2) *
                             self.seq_split, len(self.forward))
            # 最后一个stage，第一个micro前向做完之后才能做后向
            if stage == self.num_stage - 1:
                warmup = warmup + self.seq_split - 1
            self.warmup_num.append(warmup)

    def set_micro_sort(self):
        for stage in range(self.num_stage):
            stage_order = []
            stage_order += self.forward[: self.warmup_num[stage]]
            for i in range(self.warmup_num[stage], len(self.forward)):
                stage_order.append(self.forward[i])
                stage_order.append(self.backward[i - self.warmup_num[stage]])
            stage_order += self.backward[len(self.forward) - self.warmup_num[stage]:]
            self.final_orders.append(stage_order)


class PeakNum:
    sort_micro = SortMicro

    def __init__(self, sort_micro: SortMicro):
        self.peak_num_recompute_type1 = {}
        self.peak_num_recompute_type2 = {}
        self.peak_num_select_recom_type1 = {}
        self.peak_num_select_recom_type2 = {}
        self.peak_num_act_type1 = {}
        self.peak_num_act_type2 = {}
        self.max_mem = {}
        self.micro_num_of_max_mem = {}
        self.sort_micro = sort_micro
        self.num_stage = sort_micro.num_stage
        self.num_vpp = sort_micro.num_vpp

    def set_peak_act_recompute_num(self, x_type2, rs_type2, ra_type2, x_type1, rs_type1, ra_type1, memory: Memory):
        for stage in range(self.sort_micro.num_stage):
            # 各个micro处的内存及激活、重计算份数
            self.max_mem[stage] = 0
            if stage == 0:
                static_mem = memory.static_mem0
            elif stage == self.sort_micro.num_stage - 1:
                static_mem = memory.lm_head_mem
            else:
                static_mem = memory.static_mem
            layer_mem = (memory.layer_mem012 * sum(x_type1[vpp][stage] for vpp in range(self.num_vpp)) + memory.
                         layer_mem * sum(x_type2[vpp][stage] for vpp in range(self.num_vpp)))
            mem_stage = static_mem + layer_mem
            num_recom_type1_stage = 0
            num_recom_type2_stage = 0
            num_select_recom_type1_stage = 0
            num_select_recom_type2_stage = 0
            num_act_type1_stage = 0
            num_act_type2_stage = 0
            for i in range(len(self.sort_micro.final_orders[stage])):
                micro_batch = self.sort_micro.final_orders[stage][i]
                vpp = micro_batch.vpp
                act_num_type1 = x_type1[vpp][stage] - rs_type1[vpp][stage] - ra_type1[vpp][stage]
                act_num_type2 = x_type2[vpp][stage] - rs_type2[vpp][stage] - ra_type2[vpp][stage]
                act_mem = memory.act_mem12 * act_num_type1 + memory.act_mem * act_num_type2
                ra_mem = memory.re_comp_mem12 * ra_type1[vpp][stage] + memory.re_comp_mem * ra_type2[vpp][stage]
                rs_mem = memory.select_mem12 * rs_type1[vpp][stage] + memory.select_mem * rs_type2[vpp][stage]
                # 计算时间切成seq_split份，可以看成数据切成seq_split份，即动态内存切分；layer_mem与static_mem不变
                total_mem = (act_mem + ra_mem + rs_mem) / self.sort_micro.seq_split
                if micro_batch.state == 'f':
                    mem_stage += total_mem
                    num_recom_type1_stage += ra_type1[vpp][stage]
                    num_recom_type2_stage += ra_type2[vpp][stage]
                    num_select_recom_type1_stage += rs_type1[vpp][stage]
                    num_select_recom_type2_stage += rs_type2[vpp][stage]
                    num_act_type1_stage += act_num_type1
                    num_act_type2_stage += act_num_type2
                else:
                    mem_stage -= total_mem
                    num_recom_type1_stage -= ra_type1[vpp][stage]
                    num_recom_type2_stage -= ra_type2[vpp][stage]
                    num_select_recom_type1_stage -= rs_type1[vpp][stage]
                    num_select_recom_type2_stage -= rs_type2[vpp][stage]
                    num_act_type1_stage -= act_num_type1
                    num_act_type2_stage -= act_num_type2
                if mem_stage > self.max_mem[stage]:
                    self.max_mem[stage] = mem_stage
                    self.peak_num_recompute_type1[stage] = num_recom_type1_stage
                    self.peak_num_recompute_type2[stage] = num_recom_type2_stage
                    self.peak_num_select_recom_type1[stage] = num_select_recom_type1_stage
                    self.peak_num_select_recom_type2[stage] = num_select_recom_type2_stage
                    self.peak_num_act_type1[stage] = num_act_type1_stage
                    self.peak_num_act_type2[stage] = num_act_type2_stage
                    self.micro_num_of_max_mem[stage] = i + 1
