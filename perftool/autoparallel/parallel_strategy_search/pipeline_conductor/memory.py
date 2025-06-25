from utils.logger import logger


class Memory:
    select_mem0 = 76
    select_mem12 = 76
    select_mem = 77
    re_comp_mem0 = 3
    re_comp_mem12 = 3
    re_comp_mem = 5
    act_mem0 = 79
    act_mem12 = 79
    act_mem = 79
    layer_mem012 = 349
    layer_mem = 340
    static_mem0 = 734
    static_mem = 116
    lm_head_mem = 690

    def __init__(self, mem_lim):
        self.mem_lim = mem_lim
        self.mem_lim_stage0 = mem_lim - self.static_mem0
        self.mem_lim_others = mem_lim - self.static_mem
        self.mem_lim_last = mem_lim - self.lm_head_mem

    def update_up_mem(self):
        self.mem_lim_stage0 = self.mem_lim - self.static_mem0
        self.mem_lim_others = self.mem_lim - self.static_mem
        self.mem_lim_last = self.mem_lim - self.lm_head_mem

    def print_mem(self):
        logger.info(f'select_mem0={self.select_mem0}, select_mem12={self.select_mem12}, select_mem={self.select_mem}, '
                    f're_comp_mem0={self.re_comp_mem0}, re_comp_mem12={self.re_comp_mem12}, '
                    f're_comp_mem={self.re_comp_mem}, '
                    f'act_mem0={self.act_mem0}, act_mem12={self.act_mem12}, act_mem={self.act_mem}, '
                    f'layer_mem012={self.layer_mem012}, layer_mem={self.layer_mem}, '
                    f'static_mem0={self.static_mem0}, static_mem={self.static_mem}, '
                    f'lm_head_mem={self.lm_head_mem}, mem_lim_stage0={self.mem_lim_stage0}, '
                    f'mem_lim_others={self.mem_lim_others}, mem_lim_last={self.mem_lim_last}')

    def write_memory_to_file(self, mem_file):
        with open(mem_file, 'w') as file:
            file.write(f'select_mem0={self.select_mem0}\n')
            file.write(f'select_mem12={self.select_mem12}\n')
            file.write(f'select_mem={self.select_mem}\n')
            file.write(f're_comp_mem0={self.re_comp_mem0}\n')
            file.write(f're_comp_mem12={self.re_comp_mem12}\n')
            file.write(f're_comp_mem={self.re_comp_mem}\n')
            file.write(f'act_mem0={self.act_mem0}\n')
            file.write(f'act_mem12={self.act_mem12}\n')
            file.write(f'act_mem={self.act_mem}\n')
            file.write(f'layer_mem012={self.layer_mem012}\n')
            file.write(f'layer_mem={self.layer_mem}\n')
            file.write(f'static_mem0={self.static_mem0}\n')
            file.write(f'static_mem={self.static_mem}\n')
            file.write(f'lm_head_mem={self.lm_head_mem}\n')
            logger.info(f'Write memory info to {mem_file}')

    def get_mem(self):
        mem = (f'select_mem0={self.select_mem0}\n'
              f'select_mem12={self.select_mem12}\n'
              f'select_mem={self.select_mem}\n'
              f're_comp_mem0={self.re_comp_mem0}\n'
              f're_comp_mem12={self.re_comp_mem12}\n'
              f're_comp_mem={self.re_comp_mem}\n'
              f'act_mem0={self.act_mem0}\n'
              f'act_mem12={self.act_mem12}\n'
              f'act_mem={self.act_mem}\n'
              f'layer_mem012={self.layer_mem012}\n'
              f'layer_mem={self.layer_mem}\n'
              f'static_mem0={self.static_mem0}\n'
              f'static_mem={self.static_mem}\n'
              f'lm_head_mem={self.lm_head_mem}')
        return mem
