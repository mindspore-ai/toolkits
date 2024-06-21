import os
from collections import defaultdict
from pathlib import Path
from troubleshooter.migrator.api_dump.ms_dump.hooks import DumpUtil, ad_dump_acc_cmp
from mindtorch.torch.tensor import cast_to_adapter_tensor

NNCount = defaultdict(int)

def make_adapter_dump_dirs(rank):
    dump_file_name, dump_path = "mindtorch_api_dump_info.csv", "mindtorch_api_dump"
    dump_stack_file = "mindtorch_api_dump_stack.json"
    dump_root_dir = DumpUtil.dump_ori_dir if DumpUtil.dump_ori_dir else "./"
    Path(dump_root_dir).mkdir(mode=0o700, parents=True, exist_ok=True)
    rank_dir = os.path.join(dump_root_dir, 'rank' + str(rank))
    if not os.path.exists(rank_dir):
        os.mkdir(rank_dir, mode=0o700)
    DumpUtil.dump_dir = rank_dir
    dump_file_path = os.path.join(rank_dir, dump_path)
    dump_file_name = os.path.join(rank_dir, dump_file_name)
    dump_stack_path = os.path.join(rank_dir, dump_stack_file)
    DumpUtil.set_dump_path(dump_file_path, dump_file_name, dump_stack_path)

def make_pth_dir():
    return os.path.join(DumpUtil.dump_dir, "ad_net.pth")

def acc_cmp_dump(name, **kwargs):
    dump_step = kwargs.get('dump_step', 1)
    pid = kwargs.get('pid')
    DumpUtil.dump_config = kwargs.get('dump_config')
    name_template = name
    if not pid:
        return RuntimeError("Not get the specified process pid.")

    def acc_cmp_hook(cell, in_feat, out_feat):
        nonlocal name, name_template
        global NNCount
        if "{}_" in name_template:
            # name_template like 'NN_Conv2d_{}_forward'
            nn_name = name_template.split('_')[1]
            id = NNCount[nn_name]
            NNCount[nn_name] = id + 1
            name = name_template.format(id)
        if pid == os.getpid():
            return cast_to_adapter_tensor(ad_dump_acc_cmp(name, in_feat, out_feat, dump_step))

    return acc_cmp_hook