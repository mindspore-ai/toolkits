import pytest
import tempfile
import shutil
import troubleshooter as ts
import mindspore as ms
import numpy as np

from mindspore import nn, ops
from pathlib import Path
from tests.st.troubleshooter.migrator.dump.utils import get_l2norm_list
from troubleshooter.migrator import api_dump_init, api_dump_start, api_dump_stop


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.a = 2
    
    def construct(self, x):
        y = x + x
        z = x * x
        out = ops.div(y, z)
        return out * self.a


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_l2norm_ms():
    ms.set_context(mode=ms.PYNATIVE_MODE)
    dump_path = Path(tempfile.mkdtemp(prefix="ms_api_dump_l2norm"))
    try:
        net = Net()
        api_dump_init(net, dump_path, retain_backward=True)

        api_dump_start()
        data = ms.Tensor(np.ones([2,10]), ms.float16)
        out = net(data)
        api_dump_stop()

        l2norm_list = get_l2norm_list(dump_path, 'mindspore')
        assert len(l2norm_list) == 11
        assert l2norm_list[0] == 4.47265625
        assert l2norm_list[10] == 17.890625
        assert l2norm_list[5] == 4.47265625
        
    finally:
        shutil.rmtree(dump_path)
