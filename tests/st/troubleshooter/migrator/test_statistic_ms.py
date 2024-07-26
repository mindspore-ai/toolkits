import pytest
import tempfile
import shutil
import troubleshooter as ts
import mindspore as ms
import numpy as np

from mindspore import nn, ops
from pathlib import Path
from tests.st.troubleshooter.migrator.dump.utils import get_l2norm_list, get_summary_list
from troubleshooter.migrator import api_dump_init, api_dump_start, api_dump_stop


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.a = 2
    
    def construct(self, x):
        y = x + x
        z = x * x
        out = ops.div(y, z)
        out = out * self.a
        return out


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_statistic_float_ms():
    ms.set_context(mode=ms.PYNATIVE_MODE)
    dump_path = Path(tempfile.mkdtemp(prefix="ms_api_dump_statistic_float"))
    try:
        net = Net()
        api_dump_init(net, dump_path, retain_backward=True)
        api_dump_start(statistic_category=['max', 'min', 'avg', 'md5', 'l2norm'])
        data = ms.Tensor([1,2,-3,4,5,6,7,8,9,0], ms.float32)
        out = net(data)
        api_dump_stop()

        summary_list = get_summary_list(dump_path, 'mindspore')
        assert len(summary_list) == 11
        max = summary_list[3][0]
        min = summary_list[3][1]
        avg = summary_list[3][2]
        assert max == 9.0
        assert min == -3.0
        assert avg == 3.9000000953674316

        l2norm_list = get_l2norm_list(dump_path, 'mindspore')
        assert len(l2norm_list) == 11
        assert l2norm_list[3] == 16.881942749023438
        assert l2norm_list[6] == 33.763885498046875        
    finally:
        shutil.rmtree(dump_path)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_statistic_notfloat_ms():
    ms.set_context(mode=ms.PYNATIVE_MODE)
    dump_path = Path(tempfile.mkdtemp(prefix="ms_api_dump_statistic_notfloat"))
    try:
        net = Net()
        api_dump_init(net, dump_path, retain_backward=True)
        api_dump_start(statistic_category=['max', 'min', 'avg', 'md5', 'l2norm'])
        data = ms.Tensor([1,2,3,4,5,6], ms.int32)
        out = net(data)
        api_dump_stop()

        summary_list = get_summary_list(dump_path, 'mindspore')
        assert len(summary_list) == 3
        max = summary_list[2][0]
        min = summary_list[2][1]
        avg = summary_list[2][2]
        assert max == 4.0
        assert min == 0.6666666865348816
        assert avg == 1.6333333253860474

        l2norm_list = get_l2norm_list(dump_path, 'mindspore')
        assert len(l2norm_list) == 3
        assert l2norm_list[0] == 2.442448616027832
        assert l2norm_list[2] == 4.884897232055664        
    finally:
        shutil.rmtree(dump_path)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_statistic_scalar_ms():
    ms.set_context(mode=ms.PYNATIVE_MODE)
    dump_path = Path(tempfile.mkdtemp(prefix="ms_api_dump_statistic_scalar"))
    try:
        net = Net()
        api_dump_init(net, dump_path, retain_backward=True)
        api_dump_start(statistic_category=['max', 'min', 'avg', 'md5', 'l2norm'], filter_data=False)
        data = 100
        out = net(data)
        api_dump_stop()

        summary_list = get_summary_list(dump_path, 'mindspore')
        assert len(summary_list) == 6
        max = summary_list[5][0]
        min = summary_list[5][1]
        avg = summary_list[5][2]
        assert max == 0.03999999910593033
        assert min == 0.03999999910593033
        assert avg == 0.03999999910593033

        l2norm_list = get_l2norm_list(dump_path, 'mindspore')
        assert len(l2norm_list) == 6
        assert l2norm_list[0] == 200.0
        assert l2norm_list[5] == 0.03999999910593033      
    finally:
        shutil.rmtree(dump_path)