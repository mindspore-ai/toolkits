import numpy as np
import pytest
import troubleshooter as ts

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fix_random():
    """
    Feature: ts.fix_random
    Description: Verify the result of fix_random
    Expectation: success
    """
    ts.fix_random()
    x1 = np.random.random()
    ts.widget.fix_random()
    x2 = np.random.random()
    assert np.allclose(x1, x2)
