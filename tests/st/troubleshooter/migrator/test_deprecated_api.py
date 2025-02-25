import pytest
import troubleshooter as ts

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_api_dump_exectption():
    with pytest.raises(RuntimeError):
        ts.migrator.api_dump_init()
    with pytest.raises(RuntimeError):
        ts.migrator.api_dump_start()
    with pytest.raises(RuntimeError):
        ts.migrator.api_dump_stop()
    with pytest.raises(RuntimeError):
        ts.migrator.api_dump_compare()
    with pytest.raises(RuntimeError):
        _ = ts.migrator.NetDifferenceFinder()
