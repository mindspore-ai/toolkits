import tempfile
from pathlib import Path
import shutil

import numpy as np
import pytest

from troubleshooter.migrator import api_dump_compare



def generate_data():
    data_path = Path(tempfile.mkdtemp(prefix="test_data"))
    np.save(data_path / 'label.npy',
            np.random.randn(1, 10).astype(np.float32))
    np.save(data_path / 'data.npy',
            np.random.randn(1, 3, 90, 300).astype(np.float32))
    return data_path

def analyse(result):
    print(result)
    print(f"True has {result.count('True')}")
    print(f"False has {result.count('False')}")

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_compare_api_dump_all(capsys):
    from tests.st.troubleshooter.migrator.dump.demo.test_torch_dump import train_pt_one_step_all
    from tests.st.troubleshooter.migrator.dump.demo.test_ms_dump import train_ms_one_step_all

    data_path = generate_data()
    torch_info_path = Path(tempfile.mkdtemp(prefix="torch_info"))
    torch_dump_path = Path(tempfile.mkdtemp(prefix="torch_dump"))
    ms_dump_path = Path(tempfile.mkdtemp(prefix="ms_dump"))
    try:
        train_pt_one_step_all(data_path, torch_dump_path, torch_info_path)
        train_ms_one_step_all(data_path, ms_dump_path, torch_info_path)
        api_dump_compare(torch_dump_path, ms_dump_path, rtol=1e-3, atol=1e-3)
        result = capsys.readouterr().out
        assert result.count("True") >= 20
        assert result.count("False") <= 1
    finally:
        shutil.rmtree(torch_dump_path)
        shutil.rmtree(torch_info_path)
        shutil.rmtree(ms_dump_path)
        shutil.rmtree(data_path)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_compare_api_dump_part(capsys):
    from tests.st.troubleshooter.migrator.dump.demo.test_torch_dump import train_pt_one_step_part
    from tests.st.troubleshooter.migrator.dump.demo.test_ms_dump import train_ms_one_step_part

    data_path = generate_data()
    torch_info_path = Path(tempfile.mkdtemp(prefix="torch_info"))
    torch_dump_path = Path(tempfile.mkdtemp(prefix="torch_dump"))
    ms_dump_path = Path(tempfile.mkdtemp(prefix="ms_dump"))
    try:
        train_pt_one_step_part(data_path, torch_dump_path, torch_info_path)
        train_ms_one_step_part(data_path, ms_dump_path, torch_info_path)
        api_dump_compare(torch_dump_path, ms_dump_path, rtol=1e-3, atol=1e-3)

        result = capsys.readouterr().out
        # mindspore backward miss
        assert result.count("True") >= 4
        assert result.count("False") <= 2
    finally:
        shutil.rmtree(data_path)
        shutil.rmtree(torch_dump_path)
        shutil.rmtree(torch_info_path)
        shutil.rmtree(ms_dump_path)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_compare_api_dump_api_list(capsys):
    from tests.st.troubleshooter.migrator.dump.demo.test_torch_dump import train_pt_one_step_api_list
    from tests.st.troubleshooter.migrator.dump.demo.test_ms_dump import train_ms_one_step_api_list

    data_path = generate_data()
    torch_info_path = Path(tempfile.mkdtemp(prefix="torch_info"))
    torch_dump_path = Path(tempfile.mkdtemp(prefix="torch_dump"))
    ms_dump_path = Path(tempfile.mkdtemp(prefix="ms_dump"))
    try:
        train_pt_one_step_api_list(data_path, torch_dump_path, torch_info_path)
        train_ms_one_step_api_list(data_path, ms_dump_path, torch_info_path)
        api_dump_compare(torch_dump_path, ms_dump_path, rtol=1e-3, atol=1e-3)

        result = capsys.readouterr().out
        # mindspore conv2d backward miss
        assert result.count("True") >= 8
        assert result.count("False") <= 1
    finally:
        shutil.rmtree(data_path)
        shutil.rmtree(torch_dump_path)
        shutil.rmtree(torch_info_path)
        shutil.rmtree(ms_dump_path)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_compare_api_dump_list(capsys):
    from tests.st.troubleshooter.migrator.dump.demo.test_torch_dump import train_pt_one_step_list
    from tests.st.troubleshooter.migrator.dump.demo.test_ms_dump import train_ms_one_step_list

    data_path = generate_data()
    torch_info_path = Path(tempfile.mkdtemp(prefix="torch_info"))
    torch_dump_path = Path(tempfile.mkdtemp(prefix="torch_dump"))
    ms_dump_path = Path(tempfile.mkdtemp(prefix="ms_dump"))
    try:
        train_pt_one_step_list(data_path, torch_dump_path, torch_info_path)
        train_ms_one_step_list(data_path, ms_dump_path, torch_info_path)
        api_dump_compare(torch_dump_path, ms_dump_path, rtol=1e-3, atol=1e-3)

        result = capsys.readouterr().out
        assert result.count("True") == 6
        assert result.count("False") == 0
    finally:
        shutil.rmtree(data_path)
        shutil.rmtree(torch_dump_path)
        shutil.rmtree(torch_info_path)
        shutil.rmtree(ms_dump_path)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_compare_api_dump_range(capsys):
    from tests.st.troubleshooter.migrator.dump.demo.test_torch_dump import train_pt_one_step_range
    from tests.st.troubleshooter.migrator.dump.demo.test_ms_dump import train_ms_one_step_range

    data_path = generate_data()
    torch_info_path = Path(tempfile.mkdtemp(prefix="torch_info"))
    torch_dump_path = Path(tempfile.mkdtemp(prefix="torch_dump"))
    ms_dump_path = Path(tempfile.mkdtemp(prefix="ms_dump"))
    try:
        train_pt_one_step_range(data_path, torch_dump_path, torch_info_path)
        train_ms_one_step_range(data_path, ms_dump_path, torch_info_path)
        api_dump_compare(torch_dump_path, ms_dump_path, rtol=1e-3, atol=1e-3)

        result = capsys.readouterr().out
        assert result.count("True") == 15
        assert result.count("False") == 0
    finally:
        shutil.rmtree(data_path)
        shutil.rmtree(torch_dump_path)
        shutil.rmtree(torch_info_path)
        shutil.rmtree(ms_dump_path)