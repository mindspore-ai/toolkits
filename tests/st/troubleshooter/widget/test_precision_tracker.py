import pytest
import csv
import glob
import tempfile
import troubleshooter as ts


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_precision_tracker_all():
    """
    Feature: ts.widget.precision_tracker
    Description: Test precision_tracker can correctly parse pb file
    Expectation: The csv file generated successfully and s1 == s2
    """
    pb_file_path = "./test_files/ms_output_after_hwopt_0.pb"
    with tempfile.TemporaryDirectory() as tmpdir:
        ts.widget.precision_tracker(pb_file_path, output_path=tmpdir)
        result_csv_path_list = glob.glob(f"{tmpdir}/ms_output_after_hwopt_0.csv")
        assert len(result_csv_path_list) == 1
        s1 = {'normal', 'raise'}
        s2 = set()
        with open(result_csv_path_list[0], 'r') as fr:
            reader = csv.DictReader(fr)
            for row in reader:
                s2.add(row.get('precision_flag'))
        assert s1 == s2


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_precision_tracker_part():
    """
    Feature: ts.widget.precision_tracker
    Description: Test precision_tracker can correctly parse pb file
    Expectation: The csv file generated successfully and s1 == s2
    """
    pb_file_path = "./test_files/ms_output_after_hwopt_0.pb"
    with tempfile.TemporaryDirectory() as tmpdir:
        ts.widget.precision_tracker(pb_file_path, precision_flags=('raise',), output_path=tmpdir)
        result_csv_path_list = glob.glob(f"{tmpdir}/ms_output_after_hwopt_0.csv")
        assert len(result_csv_path_list) == 1
        s1 = {'raise'}
        s2 = set()
        with open(result_csv_path_list[0], 'r') as fr:
            reader = csv.DictReader(fr)
            for row in reader:
                s2.add(row.get('precision_flag'))
        assert s1 == s2
