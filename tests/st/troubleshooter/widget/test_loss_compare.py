import os
import pytest
import tempfile
import troubleshooter as ts


def test_loss_compare():
    left_file = {
    "path": os.path.realpath("./test_files/npu_loss.txt"),
    "loss_tag": "lm_loss:",
    "label": "npu_loss"
    }

    right_file = {
    "path": os.path.realpath("./test_files/gpu_loss.txt"),
    "loss_tag": "lm_loss:",
    "label": "npu_loss"
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        ts.widget.loss_compare(left_file, right_file)
        output_dir = os.path.join(os.getcwd(), "loss_compare")
        file_count = len(
            [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))])
        expected_count = 5
        assert file_count == expected_count
        expected_type = [".csv", ".png"]
        for f in os.listdir(output_dir):
            file_type = os.path.splitext(f)[1]
            assert file_type in expected_type
