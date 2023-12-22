import os, sys
sys.path.append(os.getcwd())
import pytest
import tempfile
from mindconverter.cli import _run
from pathlib import Path

standared_code_template = '''import torch
import torch.nn as nn
from torch import optim

class Model(nn.Module):
'''
init_code_template = '''
    def __init__(self):
        super(Model, self).__init__()
'''
forward_code_template = '''
    def forward(self, x):
'''

#Test 1#测试单独的算子
#输入格式 需要转换的torch算子，目标转换的mindspore的算子
#测试torch
@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_torch_migrate(torch_api, ms_template_api):
    model_temp = standared_code_template + init_code_template + forward_code_template

    test_torch = "        " + torch_api
    model_temp += test_torch
    temp_dir_t = tempfile.TemporaryDirectory()
    temp_dir_file = temp_dir_t.name + "/temp.py"
    temp_file = open(temp_dir_file, "w")
    temp_file.write(model_temp)
    temp_file.close()

    output_dir_t = tempfile.TemporaryDirectory()
    output_dir = output_dir_t.name
    rep_dir_t = tempfile.TemporaryDirectory()
    rep_dir = rep_dir_t.name
    _run(temp_dir_file,output_dir,rep_dir)
    output_file = open(output_dir + "/temp.py", 'r')
    converter_code = output_file.readlines()
    output_file.close()

    assert converter_code[-1].rstrip("\n").replace(" ","") == ms_template_api

# #测试torch.nn
@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_torch_nn_migrate(torch_nn_api, ms_nn_template_api):
    model_temp = standared_code_template + init_code_template

    test_torch = "        " + torch_nn_api
    model_temp += test_torch
    temp_dir_t = tempfile.TemporaryDirectory()
    temp_dir_file = temp_dir_t.name + "/temp.py"
    temp_file = open(temp_dir_file, "w")
    temp_file.write(model_temp)
    temp_file.close()

    output_dir_t = tempfile.TemporaryDirectory()
    output_dir = output_dir_t.name
    rep_dir_t = tempfile.TemporaryDirectory()
    rep_dir = rep_dir_t.name
    _run(temp_dir_file,output_dir,rep_dir)
    output_file = open(output_dir + "/temp.py", 'r')
    converter_code = output_file.readlines()
    output_file.close()

    assert converter_code[-1].rstrip("\n").replace(" ","") == ms_nn_template_api

# #测试Tensor
@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_tensor_migrate(tensor_api, ms_tensor_template_api):
    model_temp = standared_code_template + init_code_template + forward_code_template

    test_torch = "        " + tensor_api
    model_temp += test_torch
    temp_dir_t = tempfile.TemporaryDirectory()
    temp_dir_file = temp_dir_t.name + "/temp.py"
    temp_file = open(temp_dir_file, "w")
    temp_file.write(model_temp)
    temp_file.close()

    output_dir_t = tempfile.TemporaryDirectory()
    output_dir = output_dir_t.name
    rep_dir_t = tempfile.TemporaryDirectory()
    rep_dir = rep_dir_t.name
    _run(temp_dir_file,output_dir,rep_dir)
    output_file = open(output_dir + "/temp.py", 'r')
    converter_code = output_file.readlines()
    output_file.close()
    
    assert converter_code[-1].rstrip("\n").replace(" ","") == ms_tensor_template_api

# #测试optim
@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_torch_optim_migrate(torch_optim_api, ms_template_api):
    model_temp = standared_code_template + init_code_template + forward_code_template

    test_torch = "        " + torch_optim_api
    model_temp += test_torch
    temp_dir_t = tempfile.TemporaryDirectory()
    temp_dir_file = temp_dir_t.name + "/temp.py"
    temp_file = open(temp_dir_file, "w")
    temp_file.write(model_temp)
    temp_file.close()

    output_dir_t = tempfile.TemporaryDirectory()
    output_dir = output_dir_t.name
    rep_dir_t = tempfile.TemporaryDirectory()
    rep_dir = rep_dir_t.name
    _run(temp_dir_file,output_dir,rep_dir)
    output_file = open(output_dir + "/temp.py", 'r')
    converter_code = output_file.readlines()
    output_file.close()
    
    assert converter_code[-1].rstrip("\n").replace(" ","") == ms_template_api

#Test 2测试整个模型文件
#输入torch模型文件，以及标准转换的mindspore模型文件
#测试整个模型文件
model_file_path = "./modeltest/"
ms_template_file_path = "./template_model/"
@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize("model_file_path, ms_template_file_path", model_file_path, ms_template_file_path)
def test_torch_mindspore_model(capsys, model_file_path, ms_template_file_path):
    infile = model_file_path
    output_dir_t = tempfile.TemporaryDirectory()
    output_dir = output_dir_t.name
    rep_dir_t = tempfile.TemporaryDirectory()
    rep_dir = rep_dir_t.name
    modelname = infile.split("/")[-1]
    path_converter = Path(output_dir + "/" + modelname)
    _run(infile,output_dir,rep_dir)
    output_file = open(path_converter, 'r')
    converter_code = output_file.readlines()
    output_file.close()
    ms_code_file = open(ms_template_file_path, 'r')
    ms_code = ms_code_file.readlines()
    ms_code_file.close()
    pos_converter = 0
    pos_ms_code = 0
    #忽略import和from import语句
    for i in range(len(converter_code)):
        if converter_code[i].startswith("import") or converter_code[i].startswith("from") or converter_code[i] == "\n":
            continue
        else:
            pos_converter = i
            break

    for i in range(len(ms_code)):
        if ms_code[i].startswith("import") or ms_code[i].startswith("from") or converter_code[i] == "\n":
            continue
        else:
            pos_ms_code = i
            break

    assert (len(converter_code) - pos_converter) == (len(ms_code) - pos_ms_code)

    for i in range(pos_converter, len(converter_code)):
        assert converter_code[i].rstrip('\n') == ms_code[i].rstrip('\n')