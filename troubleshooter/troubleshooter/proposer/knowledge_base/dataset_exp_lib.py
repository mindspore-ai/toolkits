# Copyright 2022-2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

dataset_experience_list_cn = [
    {
     "ID": "dataset_id_1",
     "Fault Name": "保存MindRecord文件报错",
     "Key Log Information": "Invalid data, Page size.* is too small",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "raw page或者blob page太小导致数据保存失败，默认值32MB",
     "Modification Suggestion": "使用set_page_size接口修改Page size",
     "Fixed Code": """from mindspore.mindrecord import FileWriter
                      writer = FileWriter(file_name="test.mindrecord", shard_num=1)
                      writer.set_page_size(1 << 26)
                                             ^~~~~~~~修改Page size为128MB""",
     "Fault Case": "1. Page size报错案例："
                   "https://bbs.huaweicloud.com/forum/thread-171564-1-1.html "
                   "2. set_page_size API: "
                   "https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.mindrecord.html#mindspore.mindrecord.FileWriter.set_page_size"},
    {
     "ID": "dataset_id_2",
     "Fault Name": "batch操作报错",
     "Key Log Information": "Inconsistent batch shapes, batch operation .* same shape for each data row",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "图像数据shape不一致，batch操作失败",
     "Modification Suggestion": "建议在进行batch操作之前，使用Resize算子将图像缩放到相同尺寸 ",
     "Fixed Code":"""dataset = ...
                     dataset = dataset.map(input_columns='data', operations=c_transforms.Resize(size=(388, 388))
                     ^~~~~~~~~~~ 使用Resize算子将图像缩放到相同尺寸
                     dataset = dataset.batch(batch_size)""",
     "Fault Case": "https://bbs.huaweicloud.com/forum/thread-183195-1-1.html"},
    {
     "ID": "dataset_id_3",
     "Fault Name": "图像数据集格式错误",
     "Key Log Information": "img should be PIL image",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "使用py_transform图像处理接口时，如果输入数据类型不为PIL，会引起报错",
     "Modification Suggestion": "使用py_transform图像处理接口之前，使用py_transforms.Decode()或者py_transforms.ToPIL将图像转换为PIL格式",
     "Fault Case": "1. dataset.vision.py_transforms API列表: "
                   "https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/mindspore.dataset.vision.html "
                   "2. py_transforms数据增强报错案例: "
                   "https://bbs.huaweicloud.com/forum/thread-183195-1-1.html"},
    {
     "ID": "dataset_id_4",
     "Fault Name": "保存mindrecord文件失败",
     "Key Log Information": "Invalid file. mindrecord files already exist",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "待保存的mindrecord文件已存在，重新保存时报错",
     "Modification Suggestion": "1. 重命名待保存的mindrecord文件的文件名. "
                                "2. 删除已存在的mindrecord文件. "
                                "3. 修改接口FileWriter参数: overwrite=True, 支持覆盖旧文件.",
     "Fault Case": "1. mindrecord FileWriter: "
                   "https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/mindspore.mindrecord.html "
                   "2. mindrecord报错案例： "
                   "https://bbs.huaweicloud.com/forum/thread-184006-1-1.html"},
    {
     "ID": "dataset_id_5",
     "Fault Name": "读取mindrecord文件失败",
     "Key Log Information": "Invalid file. failed to open files .*",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "加载数据集，读取mindrecord文件时报错",
     "Modification Suggestion": "1. 检查对应文件是否存在； "
                                "2. 检查对应文件是否有读取权限； "
                                "3. 可能是系统支持的打开文件句柄数受限，增加可打开句柄数的方法：ulimit -n 2048",
     "Fault Case": "1. mindrecord FileReader: "
                   "https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/mindspore.mindrecord.html "
                   "2. MindRecord中文路径报错案例："
                   "https://bbs.huaweicloud.com/forum/forum.php?mod=viewthread&tid=183183"},
    {
     "ID": "dataset_id_6",
     "Fault Name": "GeneratorDataset使用自定义的Sampler错误",
     "Key Log Information": "has no attribute .child_sampler.",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "自定义的Sampler在__init__中没有显式调用父类dataset.Sampler的构造函数，导致丢失父类中的child_sampler",
     "Modification Suggestion": "在__init__函数中调用父类dataset.Sampler的构造函数super().__init__()",
     "Fault Case": "1. Sampler特性说明： "
                   "https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/sampler.html "
                   "2. Sampler报错案例：https://bbs.huaweicloud.com/forum/thread-184010-1-1.html"},
    {
     "ID": "dataset_id_7",
     "Fault Name": "GeneratorDataset使用自定义的Sampler错误",
     "Key Log Information": "has no attribute .num_samples.",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "自定义的Sampler在__init__中没有显式调用父类dataset.Sampler的构造函数，导致丢失父类中的num_samples",
     "Modification Suggestion": "在__init__函数中调用父类dataset.Sampler的构造函数super().__init__()",
     "Fault Case": "1. Sampler特性说明： "
                   "https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/sampler.html "
                   "2. Sampler报错案例： "
                   " https://bbs.huaweicloud.com/forum/thread-184010-1-1.html#pid1439794"},
    {
     "ID": "dataset_id_8",
     "Fault Name": "自定义Dataset返回数据格式错误",
     "Key Log Information": "Unexpected error. Invalid data type",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "自定义Dataset或者map操作需要返回值类型不是numpy array 或 numpy array 组成的元组，引起报错.",
     "Modification Suggestion": "1. 自定义的数据处理部分是否返回了dict等，要求返回值类型numpy array; "
                                "2. 确认Numpy array其数据类型，保证数据类型为int, float, str.",
     "Fault Case": "1. 自定义数据集加载教程："
                   "https://mindspore.cn/docs/programming_guide/zh-CN/r1.6/dataset_loading.html "
                   "2. 自定义数据集报错案例："
                   "https://bbs.huaweicloud.com/forum/forum.php?mod=viewthread&tid=183190; "
                   "https://bbs.huaweicloud.com/forum/thread-184026-1-1.html; "
                   "https://bbs.huaweicloud.com/forum/thread-168965-1-1.html"},
    {
      "ID": "dataset_id_9",
      "Fault Name": "自定义Iterable dataset索引报错",
      "Key Log Information": "list index out of range",
      "Key Python Stack Information": "",
      "Key C++ Stack Information": "",
      "Fault Cause": "用户可能使用自定义索引计数，迭代前未正确复位造成数组越界报错",
      "Modification Suggestion": "移除非必要的内部索引计数变量，或者在每次迭代前对索引计数赋值为0",
      "Fault Case": "1. Iterable dataset使用示例："
                    "https://mindspore.cn/docs/api/zh-CN/r1.6/api_python/dataset/mindspore.dataset.GeneratorDataset.html "
                    "2. 索引报错案例：https://bbs.huaweicloud.com/forum/thread-184036-1-1.html"},
    {
      "ID": "dataset_id_10",
      "Fault Name": "自定义Dataset使用ops算子报错",
      "Key Log Information": ".*Tensor.*the type of .* should be one of .*, but got ",
      "Key Python Stack Information": "",
      "Key C++ Stack Information": "",
      "Fault Cause": "自定义dataset数据处理不支持ops算子，不支持Tensor数据",
      "Modification Suggestion": "ops算子修改为对应的numpy算子，如ops.Stack->numpy.stack",
      "Fault Case": "1. 自定义数据集加载："
                    "https://www.mindspore.cn/doc/programming_guide/zh-CN/r1.2/dataset_loading.html "
                    "2. 自定义数据集报错案例："
                    "https://bbs.huaweicloud.com/forum/thread-133233-1-1.html"},
    {
     "ID": "dataset_id_11",
     "Fault Name": "自定义Dataset数据类型报错",
     "Key Log Information": "The pointer[cnode] is null",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "数据处理可能使用了底层算子进行执行，由于数据集处理为并行，当前Tensor的操作不支持并行，造成报错.",
     "Modification Suggestion": "用户自定义的方法中，先把输出Tensor转为为Numpy类型，再通过Numpy相关操作实现相关功能.",
     "Fault Case": "1. 常用数据集加载方法："
                   " https://www.mindspore.cn/doc/programming_guide/zh-CN/r1.6/dataset_loading.html "
                   "2. 数据集加载报错案例："
                   " https://bbs.huaweicloud.com/forum/thread-183191-1-1.html "},
    {
     "ID": "dataset_id_12",
     "Fault Name": "自定义数据增强函数输出参数类型报错",
     "Key Log Information": "PyFunc should return a numpy array or a numpy array tuple",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "自定义数据增强函数数据类型报错，一般情况下输出数据类型是numpy array或numpy array tuple",
     "Modification Suggestion": "检查自定义增强函数输出参数的类型，判断是否满足要求",
     "Fault Case": "https://bbs.huaweicloud.com/forum/thread-183196-1-1.html"},
    {
     "ID": "dataset_id_13",
     "Fault Name": "自定义python function数据增强函数的输入参数类型报错",
     "Key Log Information": "Caught IndexError in map\(or batch\) worker",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "自定义python数据增强函数的输入参数类型报错，一般情况下输入数据类型是np.array",
     "Modification Suggestion": "检查自定义数据增强函数的输入参数的类型与个数，判断是否满足要求",
     "Error Case": """
                   class py_func(): 
                     def __call_(self, input_a, input_b):
                         ...
                         return out_a, out_b

                    dataset = datset.map(operations=py_func(), 
                              input_columns=['data','label'])""",
     "Fault Case": "1. 自定义数据处理： "
                   "https://www.mindspore.cn/tutorials/zh-CN/r1.8/advanced/dataset/transform.html"
                   "2. 自定义图像增强报错示例： "
                   "https://bbs.huaweicloud.com/forum/thread-183196-1-1.html"},
    {
     "ID": "dataset_id_14",
     "Fault Name": "数据集处理内存不足",
     "Key Log Information": "Memory not enough: current free memory size.*",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "内存不足引起报错",
     "Modification Suggestion": "1.使用Dataset处理数据过程中内存占用高，参考案例进行优化； "
                                "2.避免网络中过多时候临时变量，临时变量在训练过程中消耗更多内存。",
     "Fault Case": "1. Dataset 内存优化教程: "
                   "https://bbs.huaweicloud.com/forum/thread-182342-1-1.html"},
    {
     "ID": "dataset_id_15",
     "Fault Name": "自定义python function数据增强函数的输出参数错误",
     "Key Log Information": "map operation. \[PyFunc\] failed. Failed to create tensor from numpy array",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "自定义数据处理方法的返回值类型错误，一般输出结果是np.array的列表",
     "Modification Suggestion": "检查自定义数据增强函数的输出参数的类型与个数，判断是否满足要求",
     "Error Case": """
                   class py_func(): 
                     def __call_(self, input_a, input_b):
                         ...
                         return out_a, out_b

                    dataset = datset.map(operations=py_func(), 
                              input_columns=['data','label'])""",
     "Fault Case": "1. 自定义数据处理： "
                   "https://www.mindspore.cn/tutorials/zh-CN/r1.8/advanced/dataset/transform.html#map "
                   "2. 轻量化数据处理： "
                   "https://www.mindspore.cn/tutorials/experts/zh-CN/r1.8/dataset/eager.html "}
]

data_general_experience_list_cn = [
    {
        "ID": "dataset_g_id_1",
        "Fault Name": "数据处理报错",
        "Key Log Information": "",
        "Key Python Stack Information": "",
        "Key C++ Stack Information": "",
        "Code Path": "mindspore.ccsrc.minddata.dataset.engine.datasetops",
        "Fault Cause": "数据处理操作引起报错，可能是shuffle, batch, repeat, zip, concat, map.",
        "Modification Suggestion": "根据报错内容，确认引起报错的数据处理操作，然后查看对应操作的使用说明。",
        "Fault Case": "1. MinSpore数据处理操作： "
                      "https://www.mindspore.cn/tutorials/zh-CN/r1.8/advanced/dataset/transform.html "
                      "2. MindSpore数据处理流程： "
                      "https://www.mindspore.cn/tutorials/zh-CN/r1.8/beginner/dataset.html "
                      "3. 自定义图像增强报错示例： "
                      "https://bbs.huaweicloud.com/forum/thread-127537-1-1.html"},
    {
        "ID": "dataset_g_id_2",
        "Fault Name": "数据处理报错",
        "Key Log Information": "",
        "Key Python Stack Information": "",
        "Key C++ Stack Information": "",
        "Code Path": "mindspore/ccsrc/minddata/dataset/engine/datasetops",
        "Fault Cause": "数据处理操作引起报错，可能是shuffle, batch, repeat, zip, concat, map.",
        "Modification Suggestion": "根据报错内容，确认引起报错的数据处理操作，然后查看对应操作的使用说明。",
        "Fault Case": "1. MinSpore数据处理操作： "
                      "https://www.mindspore.cn/tutorials/zh-CN/r1.8/advanced/dataset/transform.html "
                      "2. MindSpore数据处理流程： "
                      "https://www.mindspore.cn/tutorials/zh-CN/r1.8/beginner/dataset.html "
                      "3. 自定义图像增强报错示例： "
                      "https://bbs.huaweicloud.com/forum/thread-127537-1-1.html"},
    {
     "ID": "dataset_g_id_3",
     "Fault Name": "数据集加载与处理故障",
     "Key Log Information": "",
     "Key Python Stack Information": "", "Key C++ Stack Information": "",
     "Code Path":"mindspore/dataset/engine; mindspore/ccsrc/minddata",
     "Fault Cause": "识别到当前报错可能为数据集加载与处理故障。",
     "Modification Suggestion": "请参考MindSpore数据集故障处理指导进行问题排查或者在MindSpore社区求助",
     "Fault Case": "1. 论坛-数据处理与加载： "
                   "https://bbs.huaweicloud.com/forum/forum-1076-2476-1.html"
                   "2. Dataset FAQ: "
                   "https://www.mindspore.cn/docs/zh-CN/r1.8/faq/data_processing.html"
                   "3. Dataset 错误分析: "
                   "https://www.mindspore.cn/tutorials/experts/zh-CN/r1.8/debug/error_analyze.html"}
]
