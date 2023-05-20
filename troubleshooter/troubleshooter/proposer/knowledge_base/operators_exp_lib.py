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

operators_experience_list_cn = [
    {
     "ID": "operator_id_1",
     "Fault Name": "Conv2D算子参数错误",
     "Key Log Information": "For primitive.*, the x shape size must be equal to .*, but got .*",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "weight_init参数与Conv2D算子输入参数shape不匹配引起报错.",
     "Error Case": """
                    net = nn.Conv2d(120, 240, 4)                           
                    x=Tensor(np.ones([120,480,640]),ms.float32)
                    output=net(x)
                               ^~~~~~ 输入参数的Shape不是NCHW和NHWC""",
     "Modification Suggestion": "检查Conv2D算子输入参数x的shape，保证参数shape满足NHWC或者NCHW格式.",
     "Fixed Case": """
                    net = nn.Conv2d(120, 240, 4)                              
                    x=Tensor(np.ones([1,120,480,640]),ms.float32)
                    output = net(x)
                                 ^~~~~~ 输入参数的Shape符合NCHW格式""",
     "Fault Case": "1. Conv2d API: "
                   "https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/nn/mindspore.nn.Conv2d.html "
                   "2. Conv2d算子报错案例： "
                   "https://bbs.huaweicloud.com/forum/thread-182006-1-1.html"},
    {
     "ID": "operator_id_2",
     "Fault Name": "Pad算子参数错误",
     "Key Log Information": "all elements of .*paddings.* must be >= 0.*",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "Pad算子参数paddings不支持负数",
     "Error Case": """
                    paddings=((1, 1), (-1, 2))
                                        ^~~~~~~~ nn.Pad不支持负数索引
                    net = nn.Pad(paddings, mode="SYMMETRIC")
                    data = np.array([[1, 2, 3], [4, 5, 6]])
                    x = Tensor(data, mindspore.float32)
                    print("x=",x.shape)
                    y = net(x)
                    print(y.shape)""",
     "Modification Suggestion": "修改Pad算子的参数paddings，避免使用负数",
     "Fixed Case": """
                    paddings=((1, 1), (2, 2))
                                       ^~~~~~~~使用正整数索引
                    net = nn.Pad(paddings, mode="SYMMETRIC")
                    data = np.array([[1, 2, 3], [4, 5, 6]])
                    x = Tensor(data, mindspore.float32)
                    print("x=",x.shape)
                    y = net(x)
                    print(y.shape)""",
     "Fault Case": "1. nn.Pad API: " 
                   "https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/nn/mindspore.nn.Pad.html "
                   "2. nn.Pad 报错案例： "
                   "https://bbs.huaweicloud.com/forum/thread-182187-1-1.html"},
    {
     "ID": "operator_id_3",
     "Fault Name": "Pad算子参数错误-1.8",
     "Key Log Information": "the .paddings.shape. must be int and must =",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "Pad算子的参数paddings的维度与输入x的维度不一致。",
     "Error Case": """
                    paddings=((1, 1), (2, 2))
                    pad = nn.Pad(paddings, mode="SYMMETRIC")
                    data = np.array([[[[1, 2, 3], [4, 5, 6]]]])
                    print(data.shape)
                    x = Tensor(data, mindspore.float32)
                    print("x=",x.shape)
                    y = pad(x)
                            ^~~~~~~~~~~~~paddings的维度与输入x的维度不匹配
                    print(y.shape)""",
     "Modification Suggestion": "修改Pad算子的参数paddings或者输入参数x，保证shape与输入参数x一致。",
     "Fixed Case":"""
                    paddings=((0,0),(0,0),(1,1),(2,2))
                               ^~~~~~^~~~~~表示前两个维度不进行pad
                    pad = nn.Pad(paddings, mode="SYMMETRIC")
                    data = np.array([[[[1, 2, 3], [4, 5, 6]]]])
                    x = Tensor(data, mindspore.float32)
                    x = x.squeeze()
                    print("x=",x.shape)
                    y = pad(x)
                    print(y.shape)""",
     "Fault Case": "1. nn.Pad API: "
                   "https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/nn/mindspore.nn.Pad.html "},
    # Different versions have different error descriptions, and the ID is used repeatedly operator_id_3
    {
     "ID": "operator_id_3",
     "Fault Name": "Pad算子参数错误 > 1.8",
     "Key Log Information": "paddings.shape.* must equal to input",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "Pad算子的参数paddings的维度与输入x的维度不一致。",
     "Error Case": """
                paddings=((1, 1), (2, 2))
                pad = nn.Pad(paddings, mode="SYMMETRIC")
                data = np.array([[[[1, 2, 3], [4, 5, 6]]]])
                print(data.shape)
                x = Tensor(data, mindspore.float32)
                print("x=",x.shape)
                y = pad(x)
                        ^~~~~~~~~~~~~paddings的维度与输入x的维度不匹配
                print(y.shape)""",
     "Modification Suggestion": "修改Pad算子的参数paddings或者输入参数x，保证shape与输入参数x一致。",
     "Fixed Case": """
                paddings=((0,0),(0,0),(1,1),(2,2))
                           ^~~~~~^~~~~~表示前两个维度不进行pad
                pad = nn.Pad(paddings, mode="SYMMETRIC")
                data = np.array([[[[1, 2, 3], [4, 5, 6]]]])
                x = Tensor(data, mindspore.float32)
                x = x.squeeze()
                print("x=",x.shape)
                y = pad(x)
                print(y.shape)""",
     "Fault Case": "1. nn.Pad API: "
                   "https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/nn/mindspore.nn.Pad.html "},
    # only 'paddings' up to 4 dims is supported
    {
     "ID": "operator_id_5",
     "Fault Name": "Pad算子参数错误",
     "Key Log Information": "only 'paddings' up to 4 dims is supported",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "Pad算子的参数paddings的维度超过4维报错。",
     "Error Case": """
                    paddings=((0,0),(0,0),(0,0),(1,1),(2,2))
                    pad = nn.Pad(paddings, mode="SYMMETRIC")
                                    ^~~~~~~paddings维度超过4维
                    data = np.array([[[[[1, 2, 3], [4, 5, 6]]]]])
                    x = Tensor(data, mindspore.float32)
                    print("x=",x.shape)
                    y = pad(x)
                    print(y.shape)""",
     "Modification Suggestion": "修改Pad算子的参数paddings与输入参数x，保证shape不超过4维且保持一致。",
     "Fixed Case":"""
                    paddings=((1,1),(2,2))
                    pad = nn.Pad(paddings, mode="SYMMETRIC")
                                    ^~~~~~~paddings维度为2
                    data = np.array([[1, 2, 3], [4, 5, 6]]])
                    x = Tensor(data, mindspore.float32)
                    print("x=",x.shape)
                    y = pad(x)
                            ^~~~~~~x维度也为2 
                    print(y.shape)""",
     "Fault Case": "1. nn.Pad API: "
                   "https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/nn/mindspore.nn.Pad.html "},
    {
     "ID": "operator_id_6",
     "Fault Name": "ReduceSum 算子编译报错",
     "Key Log Information": "the num of dimensions of input.* should be .* range of .0, 8., but .*",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "算子输入参数的维度超过了限制，最大值支持维度8，实际维度超过8报错.",
     "Error Case": """
                    reducesum = ops.ReduceSum(keep_dims=True)
                    data = np.random.randn(1,1,4,4,4,4,4,4,4,4)
                    x = Tensor(data, mindspore.float32)
                                ^~~~~~~~输入x维度大于8
                    out = reducesum(x, (1,))""",
     "Modification Suggestion": "检查ReduceSum算子输入参数x的维度，保证维度不超过8",
     "Fixed Case": """
                    reducesum = ops.ReduceSum(keep_dims=True)
                    data = np.random.randn(4, 4, 4, 4, 4, 4, 4, 4)
                    x = Tensor(data, mindspore.float32)
                                ^~~~~~~~输入x维度为8
                    out = reducesum(x, (1,))""",
     "Fault Case": "1. ReduceSum API： "
                   "https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.ReduceSum.html "
                   "2. ReduceSum算子参数错误案例： "
                   "https://bbs.huaweicloud.com/forum/thread-182168-1-1.html"},
    {
     "ID": "operator_id_7",
     "Fault Name": "损失函数的输入参数不匹配报错",
     "Key Log Information": "For primitive\[.*Entropy.*\], the x shape.* must be equal to.*",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "损失函数要求logits和labels的shape必须一样，不一样时报错",
     "Modification Suggestion": "修改损失函数的输入，保证输入参数的shape一致。",
     "Fault Case": "1. 损失函数 API:  "
                   "https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/mindspore.ops.html "
                   "2. 损失函数报错案例:  "
                   "https://bbs.huaweicloud.com/forum/forum.php?mod=viewthread&tid=182186"},
    {
     "ID": "operator_id_8",
     "Fault Name": "损失函数的输入参数不匹配报错",
     "Key Log Information": "For '.*Loss', the 'logits_shape' should be .*",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "损失函数要求logits和labels的shape必须一样，不一样时报错",
     "Modification Suggestion": "修改损失函数的输入，保证输入参数的shape一致。",
     "Fault Case": "1. 损失函数 API:  "
                   "https://www.mindspore.cn/docs/en/r1.7/acatpi_python/mindspore.nn.html#loss-function "
                   "2. 损失函数报错案例:  "
                   "https://bbs.huaweicloud.com/forum/forum.php?mod=viewthread&tid=182186"},
    {
     "ID": "operator_id_9",
     "Fault Name": "算子输入参数不匹配报错",
     "Key Log Information": "x.shape and y.shape need to broadcast",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "算子输入参数的shape不同且不包含1或者-1，broadcast报错",
     "Error Case":"""
                    x = Tensor(np.array([2, 3]), mindspore.int32)
                    y = Tensor(np.array([4, 5, 6]), mindspore.int32)
                    output = x - y
                    #          ^~~~~~~减法解析为ops.Sub算子，输入x、y的shape不一致且不能broadcast
                    print(output)""",
     "Modification Suggestion": "1.  检查输入参数shape错误的原因，修改至符合要求. ",
     "Fixed Case":"""
                    x = Tensor(np.array([0，2, 3]), mindspore.int32)
                    y = Tensor(np.array([4, 5, 6]), mindspore.int32)
                    output = x - y
                    #          ^~~~~~~减法解析为ops.Sub算子，输入x、y的shape一致
                    print(output)""",
     "Fault Case": "1. ops.Sub API: "
                   "https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.Sub.html "
                   "2. Sub算子broadcast报错: "
                   "https://bbs.huaweicloud.com/forum/forum.php?mod=viewthread&tid=182004"},
    {
     "ID": "operator_id_10",
     "Fault Name": "view/Reshape算子参数不匹配报错",
     "Key Log Information": "product of the shape of 'input_x' should be equal to",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "调用Reshape算子，或者使用其他API间接调用Reshape算子。如果reshape后Tensor元素总数量与原Tensor的不匹配，则报错。",
     "Error Case": """
                    data = np.random.randint(100, size=(20, 4))
                    x = mindspore.Tensor(data,mindspore.float32)
                    x = x.view((-1,3))
                    #              ^~~~~~~~输入参数x的元素数量并不能被3整除
                    print(x.shape)""",
     "Modification Suggestion": "检查Tensor.view或者ops.Reshape算子的输入参数shape与原Tensor的shape的差异，找出元素总量不匹配的原因，修正至符合要求。",
     "Fixed Case": """
                    data = np.random.randint(100, size=(20, 4))
                    x = mindspore.Tensor(data,mindspore.float32)
                    x = x.view((-1,8))
                    #              ^~~~~~~~输入参数x的元素数量能被8整除
                    print(x.shape)""",
     "Fault Case": "1. Tensor.view API： "
                   "https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.Tensor.html#mindspore.Tensor.view  "
                   "2. Tensor.view 使用错误案例： "
                   "https://bbs.huaweicloud.com/forum/forum.php?mod=viewthread&tid=181982"},
    {
     "ID": "operator_id_11",
     "Fault Name": "view/Reshape算子参数不匹配报错-1.9",
     "Key Log Information": "For .*Reshape.* the product of the 'input_x' shape should be equal to",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "调用Reshape算子，或者使用其他接口（如view）间接调用Reshape算子。如果reshape后Tensor元素总数量与原Tensor的不匹配，则报错。",
     "Error Case": """
                    data = np.random.randint(100, size=(20, 4))
                    x = mindspore.Tensor(data,mindspore.float32)
                    x = x.view((-1,3))
                    #              ^~~~~~~~输入参数x的元素数量并不能被3整除
                    print(x.shape)""",
     "Modification Suggestion": "检查Tensor.view或者ops.Reshape算子的输入参数shape与原Tensor的shape的差异，找出元素总量不匹配的原因，修正至符合要求。",
     "Fixed Case": """
                    data = np.random.randint(100, size=(20, 4))
                    x = mindspore.Tensor(data,mindspore.float32)
                    x = x.view((-1,8))
                    #              ^~~~~~~~输入参数x的元素数量能被8整除
                    print(x.shape)""",
     "Fault Case": "1. Tensor.view API： "
                   "https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.Tensor.html#mindspore.Tensor.view  "
                   "2. Tensor.view 使用错误案例： "
                   "https://bbs.huaweicloud.com/forum/forum.php?mod=viewthread&tid=181982"},
    {
     "ID": "operator_id_12",
     "Fault Name": "view/Reshape算子参数不匹配报错-2.0",
     "Key Log Information": "For .*Reshape.* the product of the 'input_x' shape should be equal to",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "调用Reshape算子，或者使用其他接口（如view）间接调用Reshape算子。如果reshape后Tensor元素总数量与原Tensor的不匹配，则报错。",
     "Error Case": """
                data = np.random.randint(100, size=(20, 4))
                x = mindspore.Tensor(data,mindspore.float32)
                x = x.view((-1,3))
                #              ^~~~~~~~输入参数x的元素数量并不能被3整除
                print(x.shape)""",
     "Modification Suggestion": "检查Tensor.view或者ops.Reshape算子的输入参数shape与原Tensor的shape的差异，找出元素总量不匹配的原因，修正至符合要求。",
     "Fixed Case": """
                data = np.random.randint(100, size=(20, 4))
                x = mindspore.Tensor(data,mindspore.float32)
                x = x.view((-1,8))
                #              ^~~~~~~~输入参数x的元素数量能被8整除
                print(x.shape)""",
     "Fault Case": "1. Tensor.view API： "
                   "https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/Tensor/mindspore.Tensor.view.html  "
                   "2. Tensor.view 使用错误案例： "
                   "https://bbs.huaweicloud.com/forum/forum.php?mod=viewthread&tid=181982"},
    {
     "ID": "operator_id_13",
     "Fault Name": "算子不支持参数类型报错",
     "Key Log Information": "Unsupported parameter type for python primitive, the parameter value is KeywordArg",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "ConvertAbstractToPython",
     "Fault Cause": "算子不支持键值对参数而报错",
     "Error Case": """
                    reduce_mean = ops.ReduceMean(keep_dims=True)
                    data = np.ones((3, 4, 5, 6), dtype=np.float32) 
                    x = Tensor(data, mindspore.float32)
                    out = reduce_mean(x, axis=(2, 3)) 
                    #                    ^~~~~~~~~~输入参数不支持键值对""",
     "Modification Suggestion": "查询算子API接口说明，确认输入参数类型，修改至符合要求. ",
     "Fixed Case": """
                    reduce_mean = ops.ReduceMean(keep_dims=True)
                    data = np.ones((3, 4, 5, 6), dtype=np.float32) 
                    x = Tensor(data, mindspore.float32)
                    out = reduce_mean(x, (2, 3)) 
                    #                    ^~~~~~~~~~输入参数支持tuple类型""",
     "Fault Case": "1.ReduceMean算子输入键值对参数报错： "
                   "https://bbs.huaweicloud.com/forum/forum.php?mod=viewthread&tid=182167 "
                   "2.ReduceMean API: "
                   "https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.ReduceMean.html"},
    {
     "ID": "operator_id_14",
     "Fault Name": "SoftmaxCrossEntropyWithLogits接口使用报错",
     "Key Log Information": "For primitive\[SoftmaxCrossEntropyWithLogits\], the dimension of logits must be equal to ., but got .",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "SoftmaxCrossEntropyWithLogits 只支持2维的输入",
     "Error Case":"""
                    loss=nn.SoftmaxCrossEntropyWithLogits()
                    x = np.array([[2,4,1,4,5],[2,1,2,4,3]])
                    y = np.array([[0,0,0,0,1],[0,0,0,1,0]])
                    logits = Tensor(x, mindspore.float32)
                    labels = Tensor(y.astype(np.float32))
                    print(logits.shape, labels.shape)
                    out = loss(logits, labels)
                    #             ^~~~~~~~^~~~~~~~输入入参数维度，只支持（N,C）格式""",
     "Modification Suggestion": "检查SoftmaxCrossEntropyWithLogits输入参数维度，只支持（N,C）两维输入，可使用Reshape算子进行降维. ",
     "Fixed Case":"""
                    loss=nn.SoftmaxCrossEntropyWithLogits()
                    x = np.array([[2,4,1,4,5],[2,1,2,4,3]])
                    y = np.array([[0,0,0,0,1],[0,0,0,1,0]])
                    logits = Tensor(x, mindspore.float32)
                    labels = Tensor(y.astype(np.float32))
                    print(logits.shape, labels.shape)
                    out = loss(logits, labels)
                    #             ^~~~~~~~^~~~~~~~输入入参数维度，符合（N,C）格式""",
     "Fault Case": "1. SoftmaxCrossEntropyWithLogits API: "
                   "https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/nn/mindspore.nn.SoftmaxCrossEntropyWithLogits.html"},
    {
     "ID": "operator_id_15",
     "Fault Name": "算子选择类型不支持报错",
     "Key Log Information": "Can not select a valid kernel info for .* in .*",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "算子选择因输入参数类型不支持，引起报错",
     "Error Case": """
                    data = np.array([[1.0], [2.0], [4.0]])
                    input_x = Tensor(data, mindspore.float64)
                    input_y = 3.0
                    _pow = ops.Pow()
                    out = _pow(input_x, input_y)
                    #            ^~~~~~~~~~~~~ops.Pow不支持float64类型
                    print(out)""",
     "Modification Suggestion": "找到报错算子，查询算子API说明， 修改算子输入参数类型至符合要求. ",
     "Fixed Case": """
                    data = np.array([[1.0], [2.0], [4.0]])
                    input_x = Tensor(data, mindspore.float32)
                    input_y = 3.0
                    _pow = ops.Pow()
                    out = _pow(input_x, input_y)
                    #            ^~~~~~~~~~~~~ops.Pow支持float32类型
                    print(out)""",
     "Fault Case": "1.ops.Pow API: "
                   "https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.Pow.html"
                   "2.ops.Pow算子报错案例: "
                   "https://bbs.huaweicloud.com/forum/forum.php?mod=viewthread&tid=182027"},
    {
     "ID": "operator_id_16",
     "Fault Name": "算子参数不支持报错",
     "Key Log Information": "Unsupported input type for .*",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "算子不支持当前输入参数类型报错",
     "Error Case": """
                    x = False
                    y = True
                    output = ops.scalar_add(x, y)
                    #                       ^~~~~~~不支持输入参数类型BOOL
                    print('output', output)""",
     "Modification Suggestion": "参考算子接口官方文档，修改输入参数类型",
     "Fixed Case": """
                    x = 0
                    y = 1
                    output = ops.scalar_add(x, y)
                    #                       ^~~~~~~不支持输入参数类型BOOL
                    print('output', output)""",
     "Fault Case": "1.ops.scalar_add API: "
                   "https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/mindspore.ops.html?highlight=scalar_add "
                   "2.ops.scalar_add()算子报错案例: "
                   "https://bbs.huaweicloud.com/forum/forum.php?mod=viewthread&tid=182173"},
    {
     "ID": "operator_id_17",
     "Fault Name": "算子参数不支持报错-r1.6",
     "Key Log Information": "input var is a not support implicit conversion",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "算子输入参数的类型不匹配，且不支持隐式转换报错。",
     "Error Case": """
                    x=Tensor(np.array([1.0,2.0,3.0]), ms.float32)
                    y = np.array([4.0,5.0,6.0])
                    output = x * y
                    #            ^~~~~~乘法解释为ops.Mul算子
                    #            ops.Mul算子不支持np.array
                    print(output)""",
     "Modification Suggestion": "修改算子输入参数类型，通常支持Tensor或Scale",
     "Fixed Case":"""
                    x=Tensor(np.array([1.0,2.0,3.0]), ms.float32)
                    y=Tensor(np.array([4.0,5.0,6.0]), ms.float32)
                    #   ^~~~~~修改参数类型为Tensor
                    output = x * y
                    print(output)""",
     "Fault Case": "1. ops.Mul API: "
                   "https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.Mul.html "},
    {
     "ID": "operator_id_18",
     "Fault Name": "算子参数不支持报错-r1.8",
     "Key Log Information": "input var can not be implicitly converted",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "算子输入参数的类型不匹配，且不支持隐式转换报错。",
     "Error Case": """
                    x=Tensor(np.array([1.0,2.0,3.0]), ms.float32)
                    y = np.array([4.0,5.0,6.0])
                    output = x * y
                    #            ^~~~~~乘法解释为ops.Mul算子
                    #            ops.Mul算子不支持np.array
                    print(output)""",
     "Modification Suggestion": "修改算子输入参数类型，通常支持Tensor或Scale",
     "Fixed Case":"""
                    x=Tensor(np.array([1.0,2.0,3.0]), ms.float32)
                    y=Tensor(np.array([4.0,5.0,6.0]), ms.float32)
                    #   ^~~~~~修改参数类型为Tensor
                    output = x * y
                    print(output)""",
     "Fault Case": "1. ops.Mul API: "
                   "https://www.mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.Mul.html "},
    {
     "ID": "operator_id_19",
     "Fault Name": "算子反向未定义错误",
     "Key Log Information": "Illegal primitive: Primitive .* bprop not defined",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "该算子反向可能没有定义，反向求导时报错",
     "Modification Suggestion": "建议自定义算子反向或者替换为其他算子",
     "Fault Case": "1. 算子自定义反向示例： "
                   "https://www.mindspore.cn/tutorials/experts/zh-CN/r1.7/operation/op_cpu.html#"
                   "2. 算子反向未定义案例： "
                   "https://bbs.huaweicloud.com/forum/thread-182166-1-1.html"}
]
operators_general_experience_list_cn = [
 {
  "ID": "operator_g_id_1",
  "Fault Name": "算子选择错误",
  "Key Log Information": "Can not find any available kernel info;Can not select a valid kernel info",
  "Key Python Stack Information": "",
  "Key C++ Stack Information": "",
  "Code Path":"",
  "Fault Cause": "Ascend设备算子选择出现错误，可能原因是使用的算子不存在或者算子参数/输入配置错误。",
  "Modification Suggestion": "1. 检查报错算子API定义，确认使用方式，输入参数类型及shape格式是否支持;"
                             "2. 对比算子使用示例与自身代码的差异，从不同点查找报错原因.",
  "Fault Case": "1.API接口文档： "
                "https://www.mindspore.cn/docs/zh-CN/r1.8/api_python/mindspore.html "
                "2.算子编译问题 FAQ： "
                "https://www.mindspore.cn/docs/zh-CN/r1.8/faq/operators_compile.html "
 },
 {
  "ID": "operator_g_id_2",
  "Fault Name": "算子使用错误",
  "Key Log Information": "For primitive\[.*\], the .* must be .*;For primitive\[.*\], the .* should be .*;"
                         "For '.*', the .* must be;For '.*', the .* should be",
  "Key Python Stack Information": "mindspore.python.mindspore.ops",
  "Key C++ Stack Information": "mindspore.core.ops",
  "Code Path":"mindspore.core.ops;mindspore.python.mindspore.ops",
  "Fault Cause": "识别当前报错为算子使用错误",
  "Modification Suggestion": "1.  确认报错算子，检查算子使用方式是否正确;  #                 "
                             "2. 检查算子是否支持当前执行模式，如图模式、并行模式等. #              ",
  "Fault Case": "1.MindSpore接口文档： "
                "https://www.mindspore.cn/docs/zh-CN/r1.8/api_python/mindspore.nn.html "
                "2.常见报错案例汇总： "
                "https://bbs.huaweicloud.com/forum/thread-194053-1-1.html "}
]