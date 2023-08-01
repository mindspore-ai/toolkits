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

compiler_experience_list_cn = [
    {
        "ID": "compiler_id_1",
        "Fault Name": "construct 参数错误 <2.0",
        "Key Log Information": "For 'Cell', the function construct .* 0 positional argument and 0 default argument",
        "Key Python Stack Information": "",
        "Key C++ Stack Information": "",
        "Fault Cause": "继承nn.Cell类，重定义construct函数时，缺少self参数报错",
        "Error Case": """
                      def construct(a,b):
                                    ^~~~~~~ 此处缺少了self""",
        "Modification Suggestion": "修改construct定义,添加self参数.",
        "Fixed Case": """
                      def construct(self,a,b):
                                      ^~~~~~~ 添加self""",
        "Fault Case": """1.nn.Cell 官方文档:
                   https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/nn/mindspore.nn.Cell.html
                   2.缺少self参数报错案例:
                   https://bbs.huaweicloud.com/forum/thread-178902-1-1.html""",
        "Test Case": "test_compiler_no_self.py"},
    # Different versions have different error descriptions, and the ID is used repeatedly
    {
        "ID": "compiler_id_1",
        "Fault Name": "construct 参数错误 >2.0",
        "Key Log Information": "For 'Cell', the method 'construct' must have parameter 'self'",
        "Key Python Stack Information": "",
        "Key C++ Stack Information": "",
        "Fault Cause": "继承nn.Cell类，重定义construct函数时，缺少self参数报错",
        "Error Case": """
                  def construct(a,b):
                                ^~~~~~~ 此处缺少了self""",
        "Modification Suggestion": "修改construct定义,添加self参数.",
        "Fixed Case": """
                  def construct(self,a,b):
                                  ^~~~~~~ 添加self""",
        "Fault Case": """1.nn.Cell 官方文档:
               https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/nn/mindspore.nn.Cell.html
               2.缺少self参数报错案例:
               https://bbs.huaweicloud.com/forum/thread-178902-1-1.html""",
        "Test Case": "test_compiler_no_self.py"},

    {
         "ID": "compiler_id_2",
         "Fault Name":"construct 参数错误",
         "Key Log Information":"For 'Cell', the function construct .* positional argument, but got .*",
         "Key Python Stack Information":"",
         "Key C++ Stack Information":"",
         "Fault Cause":"继承nn.Cell类，重定义construct函数时，定义参数个数与调用时输入参数不匹配报错",
         "Error Case":"""
                         class Net_LessInput(Cell):
                           def construct(self,x,y):
                              return x + y
                         net = Net_LessInput()
                         out = net(1)
                                   ^~~~~~~ 此处缺少1个输入参数""",
         "Modification Suggestion":"检查construct函数定义参数与调用输入参数，如果参数不匹配，"
                                   "请根据情况修改参数定义或输入参数。",
         "Fixed Case":"""
                        class Net_LessInput(Cell):
                         def construct(self,x,y):
                             return x + y
                        net = Net_LessInput()
                        out = net(1,2)
                                   ^~~~~~~ 此处增加1个输入参数""",
         "Fault Case":"""1.nn.Cell官方文档：
                      https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/nn/mindspore.nn.Cell.html
                      2.参考缺少self参数报错案例：
                      https://bbs.huaweicloud.com/forum/thread-178902-1-1.html"""},
    {
        "ID": "compiler_id_3",
        "Fault Name":"construct 参数错误",
        "Key Log Information":"construct\(\) missing [1-9] required positional argument.*",
        "Key Python Stack Information":"",
        "Key C++ Stack Information":"",
        "Fault Cause":"继承nn.Cell类，重定义construct函数时，定义参数个数与调用时输入参数不匹配报错",
        "Error Case":"""
                     class Net_LessInput(Cell):
                       def construct(self,x,y):
                          return x + y
                     net = Net_LessInput()
                     out = net(1)
                               ^~~~~~~ 此处缺少1个输入参数""",
        "Modification Suggestion":"检查construct函数定义参数与调用输入参数，如果参数不匹配，"
                                  "请根据情况修改参数定义或输入参数。",
        "Fixed Case":"""
                    class Net_LessInput(Cell):
                     def construct(self,x,y):
                         return x + y
                    net = Net_LessInput()
                    out = net(1,2)
                               ^~~~~~~ 此处增加1个输入参数""",
        "Fault Case":"""1.nn.Cell官方文档：
                  https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/nn/mindspore.nn.Cell.html
                  2.参考缺少self参数报错案例：
                  https://bbs.huaweicloud.com/forum/thread-178902-1-1.html"""},
    {
         "ID": "compiler_id_4",
         "Fault Name":"construct 参数错误",
         "Key Log Information":"For 'Cell', the function construct .* [1-9] positional argument and "
                             "[0-9] default argument, total .*",
         "Key Python Stack Information":"",
         "Key C++ Stack Information":"",
         "Fault Cause":"继承nn.Cell类，重定义construct函数时，定义参数个数与调用时输入参数不匹配报错",
         "Error Case":"""
                    class Net_MoreInput(Cell):
                       def construct(self,x):
                           return x
                     net = Net_MoreInput()
                     out = net(1, 2)
                                  ^~~~~~~ 此处多1个输入参数""",
         "Modification Suggestion":"检查construct函数定义参数与调用输入参数，如果参数不匹配，请根据情况修改参数定义或输入参数。",
         "Fixed Case":"""
                     class Net_MoreInput(Cell):
                       def construct(self,x):
                           return x
                     net = Net_MoreInput()
                     out = net(1)
                               ^~~~~~~ 此处去掉1个输入参数""",
         "Fault Case":"""1.nn.Cell官方文档:
                  https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/nn/mindspore.nn.Cell.html
                  2.参考缺少self参数报错案例:
                  https://bbs.huaweicloud.com/forum/thread-178902-1-1.html"""},
    {
        "ID": "compiler_id_5",
        "Fault Name":"construct 参数错误",
        "Key Log Information":"construct\(\) takes [1-9] positional arguments but [0-9] were given",
        "Key Python Stack Information":"",
        "Key C++ Stack Information":"",
        "Fault Cause":"继承nn.Cell类，重定义construct函数时，定义参数个数与调用时输入参数不匹配报错",
        "Error Case":"""
                class Net_MoreInput(Cell):
                   def construct(self,x):
                       return x
                 net = Net_MoreInput()
                 out = net(1, 2)
                              ^~~~~~~ 此处多1个输入参数""",
        "Modification Suggestion":"检查construct函数定义参数与调用输入参数，如果参数不匹配，请根据情况修改参数定义或输入参数。",
        "Fixed Case":"""
                 class Net_MoreInput(Cell):
                   def construct(self,x):
                       return x
                 net = Net_MoreInput()
                 out = net(1)
                           ^~~~~~~ 此处去掉1个输入参数""",
        "Fault Case":"""1.nn.Cell官方文档:
              https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/nn/mindspore.nn.Cell.html
              2.参考缺少self参数报错案例:
              https://bbs.huaweicloud.com/forum/thread-178902-1-1.html"""},
    {
     "ID": "compiler_id_6",
     "Fault Name": "自定义接口参数错误",
     "Key Log Information": ".* parameters number of the function is .*but .*",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "自定义函数参数定义列表，与函数输入参数列表不匹配引起报错",
     "Error Case": """
                    class Net(nn.Cell):
                        ...
                        def func(x, y):
                            return self.div(x, y)
                        def construct(self, x, y):
                            a=self.sub(x, 1)
                            b=self.add(a, y)
                            c=self.mul(self.func(a,a,b),b)
                                            ^~~~~~~~~~参数与函数定义不匹配
                            return c""",
     "Modification Suggestion": "参考函数定义，修改自定义函数调用参数",
     "Fixed Case": """
                    class Net(nn.Cell):
                        ...
                        def func(x, y):
                            return self.div(x, y)
                        def construct(self, x, y):
                            a=self.sub(x, 1)
                            b=self.add(a, y)
                            c=self.mul(self.func(a,b),b)
                                            ^~~~~~~~~~修改参数与函数定义匹配
                            return c""",
     "Fault Case": """1.使用analyze_fail.dat分析函数参数错误案例:
                     https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/read_ir_files.html#analyze-fail-dat"""},
    {
        "ID": "compiler_id_7",
        "Fault Name": "自定义接口参数错误",
        "Key Log Information": ".* takes . positional arguments but . were given",
        "Key Python Stack Information": "",
        "Key C++ Stack Information": "",
        "Fault Cause": "自定义函数参数定义列表，与函数输入参数列表不匹配引起报错",
        "Error Case": """
                    class Net(nn.Cell):
                        ...
                        def func(x, y):
                            return self.div(x, y)
                        def construct(self, x, y):
                            a=self.sub(x, 1)
                            b=self.add(a, y)
                            c=self.mul(b,self.func(a,a,b))
                                            ^~~~~~~~~~参数与函数定义不匹配
                            return c""",
        "Modification Suggestion": "参考函数定义，修改自定义函数调用参数.",
        "Fixed Case": """
                    class Net(nn.Cell):
                        ...
                        def func(x, y):
                            return self.div(x, y)
                        def construct(self, x, y):
                            a=self.sub(x, 1)
                            b=self.add(a, y)
                            c=self.mul(b, self.func(a,b))
                                            ^~~~~~~~~~修改参数与函数定义匹配
                            return c""",
        "Fault Case":"1.使用analyze_fail.dat分析函数参数错误案例: "
                     "https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/read_ir_files.html#analyze-fail-dat"},
    {
     "ID": "compiler_id_8",
     "Fault Name": "抽象类型合并失败",
     "Key Log Information": "Type Join Failed: dtype1 = .*, dtype2 = .*",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "静态图的控制流(例如:if...else...)语法要求不同分支返回值的类型必须相同，不相同时(例如：Float32与Float16)会报错。",
     "Error Case" : """
                       def construct(self, x, a, b):
                         if a > b:
                           return self.relu(x)
                         else:
                           # dtype:Float32 -> Float16
                         return self.cast(self.relu(x), ms.float16)
                             ^~~~~~~ 返回值类型与if分支不一致""",
     "Modification Suggestion": "检查不同分支的返回结果的类型，如果类型不相同，请修改至相同类型。",
     "Fault Case": """1.编译时报错Type Join Failed案例:
                   https://www.mindspore.cn/docs/faq/zh-CN/r1.6/network_compilation.html"""},
    {
     "ID": "compiler_id_9",
     "Fault Name": "抽象类型合并失败",
     "Key Log Information": "Shape Join Failed: shape1 = .*, shape2 = .*",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "静态图的控制流(例如:if...else...)语法要求不同分支返回值的shape相同，不相同时会报错。",
     "Error Case": """
                     def construct(self, x, a, b):
                         if a > b:
                             return self.relu(x)
                         else:
                             # shape: (*)->()
                             return self.reducesum(x)
                                 ^~~~~~~返回值shape与if分支不一致 """,
     "Modification Suggestion": "检查不同分支的返回结果的shape，如果不相同，请修改至相同shape",
     "Fault Case": """1.编译时报错Shape Join Failed案例:
                   https://www.mindspore.cn/docs/faq/zh-CN/r1.6/network_compilation.html
                   2.静态图语法：
                   https://www.mindspore.cn/docs/note/zh-CN/r1.6/static_graph_syntax_support.html"""},
    {
     "ID": "compiler_id_10",
     "Fault Name": "抽象类型合并失败",
     "Key Log Information": "Abstract type .* cannot join with .*",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "对函数输出求梯度时，抽象类型不匹配，导致抽象类型合并失败",
     "Error Case": """
                    grad = ops.GradOperation(sens_param=True)
                    # test_net输出类型为tuple(Tensor, Tensor)
                    def test_net(a, b):
                      return a, b
                    _type=ops.DType()
                    _shape=ops.Shape()
                    @ms_function()
                    def join_fail(x, y):
                      sens_i=ops.Fill()(_type(x),_shape(x),1.0)
                      a=grad(test_net)(x,y,sens_i)
                                           ^~~~sens_i类型为Tensor
                      return a

                    x = Tensor([1.0])
                    y = Tensor([2.0])
                    join_fail(x,y)""",
     "Modification Suggestion": "检查求梯度的函数的输出类型与sens_param的类型是否相同，如果不相同，修改为相同类型.",
     "Fixed Case":"""
                  grad = ops.GradOperation(sens_param=True)
                  # test_net输出类型为tuple(Tensor, Tensor)
                  def test_net(a, b):
                    return a, b
                  _type=ops.DType()
                  _shape=ops.Shape()
                  @ms_function()
                  def join_fail(x, y):
                    sens_i=ops.Fill()(_type(x),_shape(x),1.0)
                    sens=(sens_i, sens_i) #
                    a=grad(test_net)(x,y,sens)
                                         ^~~~sens类型为tuple
                    return a
                  x = Tensor([1.0])
                  y = Tensor([2.0])
                  join_fail(x,y)""",
     "Fault Case": """1.自动求导报错Type Join Failed案例:
               https://www.mindspore.cn/docs/faq/zh-CN/r1.6/network_compilation.html""" },
    {
     "ID": "compiler_id_11",
     "Fault Name": "静态图找不到属性-1.8.1",
     "Key Log Information": "operation does not support the type",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "GenerateFromTypes",
     "Fault Cause": "静态图语法使用未定义的变量报错",
     "Error Case": """
                      class Net(nn.Cell):
                        def construct(self, x):
                          return x + self.y
                                        ^~~~~~~ self.y未定义""",
     "Modification Suggestion": "定位到报错代码行，如果变量未定义，可以在__init__方法中初始化。",
     "Fixed Case": """
                      class Net(nn.Cell):
                        def __init__(self):
                          super(Net, self).__init__()
                          self.y = 1.0
                             ^~~~~~~ 对self.y初始化
                        def construct(self, x):
                          return x + self.y""",
     "Fault Case": "1.静态图语法： "
                   "https://www.mindspore.cn/docs/note/zh-CN/r1.6/static_graph_syntax_support.html "
                   "2.使用未定义变量报错案例： "
                   "https://bbs.huaweicloud.com/forum/thread-180857-1-1.html",
     "Test Case": "test_compiler_kmetatypenone.py"},
    {
        "ID": "compiler_id_12",
        "Fault Name": "静态图找不到属性-1.9",
        "Key Log Information": "External object has no attribute",
        "Key Python Stack Information": "",
        "Key C++ Stack Information": "GetEvaluatedValueForNameSpaceString",
        "Fault Cause": "静态图语法使用未定义的变量报错",
        "Error Case": """
                  class Net(nn.Cell):
                    def construct(self, x):
                      return x + self.y
                                    ^~~~~~~ self.y未定义""",
        "Modification Suggestion": "定位到报错代码行，如果变量未定义，可以在__init__方法中初始化。",
        "Fixed Case": """
                  class Net(nn.Cell):
                    def __init__(self):
                      super(Net, self).__init__()
                      self.y = 1.0
                         ^~~~~~~ 对self.y初始化
                    def construct(self, x):
                      return x + self.y""",
        "Fault Case": "1.静态图语法： "
                      "https://www.mindspore.cn/docs/note/zh-CN/r1.6/static_graph_syntax_support.html "
                      "2.使用未定义变量报错案例： "
                      "https://bbs.huaweicloud.com/forum/thread-180857-1-1.html",
        "Test Case": "test_compiler_kmetatypenone.py"},
    {
     "ID": "compiler_id_13",
     "Fault Name": "函数调用栈超限",
     "Key Log Information": "Exceed function call depth limit",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "函数嵌套调用存在死循环或者嵌套调用深度超过限制而引起报错",
     "Modification Suggestion": "1.检查代码中是否存在无穷递归或死循环，简化代码逻辑减少函数循环嵌套调用 "
                                "2.使用context.set_context(max_call_depth=value)方法，修改函数调用栈最大深度限制。",
     "Fault Case": "1.GRU算子参数过大引起栈超限制案例: "
                   "https://bbs.huaweicloud.com/forum/thread-182256-1-1.html"},
    {
     "ID": "compiler_id_14",
     "Fault Name": "JIT Fallback 使用错误",
     "Key Log Information": "not use Python object in runtime",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "JIT Fallback暂不支持Runtime阶段解析执行代码",
     "Error Case": """
                    @ms_function
                    def test_np_add():
                        x = np.array([1, 2, 3, 4, 5])
                        y = np.array([1, 2, 3, 4, 5])
                        return np.add(x, y)
                                    ^~~~~~不支持的语法作为函数返回值报错""",
     "Modification Suggestion": "静态图模式下不支持的语法，将会生成解释节点，避免将解释节点传递到运行时。",
     "Fixed Case":"""
                    @ms_function
                    def test_np_add():
                        x = np.array([1, 2, 3, 4, 5])
                        y = np.array([1, 2, 3, 4, 5])
                        z = Tensor(np.add(x, y))
                                      ^~~~不支持的语法，在编译期执行。
                        return z""",
     "Fault Case": "1.JIT Fallback使用说明： "
                   "https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/jit_fallback.html"},
    {
     "ID": "compiler_id_15",
     "Fault Name": "使用自定义class的属性与方法-1.8",
     "Key Log Information": "Not supported to get attribute for InterpretedObject.*",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "静态图语法不支持自定义class的属性与方法， 在construct函数中使用出现报错",
     "Error Case":"""
                    class LayerParams:
                        def get_weights(self, shape):
                            ...
                            nn_param=mindspore.Parameter(nn_param)
                            return nn_param
                    class MyCell(nn.Cell):
                        def __init__(self):
                            super().__init__()
                            self._params=LayerParams(...)
                        def _fc(self, inputs, output_size):
                            width=inputs.shape[-1]
                            weight=self._params.get_weights(...)
                                                ^~~~~~~~~~~不支持
                            return weight
                    def construct(self, x, output_size):
                        weight = self._fc(x, output_size)
                        ...""",
     "Modification Suggestion": "当前版本不支持使用自定义类的属性与方法，请修改自定义类方法，或尝试使用最新版本。",
     "Fault Case": "1.静态图语法支持： "
                   "https://www.mindspore.cn/docs/note/zh-CN/r1.6/static_graph_syntax_support.html"},
    {
        "ID": "compiler_id_16",
        "Fault Name": "使用自定义class的属性与方法-1.9",
        "Key Log Information": "Do not support to get attribute from .* object .*",
        "Key Python Stack Information": "",
        "Key C++ Stack Information": "StaticGetter",
        "Fault Cause": "静态图语法不支持调用自定义class的属性与方法， 在construct函数中使用出现报错",
        "Error Case": """
                class LayerParams:
                    def get_weights(self, shape):
                        ...
                        nn_param=mindspore.Parameter(nn_param)
                        return nn_param
                class MyCell(nn.Cell):
                    def __init__(self):
                        super().__init__()
                        self._params=LayerParams(...)
                    def _fc(self, inputs, output_size):
                        width=inputs.shape[-1]
                        weight=self._params.get_weights(...)
                                            ^~~~~~~~~~~不支持
                        return weight
                def construct(self, x, output_size):
                    weight = self._fc(x, output_size)
                    ...""",
        "Modification Suggestion": "当前版本不支持使用自定义类的属性与方法，请修改自定义类方法，或尝试使用最新版本。",
        "Fault Case": "1.静态图语法支持： "
                      "https://www.mindspore.cn/docs/note/zh-CN/r1.6/static_graph_syntax_support.html"},
    {
     "ID": "compiler_id_17",
     "Fault Name": "静态图代码缩进错误",
     "Key Log Information": "incorrect indentations in definition or comment",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "可能是静态图语法中代码行或者注释的缩进没有对齐，引起报错",
     "Error Case":"""
                class Net(nn.Cell):
                    def __init__(self):
                        super(Net, self).__init__()
                        self.y = 2.0
                    def construct(self, x):
                #       x + 2.0
                ^~~~~~~~~~~~~~注释行没有对齐
                        return x + self.y""",
     "Modification Suggestion": "修改缩进没有对齐的代码行或者注释行，使对齐。",
     "Fixed Case":"""
                class Net(nn.Cell):
                    def __init__(self):
                        super(Net, self).__init__()
                        self.y = 2.0
                    def construct(self, x):
                        # x + 2.0
                        ^~~~~~~~~~~~~~修改注释行对齐
                        return x + self.y""",
     "Fault Case": """1.静态图语法： 
                   https://www.mindspore.cn/docs/note/zh-CN/r1.6/static_graph_syntax_support.html"""},
    {
     "ID": "compiler_id_18",
     "Fault Name": "Parameter重名报错",
     "Key Log Information":"value.* Parameter (.*) , its name .* already exists.",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "在一个神经网络内，Parameter出现了重名参数而引起报错，需要修改参数名称。",
     "Modification Suggestion":"1.如果是一个网络有多个参数Parameter对象，需要修改重名的Parameter的名称； "
                               "2.如果是一个网络使用PrameterTuple时，需要指定其内部的Parameter为不同名称； "
                               "3.如果是2个或以上的网络使用了PrameterTuple, 需要使用CellList来避免重名; "
                               "4.如果存在多个Cell，可以使用Cell.update_parameters_name()增加参数名称前缀，避免重名。",
      "Fault Case":"1. mindspore.Parameter接口: "
                   "https://www.mindspore.cn/docs/zh-CN/r1.8/api_python/mindspore/mindspore.Parameter.html "
                   "2. mindspore.PrameterTuple接口： "
                   "https://www.mindspore.cn/docs/zh-CN/r1.8/api_python/mindspore/mindspore.ParameterTuple.html "},
    {
        "ID": "compiler_id_19",
        "Fault Name": "静态图模式下使用了错误的框架API",
        "Key Log Information": "Unsupported statement \'Try\'",
        "Key Python Stack Information": "",
        "Key C++ Stack Information": "ParseStatement",
        "Fault Cause": "静态图模式下网络的construct函数、@ms_function修饰的函数或@ms.jit修饰的函数中出现如下问题： "
                       "1）直接或者间接使用了try语法，此语法静态图模式不支持; "
                       "2）直接或间接的调用了MindSpore的get_context接口（get_context接口会使用try语法），此接口不能在construct "
                       "等函数中调用; ",
        "Modification Suggestion": "1）找到函数中直接或者间接使用了try语法, 将try语法在函数中移除; "
                                   "2）函数中调用的get_context()接口，迁移到函数外调用; "
                                   "3）如果没发现调用函数位置，则可能为函数多级调用或者框架内部多级调用，可去社区提单求助",
        "Fault Case": "1.静态图语法支持： "
                      "https://www.mindspore.cn/docs/note/zh-CN/r1.6/static_graph_syntax_support.html "},
    {
         "ID": "compiler_id_20",
         "Fault Name": "静态图自定义类语法",
         "Key Log Information": "initializing input to create instance for MsClassObject.*?should be a constant",
         "Key Python Stack Information": "",
         "Key C++ Stack Information": "",
         "Fault Cause": "静态图模式下，不支持在construct中用动态输入去初始化自定义类，创建自定义类的实例时，参数必须是常量。",
         "Error Case": """
                    class Net(nn.Cell):
                        def construct(self, x):
                            net = InnerNet(x)
                                           ^~~~~~~~~~~~~~不支持
                            return net.number""",
         "Modification Suggestion": "在__init__中创建自定义类的实例",
         "Fixed Case": """
                    class Net(nn.Cell):
                        def __init__(self):
                            super(Net, self).__init__()
                            self.inner = InnerNet(val)
                              ^~~~~~~~~~~~~~在__init__中创建实例
                        def construct(self, x):
                            return self.inner.number""",
         "Fault Case": """1.静态图语法： 
                       https://www.mindspore.cn/tutorials/experts/zh-CN/r2.0.0-alpha/network/jit_class.html#%E5%88%9B%E5%BB%BA%E8%87%AA%E5%AE%9A%E4%B9%89%E7%B1%BB%E7%9A%84%E5%AE%9E%E4%BE%8B"""},
    {
         "ID": "compiler_id_21",
         "Fault Name": "静态图变量未赋初值报错",
         "Key Log Information": "The name.*?is not defined, or not supported in graph mode",
         "Key Python Stack Information": "",
         "Key C++ Stack Information": "",
         "Fault Cause": "静态图模式下，需要保证变量在每一个分支都有定义，因此建议在if判断前给变量赋一个初始值。",
         "Error Case": """
                    class Net(nn.Cell):
                        def construct(self, x):
                            if condition:
                                var = (1, 2, 3)
                            for i in var:
                                      ^~~~~~~~~~~~~~未给变量var赋一个初始值
                                print(i)
                            return True""",
         "Modification Suggestion": "在if判断条件前给变量赋一个初始值。",
         "Fixed Case": """
                    class Net(nn.Cell):
                        def construct(self, x):
                            var = ()
                            ^~~~~~~~~~~~~~在if判断前给变量var赋一个初值
                            if condition:
                                var = (1, 2, 3)  
                            for i in var:
                                print(i)
                            return True""",
         "Fault Case": """1.静态图语法： 
                       https://www.mindspore.cn/tutorials/experts/zh-CN/r2.0.0-alpha/network/jit_class.html#%E5%88%9B%E5%BB%BA%E8%87%AA%E5%AE%9A%E4%B9%89%E7%B1%BB%E7%9A%84%E5%AE%9E%E4%BE%8B"""},
    {
         "ID": "compiler_id_22",
         "Fault Name": "静态图Tensor索引报错",
         "Key Log Information": "switch_layer index must be an int32, but got Int64",
         "Key Python Stack Information": "",
         "Key C++ Stack Information": "",
         "Fault Cause": "静态图模式下，索引值为Tensor有限制。索引Tensor是一个dtype为int32的标量",
         "Error Case": """
                    @jit
                    def index_get(y):
                        return x[y]
                                 ^~~~~~~~~~~~~~索引值为int32的标量Tensor
                    x = (Tensor(4), Tensor(5), Tensor(6))
                    y = Tensor(2)
                    out = index_get(y)""",
         "Modification Suggestion": "改用索引值为scalar。",
         "Fixed Case": """
                   @jit
                    def index_get(y):
                        return x[y]
                                 ^~~~~~~~~~~~~~改用索引值为scalar。
                    x = (Tensor(4), Tensor(5), Tensor(6))
                    y = 2
                    out = index_get(y)""",
         "Fault Case": """1.静态图Tensor索引语法支持： 
                       https://www.mindspore.cn/docs/zh-CN/master/note/static_graph_syntax_support.html#tuple"""},
    {
         "ID": "compiler_id_23",
         "Fault Name": "静态图Tensor索引报错",
         "Key Log Information": "switch_layer requires that the 2th arg be tuple of functions, but got AbstractTensor",
         "Key Python Stack Information": "",
         "Key C++ Stack Information": "",
         "Fault Cause": "静态图模式下，索引值为Tensor有限制。索引Tensor是一个dtype为int32的标量",
         "Error Case": """
                    @jit
                    def index_get(y):
                        return x[y]
                                 ^~~~~~~~~~~~~~静态图中，索引值为Tensor有限制。
                                               tuple里存放的必须是Cell。
                    x = (Tensor(4), Tensor(5), Tensor(6))
                    y = Tensor(2, mindspore.int32)
                    out = index_get(y)""",
         "Modification Suggestion": "改用索引值为scalar。",
         "Fixed Case": """
                   @jit
                    def index_get(y):
                        return x[y]
                                 ^~~~~~~~~~~~~~改用索引值为scalar。
                    x = (Tensor(4), Tensor(5), Tensor(6))
                    y = 2
                    out = index_get(y)""",
         "Fault Case": """1.静态图Tensor索引语法支持： 
                       https://www.mindspore.cn/docs/zh-CN/master/note/static_graph_syntax_support.html#tuple"""},
    {
         "ID": "compiler_id_24",
         "Fault Name": "静态图语法",
         "Key Log Information": "The local variable.*?is not defined in false branch, but defined in true branch",
         "Key Python Stack Information": "",
         "Key C++ Stack Information": "",
         "Fault Cause": "在静态图模式下，if控制流分支中的变量必须都有定义。",
         "Error Case": """
                    @jit
                    def func(x):
                        if x < 0:
                            y = 0
                            ^~~~~~~~~~~~~~变量y只有true分支有定义，else分支未定义。
                        return y""",
         "Modification Suggestion": "提前定义y，或在else分支定义。",
         "Fixed Case": """
                   @jit
                    def func(x):
                        y = 1
                        ^~~~~~~~~~~~~~提前定义y，或在else分支定义。
                        if x < 0:
                            y = 0
                        return y""",
         "Fault Case": """1.静态图语法： 
                       https://www.mindspore.cn/tutorials/experts/zh-CN/r2.0/network/control_flow.html#shapejoin%E8%A7%84%E5%88%99"""},
    {
         "ID": "compiler_id_25",
         "Fault Name": "静态图语法",
         "Key Log Information": "The local variable.*?defined in the.*?loop body cannot be used outside of the loop body. Please define variable.*?before.*?",
         "Key Python Stack Information": "",
         "Key C++ Stack Information": "",
         "Fault Cause": "在静态图模式下，while/for控制流分支中的变量必须都有定义",
         "Error Case": """
                    @jit
                    def func(x):
                        while x > 0:
                            x -= 1
                            y = 0
                            ^~~~~~~~~~~~~~变量y只有在循环体内有定义，循环体外未定义。
                        return y""",
         "Modification Suggestion": "提前定义y。",
         "Fixed Case": """
                   @jit
                    def func(x):
                        y = 1
                        ^~~~~~~~~~~~~~提前定义y。
                        while x > 0:
                            x -= 1
                            y = 0
                        return y""",
         "Fault Case": """1.静态图语法： 
                       https://www.mindspore.cn/tutorials/experts/zh-CN/r2.0/network/control_flow.html#shapejoin%E8%A7%84%E5%88%99"""}
]

compiler_general_experience_list_cn = [
    {
     "ID": "compiler_g_id_1",
     "Fault Name": "静态图类型推导问题",
     "Key Log Information": "Get instructions about .analyze_fail.dat.",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Code Path":"",
     "Fault Cause": "可能为静态图的语法编译错误，包括数据类型、原型操作、运算符、复合语句、函数等使用的问题。",
     "Modification Suggestion": "1.根据报错信息，分析编译报错的类型； "
                                "2.针对报错的类型，查看官网静态图语法支持介绍，分析报错的原因; "
                                "3.使用analyze_fail.dat文件分析静态图编译结果，定位报错的代码行，分析报错原因。 ",
     "Fault Case": "1.图编译语法支持： "
                   " https://www.mindspore.cn/docs/note/zh-CN/r1.6/static_graph_syntax_support.html "
                   "2.错误分析方法： "
                   "https://www.mindspore.cn/tutorials/experts/zh-CN/r1.8/debug/error_analyze.html "
                   "3.analyze_fail.dat文件分析网络编译失败原因： "
                   "https://www.mindspore.cn/tutorials/experts/zh-CN/r1.8/debug/mindir.html"
                   "4.中间表达MindIR "
                   "https://www.mindspore.cn/docs/zh-CN/r1.8/design/mindir.html"
                   "5.社区论坛-功能调试案例集： "
                   "https://bbs.huaweicloud.com/forum/forum-1076-2704-1.html "},
    {
     "ID": "compiler_g_id_2",
     "Fault Name": "静态图编译问题",
     "Key Log Information": "",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "mindspore.*\.cc",
     "Code Path": "mindspore.core.abstract;mindspore.ccsrc.pipeline.jit",
     "Fault Cause": "可能为静态图的语法编译错误，包括数据类型、原型操作、运算符、复合语句、函数等使用的问题。",
     "Modification Suggestion": "1.根据报错信息，分析编译报错的类型； "
                                "2.针对报错的类型，查看官网静态图语法支持介绍，分析报错的原因; ",
     "Fault Case": "1.图编译语法支持： "
                   " https://www.mindspore.cn/docs/note/zh-CN/r1.6/static_graph_syntax_support.html "}
]
