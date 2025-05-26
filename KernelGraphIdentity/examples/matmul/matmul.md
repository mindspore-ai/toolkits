# 执行序比对用例1-同框架不同版本比对

## 步骤一：执行用户代码生成执行需文件
1. 设置环境变量export MS_ALLOC_CONF=memory_tracker:True，执行MindFormers DeepSeekV3脚本，获取执行序。执行序文件存于rank_0/tracker_graph.ir
2. 在DeepSeekv3中，替换P.Matmul()算子，改为mint.matmul()，再次执行脚本，获取执行序。
```
class DataParallelLinear(nn.Cell):
    def __init__(self,
                input_size: int,
                output_size: int,
                config: TransformerConfig,
                init_method: Callable = None,
                bias: bool = True,
                skip_bias_add: bool = False,
                skip_weight_param_allocation: bool = False,
                transpose_b: bool = True,
                compute_dtype: dtype = dtype.float16,
                bias_init: Callable = None
                ):
        self.matmul = MatMulExt()

    def construct(self, input_: Tensor, weight: Tensor = None) -> tuple[Tensor, Tensor]:
        output_shape = input_.shape[:-1] + (self.output_size,)
        input_ = self.reshape(input_, (-1, self.input_size))
        ori_dtype = input_.dtype
        weight = self.cast(weight, self.compute_dtype)
        input_ = self.cast(input_, self.compute_dtype)

        if self.transpose_b:
            weight = self.transpose(weight, (1, 0))
        input_ = self.matmul(input_, weight)
        #  input_ = mint.matmul(input_, weight)
```
## 步骤二：将生成的执行序进行比较
3. 打开执行序比对程序。
4. 选择并设置左右图对应的执行序文件。
![设置同框架不同版本执行序](example/matmul/pictures/设置同框架不同版本执行序.png)
5. 点击 切换至整图 按钮。
![切换至整图](example/matmul/pictures/切换至整图.png)
6. 再点击 向上比较 按钮，即可找到两执行序的差异点。如图所示，绿色为无差异的点，蓝色为未进行匹配的点，绿色与蓝色分界处为差异层。
![比较同框架不同版本差异](example/matmul/pictures/比较同框架不同版本差异.png)
7. 滚动鼠标滚轮可放大缩小图，放大图找到差异点。
![缩放图找到同框架不同版本差异点](example/matmul/pictures/缩放图找到同框架不同版本差异点.png)
8. 可以根据节点信息展示区输出得知差异原因。
可以看到差异原因是差异算子Transpose的前驱算子MatMulExt的输入地址shape存在差异。
![同框架不同版本差异](example/matmul/pictures/同框架不同版本差异.png)
