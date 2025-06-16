# 执行序比对用例-jit加速场景执行序比对

jit函数可将Python函数编译为一张可调用的MindSpore图。MindSpore可以在运行时对图进行优化。
## 用例一：rope函数jit加速
### 步骤一：执行用户代码生成执行需文件
1. 设置环境变量export MS_ALLOC_CONF=memory_tracker:True，执行MDeepSeekV3脚本，获取执行序。执行序文件存于rank_0/tracker_graph.ir
2. 在DeepSeekv3中，对相关代码进行jit加速，再次执行脚本，获取执行序。示例代码如下：

```
from mindspore import jit
@jit(capture_mode='ast')
def apply_rotary_pos_emb_bshd(t: Tensor, freqs: Tensor, rotary_interleaved: bool = False) -> Tensor:
    args = get_args()
    if args.use_glm_rope:
        return _process_partial_rope(freqs, t)

    _mscale = 1
    if args.rope_scaling_type == "yarn":
        _mscale = float(
            yarn_get_mscale(args.rope_scaling_factor, args.rope_scaling_mscale)
            / yarn_get_mscale(args.rope_scaling_factor, args.rope_scaling_mscale_all_dim)
        )
    elif args.rope_scaling_type == "longrope":
        if args.long_mscale and args.short_mscale:
            scale = args.seq_length / args.rope_scaling_original_max_position_embeddings
            _mscale = args.long_mscale if scale > 1 else args.short_mscale
        else:
            scale = args.max_position_embeddings / args.rope_scaling_original_max_position_embeddings
            _mscale = 1.0 if scale <= 1.0 else math.sqrt(
                1 + math.log(scale) / math.log(args.rope_scaling_original_max_position_embeddings))

        rot_dim = freqs.shape[-1]
        t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
        cos = (torch.cos(freqs) * _mscale).to(t.dtype)
        sin = (torch.sin(freqs) * _mscale).to(t.dtype)
    
    if args.use_fused_rotary_pos_emb:
        # t = torch_npu.npu_rotary_mul(t, cos_, sin_).to(t.dtype)
        mode = 1 if rotary_interleaved else 0
        t = torch_npu.npu_rotary_position_embedding(t.contiguous(), cos_, sin_, mode=mode).to(t.dtype)
    elif args.use_fused_rotary_pos_emb_new:
        mode = 1 if rotary_interleaved else 0
        t = torch_npu.npu_rotary_position_embedding(t.contiguous(), cos_, sin_, mode=mode).to(t.dtype)
    else:
        t = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)
    
    return torch.cat((t, t_pass), dim=-1)
```

### 步骤二：将生成的执行序进行比较
3. 打开执行序比对程序。
4. 选择并设置左右图对应的执行序文件。
5. 在执行序文件中找到jit装饰函数的最后一个算子，ConCat算子第一次出现的位置。可参考的未加jit时的调用栈确定行号。使用行号设置左右侧锚点。
6. 再点击 向上比较 按钮，即可找到两执行序的差异点。如图所示，差异点有以下几种类型：
a. 动静态图三算子类型不一致：动态图使用Slice算子；静态图使用StridedSlice算子；
b. 动态图与静态图连接处插入Contiguous连续算子；
c. 静态图没有strides和offset属性；静态图部分算子有这个属性；

## 用例二：vocabpositionembedding jit场景加速
### 步骤一：执行用户代码生成执行需文件
1. 设置环境变量export MS_ALLOC_CONF=memory_tracker:True，执行MDeepSeekV3脚本，获取执行序。执行序文件存于rank_0/tracker_graph.ir
2. 在DeepSeekv3中，对相关代码进行jit加速，再次执行脚本，获取执行序。示例代码如下：

```
from mindspore import jit
# @jit(capture_mode='ast') #报错 NameError: The name 'input_mask' is not defined, or not supported in graph mode.
def vocab_parallel_embedding_forward(self, input_, weight=None):
    @pi_jit_with_config(jit_config=jit_config)
    def get_input_mask(self, input):
        if self.tensor_model_parallel_size > 1:
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | \
                        (input_ >= self.vocab_end_index)

            # Mask the input.
            # masked_input = input_.clone() - self.vocab_start_index
            masked_input = input_ - self.vocab_start_index
            masked_input *= ~input_mask
        else:
            masked_input = input_
            # Get the embeddings.
        return masked_input, input_mask
    if weight is None:
        if self.weight is None:
            raise RuntimeError(
                "weight was not supplied to VocabParallelEmbedding forward pass "
                "and skip_weight_param_allocation is True."
            )
        weight = self.weight

    masked_input, input_mask = get_input_mask(self, input=input_)

    # For higher accumulation accuracy for bf16 on NPU.
    output_parallel = F.embedding(masked_input, weight)

    # Mask the output embedding.
    if self.tensor_model_parallel_size > 1:
        output_parallel *= ~input_mask[..., None]
    if self.reduce_scatter_embeddings:
        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        output_parallel = output_parallel.transpose(0, 1).contiguous()
        output = reduce_scatter_to_sequence_parallel_region(output_parallel)
    else:
        # Reduce across all the model parallel GPUs.
        output = reduce_from_tensor_model_parallel_region(output_parallel)
    args_ = get_args()
    if hasattr(self, 'norm'):
        output = self.norm(output)
    return output * args_.embedding_multiplier_scale if args_.embedding_multiplier_scale else output
```

### 步骤二：将生成的执行序进行比较
3. 打开执行序比对程序。
4. 选择并设置左右图对应的执行序文件。
5. 在执行序文件中找到jit装饰函数的最后一个算子，ConCat算子第一次出现的位置。可参考的未加jit时的调用栈确定行号。使用行号设置左右侧锚点。
6. 再点击 向上比较 按钮，即可找到两执行序的差异点。如图所示，差异点有以下几种类型：
a. 动静态图算子不一致，导致退避到CPU执行。执行序连线丢失；
b. 动态图与静态图连接处插入Contiguous连续算子；
c. 静态图没有strides和offset属性；静态图部分算子有这个属性；
