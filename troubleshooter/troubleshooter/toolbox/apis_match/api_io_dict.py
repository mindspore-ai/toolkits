## 这是一个算子输入输出的映射字典，用于对齐其他框架与mindspore算子的输入输入顺序以及个数
## 字典的key表示为`(算子类型,算子名称)`，value表示为`([输入顺序列表],[输出顺序列表])`
## 值得注意的是， 输入输出列表指的是原框架（如pytorch）的输出输出映射列表，输入和输出顺序列表的个数不必和实际输入输出个数相同。

# pytorch
pt_io_dict = {
    ('Functional', 'batch_norm'): ([0], [0]),
}
