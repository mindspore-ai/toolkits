## 🔧️一个Pytorch与mindspore的ops匹配工具

### dump

详见：

ms：

https://gitee.com/mindspore/toolkits/tree/master/troubleshooter/troubleshooter/toolbox/dump

pt:

https://gitee.com/lv-kaimeng/tools/tree/ms-cmp/ptdbg_ascend

### 载入pkl文件

```python
# 载入pt网络流
pt_list = OPSList("pytorch")
pt_list.Construct("net_pt.pkl")
# 算子输入输出个数与顺序映射
GetUniIO()(pt_list.ops_list)

# 载入ms网络流
ms_list = OPSList("mindspore")
ms_list.Construct("dnet_ms.pkl")
```

### 匹配

```python
# 第三个参数是对输入输出数值差异的比较（0-1，1为不考虑数值差异）
apis_map = FlowMatch()(pt_list, ms_list, 1)
print_apis_map(apis_map)
```

### demo

```python
from ops_match import *

pt_list = OPSList("pytorch")
ms_list = OPSList("mindspore")

model = 'mobilenet'
pt_list.Construct(f"demo_net/{model}_pt.pkl")
GetUniIO()(pt_list.ops_list)
ms_list.Construct(f"demo_net/{model}_ms.pkl")

print('--------------------------')
apis_map = FlowMatch()(pt_list, ms_list, 1)
print_apis_map(apis_map)
```

