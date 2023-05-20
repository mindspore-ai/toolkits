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

vm_experience_list_cn = [
    {
     "ID": "vm_id_1",
     "Fault Name": "device设备OOM",  # 设备侧内存的定义， Ascend 内存，GPU 内存
     "Key Log Information": "Allocate continuous memory failed",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "内存占用过多导致分配内存失败，可能原因有：batch_size的值设置过大；"
                    "引入了异常大的Parameter，例如：例如单个数据shape为[640,1024,80,81]，"
                    "数据类型为float32，单个数据大小超过15G，两个如此大小的数据相加时，"
                    "占用内存超过3*15G，容易造成Out of Memory",
     "Modification Suggestion": "1.检查batch_size的值，尝试将batch_size的值设置减小. "
                                "2.检查参数的shape，尝试减少shape. "
                                "3.如果以上操作还是未能解决，可以上官方论坛发帖提出问题. ",
     "Fault Case": ": https://bbs.huaweicloud.com/forum/thread-169771-1-1.html"
                   ": https://www.mindspore.cn/docs/faq/zh-CN/master/implement_problem.html?highlight=out%20memory"},
    {
     "ID": "vm_id_2",
     "Fault Name": "device设备占用报错",
     "Key Log Information": "Malloc device memory failed",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "执行网络脚本的设备可能被其他任务占用 ",
     "Modification Suggestion": "1.Ascend设备使用npu-smi info查询设备的使用情况 "
                                "2.检查是否已经有进程占用了卡,是否在同一卡上重复启动了进程,是否有进程没有正常退出 "
                                "3.查询哪些程序占用设备: ps -ef | grep python, 对占用进程进行处理 ",
     "Fault Case": "https://bbs.huaweicloud.com/forum/thread-183730-1-1.html"},
    {
     "ID": "vm_id_3",
     "Fault Name": "getnext算子超时",
     "Key Log Information": "E39999:",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "下沉模式下,设备运行的Getnext算子无法获取数据，达到超时时间报错 ",
     "Modification Suggestion": "1.检查ERROR和WARNING日志，判断是否打印了GetNext相关信息 "
                                "2.如果打印了GetNext信息，则说明是因为GetNext算子执行超时引发的错误，解决建议如下： "
                                "a.可设置dataset_sink_mode=False关闭数据下沉模式（参考：Model接口） "
                                "b.减少数据处理的batch_size,通过调整数据处理量来达成降低数据处理的耗时,避免超时发生 "
                                "c.调试数据处理性能，可参考官网提供的《数据处理性能优化》指导 "
                                "3.如果无法找到GetNext算子信息，则可能其他算子报错，可到社区提单解决 ",
     "Fault Case": "数据处理性能优化: "
                   "https://www.mindspore.cn/tutorials/experts/zh-CN/master/dataset/optimize.html"},
    {
     "ID": "vm_id_4",
     "Fault Name": "ranktable配置文件错误",
     "Key Log Information": "EI0004:",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "使用rank table方式启动分布式训练，配置文件错误（rank_table_xxx.json）或者配置与环境不一致 ",
     "Modification Suggestion": "检查配置文件（rank_table_xxx.json）内容是否正确,是否与环境匹配 "
                                "1.检查配置中的device_id或rank_id配置是否正确，是否有重复、遗漏、超出范围(超出0-7) "
                                "2.检查配置中设置的device_id在环境中是否存在此设备     "
                                "3.报错中尝试查找'invalid Reason'，如果能找到会给出一些详细的错误原因 ",
     "Fault Case": "Ascend HCCL:EI0004 错误码说明: "
                   "https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/600alpha002/troublemanage/troubleshooting/atlaserrorcode_15_0191.html "
                   "配置分布式环境变量: "
                   "https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/train_ascend.html"},
    {
     "ID": "vm_id_5",
     "Fault Name": "hccl算子notifywait超时",
     "Key Log Information": "EI0002:",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "因为异常情况(环境问题或者版本问题)，导致集合通信算子同步超时(超时默认600s)",
     "Modification Suggestion": "可能的异常原因如下，可通过日志检查集合通信各节点是否有如下情况 "
                                "1.训练执行阶段进程被误杀或者某节点网络发生中断 "
                                "2.某节点进程异常退出导致其他节点超时，可查看首个退出节点日志找到异常退出原因 "
                                "3.某进程执行过慢引起其他进程超时，可通过设置环境变量HCCL_EXEC_TIMEOUT增加时长 "
                                "4.某节点进程执行速度慢，可检查数据处理或者外部操作导致了执行慢，例如：保存checkpoint、callback、save_graphs保存IR图等",
     "Fault Case": "Ascend HCCL:EI0002 错误码说明: "
                   "https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/600alpha002/troublemanage/troubleshooting/atlaserrorcode_15_0189.html "
                   "MindSpore执行状态日志解析与应用(可用于分析MindSpore并行训练进程间的执行快慢，用于判断哪个进程在哪个阶段慢): "
                   "https://bbs.huaweicloud.com/forum/thread-180967-1-1.html "
                   "MindSpore性能调优指南: "
                   "https://mindspore.cn/mindinsight/docs/zh-CN/master/performance_tuning_guide.html"},
    {
     "ID": "vm_id_6",
     "Fault Name": "hccl_Get_Socket_Timeout",
     "Key Log Information": "EI0006:",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "因为异常情况(环境问题或者版本问题)，导致集合通信建链超时(超时默认120s)",
     "Modification Suggestion": "可能的异常原因如下，可通过日志检查集合通信各节点是否有如下情况 "
                                "1.实际执行的卡数与配置文件(rank_table_xxx.json)中配置的不一致 "
                                "2.多卡执行，在建链前一个卡的进程出现异常退出或者挂起 "
                                "3.某进程执行过慢引起其他进程超时，可设置环境变量HCCL_CONNECT_timeout增加时长 ",
     "Fault Case": "Ascend HCCL:EI0006 错误码说明: "
                   "https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/600alpha002/troublemanage/troubleshooting/atlaserrorcode_15_0193.html "
                   "MindSpore执行状态日志解析与应用(可用于分析进程执行异常/执行慢): "
                   "https://bbs.huaweicloud.com/forum/thread-180967-1-1.html "
                   "性能调优指南: "
                   "https://mindspore.cn/mindinsight/docs/zh-CN/master/performance_tuning_guide.html"},
    {
     "ID": "vm_id_7",
     "Fault Name": "hccl_p2p_Timeout",
     "Key Log Information": "EI9999:",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "因为异常情况(环境问题或者版本问题)，导致集合通信建链超时(超时默认600s) ",
     "Modification Suggestion": "可能的异常原因如下，可通过日志检查集合通信各节点是否有如下情况 "
                                "1.实际执行的卡数与配置文件(rank_table_xxx.json)中配置的不一致 "
                                "2.多卡执行,在建链前一个卡的进程出现异常退出或者挂起,导致其他节点建链超时 "
                                "3.某设备进程执行过慢引起其他进程超时 ",
     "Fault Case": "MindSpore执行状态日志解析与应用(可用于分析进程执行异常/执行慢): "
                   "https://bbs.huaweicloud.com/forum/thread-180967-1-1.html "
                   "性能调优指南: "
                   "https://mindspore.cn/mindinsight/docs/zh-CN/master/performance_tuning_guide.html"},
    {
     "ID": "vm_id_8",
     "Fault Name": "分布式训练相同训练进程连续多次启动",
     "Key Log Information": "EJ0001:",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Fault Cause": "可能为分布式训练，相同进程同时段内重复启动造成冲突 ",
     "Modification Suggestion": "1.检查是否进程出现了同时段内重复启动的情况 "
                                "2.可能存在进程还未完全退出时启动，导致冲突，等待一段时间后重试 ",
     "Fault Case": "Ascend HCCL:EJ0001 错误码说明: "
                   "https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/600alpha002/troublemanage/troubleshooting/atlaserrorcode_15_0197.html "
                   "MindSpore执行状态日志解析与应用(可用于分析进程执行异常/执行慢): "
                   "https://bbs.huaweicloud.com/forum/thread-180967-1-1.html "
                   "性能调优指南: "
                   "https://mindspore.cn/mindinsight/docs/zh-CN/master/performance_tuning_guide.html"},
    {
     "ID": "vm_id_9",
     "Fault Name": "算子编译失败",
     "Key Log Information": "Single op compile failed",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Code Path": "ascend_session.cc",
     "Fault Cause": "算子编译失败",
     "Modification Suggestion": "1.尝试查找'The function call stack:'，可找到编译失败算子关联的代码行 "
                                "2.查找errCode，相关描述会给出算子编译失败的一些原因，可基于报错原因尝试解决编译问题 ",
     },
    {
     "ID": "vm_id_10",
     "Fault Name": "device_id设置超出机器限制",
     "Key Log Information": "EE8888",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Code Path": "ascend_session.cc",
     "Fault Cause": "device_id设置错误，超出机器支持的device_id范围",
     "Modification Suggestion": "请检查环境或者配置中设置的device_id是否在范围内，通常单机8卡情况，device_id的范围是0-7 ",
     },
    {
     "ID": "vm_id_11",
     "Fault Name": "算子选择失败",
     "Key Log Information": "Can not find any available kernel info for",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "HandleKernelSelectFailure",
     "Code Path": "mindspore/ccsrc/plugin/device/ascend/hal/device/kernel_select_ascend.cc",
     "Fault Cause": "网络编译报错，某算子选择失败。",
     "Modification Suggestion": "1. 检查该硬件平台是否支持该算子，可以在算子库API中查看关于该算子的文档："
                                "2. 如果AI CORE候选算子信息为空，则可能是在算子check support阶段，所有的算子信息均校验未通过。"
                                "可以在日志中搜索关键字CheckSupport找到未通过的原因，根据具体信息修改shape或data type，或者找"
                                "开发人员进一步定位。"
                                "3. 如果AI CPU候选算子信息不为空，或者AI CORE和AI CPU候选算子信息都不为空，则可能是用户给到该"
                                "算子的输入数据类型不在候选列表中，在选择阶段被过滤导致，可以根据候选列表尝试修改该算子的输入data type。",
     "Fault Case": "1. Mindspore.ops算子库API: "
                   "https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.ops.html"
                   "2. MindSpore在Ascend后端报错算子不支持: "
                   "https://bbs.huaweicloud.com/forum/thread-183313-1-1.html"
    },
    {
     "ID": "vm_id_12",
     "Fault Name": "mindspore版本和run包不匹配",
     "Key Log Information": "errNo\[0x0000000005010001\]",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Code Path": "hcom_ops_kernel_info_store.cc",
     "Fault Cause": "mindspore版本和run包不匹配。",
     "Modification Suggestion": "使用匹配的mindspore版本和run包，mindspore版本和run包匹配关系可在【Mindspore安装指南】中查看。",
     "Fault Case": "Mindspore安装指南: "
                   "https://www.mindspore.cn/install"
    },
    {
     "ID": "vm_id_13",
     "Fault Name": "通信算子下发失败",
     "Key Log Information": "load task fail, return ret:",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "Distribute",
     "Code Path": "mindspore\ccsrc\plugin\device\ascend\hal\device\ge_runtime\task\hccl_task.cc",
     "Fault Cause": "通信算子下发失败。",
     "Modification Suggestion": "1. 检查是否是其他节点提前退出，可查看首个退出节点日志找到异常退出原因。"
                                "2. 可能是不支持的建链方式导致，检查通信组网配置。",
     "Fault Case": "多卡训练报错davinci_model : load task fail, return ret xxx: "
                   "https://bbs.huaweicloud.com/forum/thread-182395-1-1.html"
    },
    {
     "ID": "vm_id_14",
     "Fault Name": "流超限",
     "Key Log Information": "Total stream number.*?exceeds the limit of",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Code Path": "mindspore\ccsrc\plugin\device\ascend\hal\device\ascend_stream_assign.cc",
     "Fault Cause": "一般是通信算子多导致流超过硬件限制。",
     "Modification Suggestion": "1. 如果使用pipeline并行，可以尝试减少micro batch数。"
                                "2. 尝试设置通信算子融合。"
                                "3. 尝试设置通信算子图提取与复用",
     "Fault Case": "1. 报错Total stream number xxx exceeds the limit of："
                   "https://bbs.huaweicloud.com/forum/thread-181720-1-1.html"
                   "2. 通信融合特性文档："
                   "https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/comm_fusion.html"
                   "3. 通信子图提取与复用特性文档："
                   "https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/comm_subgraph.html"
    },
    {
     "ID": "vm_id_15",
     "Fault Name": "算子编译中response is empty",
     "Key Log Information": "Response is empty",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "mindspore/ccsrc/backend/common/session/kernel_build_client.h:.*?Response",
     "Code Path": "mindspore/ccsrc/backend/common/session/kernel_build_client.h",
     "Fault Cause": "一般是算子编译的子进程挂了或者调用阻塞卡住导致的超时",
     "Modification Suggestion": "1. 检查日志，在这个错误前是否有其他错误日志，如果有请先解决前面的错误，一些算子相关的问题"
                                "（比如昇腾上TBE包没装好，GPU上没有nvcc）会导致后续的Response is empty报错。"
                                "2. 如果有使用图算融合特性，有可能是图算的AKG算子编译卡死超时导致，可以尝试关闭图算特性。"
                                "3. 在昇腾上可以尝试减少算子并行编译的进程数，可以通过环境变量MS_BUILD_PROCESS_NUM设置，"
                                "取值范围为1~24。"
                                "4. 如果在云上训练环境遇到该问题，可以尝试重启内核。",
     "Fault Case": "训练报Response Is Empty："
                   "https://bbs.huaweicloud.com/forum/thread-115541-1-1.html"
    },
    {
     "ID": "vm_id_16",
     "Fault Name": "EI9999错误码：异步拷贝失败",
     "Key Log Information": "Memory async copy failed",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Code Path": "",
     "Fault Cause": "可能是算子溢出导致",
     "Modification Suggestion": "检查日志，如果算子发生溢出会有overflow报错信息，找到发生溢出的算子再进一步分析。",
     "Fault Case": "报错rtStreamSynchronize failed, ret: 507011："
                   "https://bbs.huaweicloud.com/forum/thread-0229101440416021085-1-1.html"
    },
    {
     "ID": "vm_id_17",
     "Fault Name": "数据格式不匹配",
     "Key Log Information": "Found inconsistent format! input format",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Code Path": "mindspore\ccsrc\plugin\device\ascend\optimizer\format_type\check_consistency.cc",
     "Fault Cause": "输入数据的格式与算子选择得到的数据格式不一致",
     "Modification Suggestion": "检查输入的数据格式与算子选择过程，分析导致数据格式不一致的原因。",
     "Fault Case": "输入数据要要求的数据格式不匹配："
                   "https://bbs.huaweicloud.com/forum/thread-187574-1-1.html"
    },
]

vm_general_experience_list_cn = [
    {
     "ID": "vm_g_id_1",
     "Fault Name": "网络执行报错(AICORE算子)",
     "Key Log Information": "run task error",
     "Key Python Stack Information": "",
     "Key C++ Stack Information": "",
     "Code Path": "ascend_session.cc",
     "Fault Cause": "网络执行报错，某个算子执行失败。",
     "Modification Suggestion": "此类问题成因较为复杂，常见的为用户网络中存在异常值(inf/nan)导致设备上算子执行异常，可以尝试分段屏蔽代码来排查问题： "
                                "1.查找ERROR日志中关键字'Dump node'，正常情况会打印出错的算子名称和关联的代码 "
                                "2.查找错误码EZ9999，错误码关联的信息会提供更为详细的算子执行错误信息 "
                                "3.根据算子名称，结合计算图IR信息，分析可能对应的报错代码行，区分是用户代码报错还是框架代码报错 "
                                "4.使用Dump功能，保存算子的输入和输出数据，按官网说明进行代码调试，分析算子报错的引入位置 "
                                "5.如没有解决思路，则在社区提单求助 ",
     "Fault Case": "1.Dump功能说明： "
                   "https://www.mindspore.cn/tutorials/experts/zh-CN/r1.8/debug/dump.html "
                   "2.自定义调试信息: "
                   "https://www.mindspore.cn/tutorials/experts/zh-CN/r1.8/debug/custom_debug.html"},

]
# Launch graph failed
