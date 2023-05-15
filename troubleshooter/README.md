TroubleShooter

A troubleshooting toolbox for MindSpore

# 说明
TroubleShooter 是MindSpore 网络开发调试工具包，用于提供便捷、易用的调试能力。

# 安装说明

```bash
pip install troubleshooter
```

# 应用场景

## [网络迁移&调试](docs/migrator.md)
* [应用场景1：pth到ckpt权重转换](docs/migrator.md#应用场景1pth到ckpt权重自动转换)
* [应用场景2：将转换后的ckpt与MindSpore网络生成的ckpt进行对比](docs/migrator.md#应用场景2将转换后的ckpt与mindspore网络生成的ckpt进行对比)
* [应用场景3：保存tensor](docs/migrator.md#应用场景3保存tensor)
* [应用场景4：比较两组tensor值(npy文件)是否相等](docs/migrator.md#应用场景4比较两组tensor值npy文件是否相等)
* [应用场景5：比较pytorch和mindspore的网络输出是否相等](docs/migrator.md#应用场景5比较mindspore和pytorch网络输出是否一致)

## [网络错误调试](docs/proposer.md)
* [应用场景1：网络抛出异常后自动生成报错处理建议（在线分析）](docs/proposer.md#应用场景1自动生成报错处理建议在线分析)
* [应用场景2：已产生的报错生成处理建议（离线分析）](docs/proposer.md#应用场景2已生成的报错自动分析离线分析)
* [应用场景3：显示简洁异常调用栈(屏蔽部分框架调用栈信息)](docs/proposer.md#应用场景3显示简洁异常调用栈删除部分框架栈信息)

## [网络执行跟踪](docs/tracker.md)
* [应用场景1：打印运行时的网络信息(结构、shape)](docs/tracker.md#应用场景1打印运行时的网络信息结构shape)
* [应用场景2：获取INF/NAN值抛出点](docs/tracker.md#应用场景2获取infnan值抛出点)
* [应用场景3：按照路径黑/白名单过滤跟踪信息](docs/tracker.md#应用场景3按照路径黑白名单过滤跟踪信息)
* [应用场景4：跟踪API报错的详细执行过程](docs/tracker.md#应用场景4跟踪api报错的详细执行过程)

## [其他调试功能](docs/others.md)
* [应用场景1：tracking在指定epoch/step停止跟踪（使用model.train训练）](docs/others.md#应用场景1tracking在指定epochstep停止跟踪使用modeltrain训练)
