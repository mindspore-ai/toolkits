TroubleShooter

A troubleshooting toolbox for MindSpore

# 说明
TroubleShooter 是MindSpore 网络开发调试工具包，用于提供便捷、易用的调试能力。

# 安装说明

## pip安装
```bash
pip install troubleshooter -i https://pypi.org/simple
```

## 源码安装

```bash
git clone https://gitee.com/mindspore/toolkits.git
cd toolkits/troubleshooter
bash package.sh
pip install output/troubleshooter-*-py3-none-any.whl
```

## 版本更新

推荐使用最新版本，当前最新版本为1.0.18，更新说明可查看[RELEASE NOTE](RELEASE.md)。

获取最新版本可使用pip在线安装或前往PyPI下载后离线安装，PyPI网站可查看[PyPI](https://pypi.org/project/troubleshooter/#files)。

# API列表

## [API汇总](docs/api_summary.md)

# 应用场景

## [网络迁移&调试](docs/migrator.md)
* [应用场景1：pth到ckpt权重转换](docs/migrator.md#应用场景1pth到ckpt权重自动转换)
* [应用场景2：比对MindSpore与PyTorch的ckpt/pth](docs/migrator.md#应用场景2比对mindspore与pytorch的ckptpth)
* [应用场景3：保存Tensor](docs/migrator.md#应用场景3保存Tensor)
* [应用场景4：比较两组Tensor值(npy文件)是否相等](docs/migrator.md#应用场景4比较两组Tensor值npy文件是否相等)
* [应用场景5：API级别数据保存与对比](docs/api_save_and_compare.md)

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

## [网络调试小工具](docs/widget.md)
* [应用场景1：根据传入的pb文件，识别算子升/降精度标识，输出csv文件](docs/widget.md#应用场景1：提供precision_tracker接口，根据传入的pb文件，识别图中算子执行后精度变化)
