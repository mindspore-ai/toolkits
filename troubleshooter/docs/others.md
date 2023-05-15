# 其他调试功能：

## 应用场景1：tracking在指定epoch/step停止跟踪（使用model.train训练）
### 跟踪结果展示：
跟踪第一个epoch的第一个step后退出
![avatar](images/046102819c1c65f571b06a7412f5efbf.png)

### 如何使用：
        # 引入工具的StopMonitor callback
        from troubleshooter.common.ms_utils import StopMonitor
        ......
        # 设置在第一个epoch ，第一个step停止
        stop_monitor = StopMonitor(stop_epoch=1, stop_step=1)
        ......
        # 将callback配置到model.train接口
        model.train(config.epoch_size, ds_train, callbacks=[time_cb, ckpoint_cb, LossMonitor(), stop_monitor])