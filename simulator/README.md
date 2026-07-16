# 仿真灌数脚本

## 前置条件

1. 先在后台绑定设备，拿到 `device_id` 和 `secret`（通过 App 或 admin 添加设备）。
2. 确保后端 WebSocket 服务器在 `ws://localhost:8000` 运行。

## 使用方法

```bash
pip install websockets
python feed.py <device_id> <secret>
```

可选参数：第三个参数指定 WebSocket URL（默认 `ws://localhost:8000/ws/ingest`）。

```bash
python feed.py <device_id> <secret> ws://192.168.1.100:8000/ws/ingest
```

脚本会依次发送一组预设事件（presence → posture → vital → posture），间隔 1 秒，用于配合 App 观察实时数据刷新。
