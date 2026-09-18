# 家安 housafe 全链路测试指南

> 不只是代码——从雷达点云到 App 页面，每一层怎么测、测什么、什么时候测。

---

## 测试金字塔（按层级）

```
        ┌──────────┐
        │  App 端   │  ← 手动/截图对比/真机验证
       ┌┴──────────┴┐
       │  实时推送    │  ← 集成测试 + wscat 手动
      ┌┴────────────┴┐
      │  API / 落库   │  ← 自动化测试为主 + curl 手动
     ┌┴──────────────┴┐
     │  AI 推理/回调   │  ← 规则验证 + 边界用例
    ┌┴────────────────┴┐
    │  Redis Stream 缓冲 │  ← 集成测试 + redis-cli
   ┌┴──────────────────┴┐
   │  雷达设备 / Ingest   │  ← 单元测试 + 模拟器手动
  └─────────────────────┘
```

---

## 第 1 层：雷达设备 & 数据接入（Ingest WebSocket）

### 1.1 模拟器数据质量验证

**测什么：** 模拟器发出的点云帧和体征帧是否符合真实设备行为。

```bash
# 1. 发送 walk 模式，观察点云 z 轴分布是否合理
python simulator/feed.py rad_test s1 --mode walk

# 2. 发送 fall 模式，确认 z 轴被限制在 0~0.2m（跌倒特征）
python simulator/feed.py rad_test s1 --mode fall

# 3. 逐模式验证：walk → sit → lie → fall → stand
for mode in walk sit lie fall stand; do
  echo "=== $mode ==="
  python simulator/feed.py rad_test s1 --mode $mode
done
```

**检查点：**

| 项目 | 预期 | 怎么查 |
|------|------|--------|
| 点云帧频率 | 每 2s 一帧 | 看终端输出时间间隔 |
| 体征帧频率 | 每 5 帧（~10s）一次 | 数 `[vital]` 出现次数 |
| 心跳频率 | 每 10 帧（~20s）一次 | 数 `[heartbeat]` 出现次数 |
| 点云点数 | walk: ~25, fall: ~8 | 看终端 `(N points)` |
| z 轴范围 | walk: 0~1.8m, lie: 0~0.3m | 需改代码打 log |

### 1.2 WebSocket 鉴权

```bash
# 错误 secret → 应该关闭连接（code 4401）
python -c "
import asyncio, json, websockets
async def t():
    async with websockets.connect('ws://localhost:8000/ws/ingest') as ws:
        await ws.send(json.dumps({'device_id':'rad_e2e','secret':'WRONG'}))
        print(await ws.recv())
asyncio.run(t())
"
# 预期：连接被关闭，或收到 error

# 正确 secret → auth ack
python -c "
import asyncio, json, websockets
async def t():
    async with websockets.connect('ws://localhost:8000/ws/ingest') as ws:
        await ws.send(json.dumps({'device_id':'rad_e2e','secret':'e2e_secret'}))
        print('auth:', await ws.recv())
asyncio.run(t())
"
# 预期：{"ack": "auth"}
```

### 1.3 帧格式校验

```bash
# 空 points 数组 → 应该返回 error（Pydantic min_length=1）
python -c "
import asyncio, json, websockets
async def t():
    async with websockets.connect('ws://localhost:8000/ws/ingest') as ws:
        await ws.send(json.dumps({'device_id':'rad_e2e','secret':'e2e_secret'}))
        print(await ws.recv())
        await ws.send(json.dumps({'kind':'point_cloud','payload':{'ts':1,'radar_id':'x','room':'r','frame_id':'f','points':[]}}))
        print(await ws.recv())
asyncio.run(t())
"
# 预期：{"error": "invalid"}

# 未知 kind → error
# 预期：{"error": "invalid"}

# 缺少必填字段（如 vital 无 quality）→ error
# 预期：{"error": "invalid"}
```

### 1.4 已有自动化测试覆盖

```
backend/ingest/tests/test_ingest.py
  ├── test_bad_secret_closes       ✅ 错误 secret 断连
  ├── test_point_cloud_xadd        ✅ 点云帧 → Redis XADD
  ├── test_vital_xadd              ✅ 体征帧 → Redis XADD
  ├── test_heartbeat_updates_device ✅ 心跳更新设备状态
  ├── test_invalid_frame_returns_error ✅ 非法帧返回 error
  └── test_unknown_kind_returns_error  ✅ 未知 kind 返回 error
```

**缺失的测试：**
- 高频发送（背压测试）：每秒发 10+ 帧，观察 Consumer 是否崩溃
- 断线重连：设备断线后重连，Stream 数据是否连续
- 并发多设备：10 台设备同时连入，鉴权和帧处理是否隔离

---

## 第 2 层：Redis Stream 缓冲

### 2.1 Stream 数据完整性

```bash
# 发 5 帧点云，然后检查 Stream
python simulator/feed.py rad_e2e e2e_secret --mode walk
# Ctrl+C 停掉后立即查：
redis-cli XLEN housafe:pointcloud:ingest
# 预期：>= 5

# 查看最新一帧内容
redis-cli XREVRANGE housafe:pointcloud:ingest + - COUNT 1
```

### 2.2 背压 & MAXLEN

```bash
# 开两个终端
# T1：持续大量灌数（修改 feed.py 去掉 sleep）
# T2：监控 Stream 长度
watch -n 1 'redis-cli XLEN housafe:pointcloud:ingest'
# 预期：长度不超过 1000（MAXLEN ~1000）
```

### 2.3 AI 消费确认

```bash
# 确认 AI placeholder 在消费
redis-cli XINFO GROUPS housafe:pointcloud:ingest
# 如果用 consumer group 的话

# 当前实现是 XREAD 无 group 模式，
# 确认 AI 的 last_id 在前进：
# 看 AI 日志中 msg_id 是否递增
```

### 2.4 手动检查命令

```bash
# 所有 housafe Stream 概览
python -c "
import redis
r = redis.Redis.from_url('redis://localhost:6379/0')
for s in r.keys('housafe:*'):
    info = r.xinfo_stream(s)
    print(f'{s.decode()}: {info[\"length\"]} msgs, last={info[\"last-generated-id\"].decode()}')
"

# 清空 Stream（测试前重置）
redis-cli DEL housafe:pointcloud:ingest housafe:vital:ingest
```

---

## 第 3 层：AI 推理 & 回调

### 3.1 姿态推断正确性

**核心逻辑：** 根据点云 z 轴包围盒高度 → 映射姿态。

```
高度 < 0.3m  → lie（躺卧）
高度 < 0.8m  → sit（坐着）
高度 < 1.8m  → stand（站立）
高度 ≥ 1.8m → walk（走动）
```

**手动验证：**

```bash
# 用不同 mode 的模拟器各发 3 帧，看 AI 推断结果
# T1: daphne -p 8000 housafe.asgi:application
# T2: python ai/placeholder/main.py
# T3:

# walk mode → 点云高度范围大 → AI 应推断 walk
python simulator/feed.py rad_e2e e2e_secret --mode walk
# 查 RoomState 中的 posture：
python -c "
import os, sys; sys.path.insert(0,'backend'); os.environ['DJANGO_SETTINGS_MODULE']='housafe.settings'
import django; django.setup()
from events.models import RoomState
rs = RoomState.objects.filter(device_id='rad_e2e').first()
print(f'posture={rs.posture} confidence={rs.confidence}')
"

# 分别用 lie / sit / fall 模式验证
```

### 3.2 边界用例

| 场景 | 输入 | 预期输出 |
|------|------|----------|
| 空点云（0 个点） | `points=[]` | posture=stand, confidence=0.5 |
| 单点（无法算高度） | 1 point | height=0.5 → sit |
| 极低高度（跌倒） | z 全在 0~0.2m | posture=lie |
| 大量点（房间多人） | 50+ points | moving=True（n_points > 5）|

### 3.3 AI 回调可靠性

```bash
# 1. AI 未启动时，Stream 堆积 → AI 启动后应消费积压
# 验证：先灌数 → 停 AI → 再启 AI，看是否消费

# 2. Backend 挂了 → AI 应重试 POST（目前会打日志然后丢帧）
# 改进点：应加 retry 或 DLQ

# 3. 错误 token → 返回 401
curl -s -X POST http://localhost:8000/api/internal/decoder-output \
  -H "Content-Type: application/json" \
  -d '{"token":"wrong","outputs":[]}' | python -m json.tool
# 预期：{"error": "unauthorized"}

# 4. 空 outputs → 400
curl -s -X POST http://localhost:8000/api/internal/decoder-output \
  -H "Content-Type: application/json" \
  -d '{"token":"dev-internal-token","outputs":[]}' | python -m json.tool
# 预期：{"error": "empty outputs"}
```

### 3.4 延迟测试

```bash
# AI 占位符模拟 0.3~1.0s 推理延迟
# 灌 10 帧，观察端到端延迟：设备发帧 → RoomState 更新
# 端到端延迟 ≈ 网络 + Redis 消费 + 推理延迟 + HTTP POST
# 预期：< 2s（本地）
```

---

## 第 4 层：数据存储（PostgreSQL / RoomState）

### 4.1 Upsert 正确性

```bash
# 同一 device_id 多次写入 → RoomState 只有一行
python -c "
import os, sys; sys.path.insert(0,'backend'); os.environ['DJANGO_SETTINGS_MODULE']='housafe.settings'
import django; django.setup()
from events.models import RoomState
# 查所有 RoomState，每个 device_id 最多一行
from django.db.models import Count
dupes = RoomState.objects.values('device_id').annotate(n=Count('id')).filter(n__gt=1)
print(f'dup device_ids: {list(dupes)}')
"
```

### 4.2 字段合并

```bash
# posture 写入 → vital 写入 → RoomState 应有 posture + heart_rate + resp_rate
# 验证命令同上，检查：
#   rs.posture != None
#   rs.heart_rate != None
#   rs.resp_rate != None
#   rs.ts == max(posture.ts, vital.ts)
```

### 4.3 Alert 创建

```bash
# AI 检测到 fall → 应创建 Alert
# 用 curl 注入一个 fall alert：
curl -s -X POST http://localhost:8000/api/internal/decoder-output \
  -H "Content-Type: application/json" \
  -d '{
    "token": "dev-internal-token",
    "outputs": [{
      "kind": "alert",
      "family_id": 1,
      "payload": {
        "ts": 1700000000000,
        "device_id": "rad_e2e",
        "room": "bedroom",
        "alert_type": "fall",
        "severity": "critical",
        "payload": {"detail": "test fall detected"}
      }
    }]
  }' | python -m json.tool

# 查 Alert 表：
python -c "
import os, sys; sys.path.insert(0,'backend'); os.environ['DJANGO_SETTINGS_MODULE']='housafe.settings'
import django; django.setup()
from events.models import Alert
for a in Alert.objects.all().order_by('-ts')[:5]:
    print(f'  {a.device_id} | {a.alert_type} | {a.severity} | {a.ts}')
"
```

### 4.4 已有自动化测试覆盖

```
backend/events/tests/test_store.py
  ├── test_posture_upserts_room_state    ✅ 姿态 upsert
  ├── test_vital_upserts_room_state      ✅ 体征 upsert
  ├── test_alert_creates_record          ✅ 告警创建
  ├── test_occupancy_upserts_room_state  ✅ 人数 upsert
  ├── test_anomaly_upserts_room_state    ✅ 异常分 upsert
  └── test_multiple_output_types_merge   ✅ 多类型字段合并
```

---

## 第 5 层：REST API 查询

### 5.1 Today 接口

```bash
# 正常查询
curl -s http://localhost:8000/api/families/1/today \
  -H "Authorization: Bearer <JWT>" | python -m json.tool
# 预期：
# {
#   "rooms": [
#     {
#       "room_name": "bedroom",
#       "device_online": true,
#       "posture": "walk",
#       "posture_ts": 1700000000000,
#       "heart_rate": 72.0,
#       "breath_rate": 16.0
#     }
#   ]
# }

# 不存在的 family → 404
curl -s http://localhost:8000/api/families/999/today \
  -H "Authorization: Bearer <JWT>"
# 预期：{"detail": "Not found."}

# 别人的 family → 404（owner 隔离）
# 预期：Not found（不是 403，防止信息泄露）

# 无设备绑定的 family → rooms 为空
# 预期：{"rooms": []}

# 有设备但无 RoomState（AI 未推理）→ rooms 中 posture/heart_rate 等为 null
# 预期：room 存在但 posture: null, heart_rate: null
```

### 5.2 权限隔离

```bash
# 用户 A 创建 family → 用户 B 尝试访问 → 应 404
# 验证 owner 隔离是否正确
```

### 5.3 已有自动化测试覆盖

```
backend/events/tests/test_query.py
  ├── test_today_returns_latest_per_room ✅ Today 返回最新状态
  ├── test_ai_bridge_unauthorized        ✅ 错误 token 401
  ├── test_ai_bridge_stores_and_pushes   ✅ 正确落库+推送
  └── test_ai_bridge_empty_outputs       ✅ 空 outputs 400

backend/accounts/tests/test_auth.py      ✅ 注册/登录/JWT
backend/families/tests/test_families.py  ✅ 家庭 CRUD + owner 隔离
backend/devices/tests/test_devices.py    ✅ 设备绑定/列表
```

---

## 第 6 层：实时推送（Realtime WebSocket）

### 6.1 JWT 鉴权 & 家庭隔离

```bash
# 无 token → 应拒绝
wscat -c "ws://localhost:8000/ws/app"
# 预期：连接关闭（code 4401）

# 错误 token → 应拒绝
wscat -c "ws://localhost:8000/ws/app?token=invalid&family=1"
# 预期：连接关闭

# 别人的 family → 应拒绝
wscat -c "ws://localhost:8000/ws/app?token=<JWT>&family=999"
# 预期：连接关闭

# 正确 token + 自己的 family → 连接成功
wscat -c "ws://localhost:8000/ws/app?token=<JWT>&family=1"
# 预期：Connected (press CTRL+C to quit)
```

### 6.2 room.state 推送验证

```bash
# T1: daphne 前台
# T2: wscat 连接 App WS
wscat -c "ws://localhost:8000/ws/app?token=<JWT>&family=1"

# T3: 手动注入推理结果（触发 group_send）
curl -s -X POST http://localhost:8000/api/internal/decoder-output \
  -H "Content-Type: application/json" \
  -d '{
    "token": "dev-internal-token",
    "outputs": [{
      "kind": "posture",
      "family_id": 1,
      "payload": {"ts": 1700000000000, "device_id": "rad_e2e", "room": "bedroom", "posture": "fall", "confidence": 0.95}
    }]
  }'

# T2 应立即收到：
# {"type":"room.state","room_name":"bedroom","device_online":true,"posture":"fall","posture_ts":1700000000000,"heart_rate":null,"breath_rate":null,"anomaly_score":0.0}
```

### 6.3 断线重连

```bash
# 1. wscat 连上
# 2. 手动断开（Ctrl+C）
# 3. 重新连接 → 应收到当前最新的 room.state
# （当前实现不发送初始状态，需要 App 先 GET /today 再开 WS）
```

### 6.4 向后兼容 v2.x 协议

```bash
# App Consumer 支持两种推送格式：
# v3.0: {"type": "room.state", "room_name": "...", ...}
# v2.x: {"kind": "posture", "payload": {...}}

# 在代码中验证 event_push handler 正确处理 v2 格式
# 当前 AppConsumer.event_push 直接转发 payload，前端 mergeUpdate 处理
```

### 6.5 已有自动化测试覆盖

```
backend/realtime/tests/test_realtime.py
  ├── test_app_room_state_handler          ✅ room.state 推送
  └── test_ingest_to_ai_bridge_to_app_flow ✅ 全链路推送
```

---

## 第 7 层：App 前端（React Native / Expo）

### 7.1 登录流程

| 场景 | 操作 | 预期 |
|------|------|------|
| 空用户名 | 不输入，点登录 | 显示"请输入用户名和密码" |
| 空密码 | 只输用户名，点登录 | 同上 |
| 错误密码 | 输入正确用户名+错误密码 | 显示"登录失败，请重试" |
| 正确凭据 | 输入正确用户名+密码 | 跳转到 TodayScreen |
| 网络不通 | 断网后点登录 | 显示错误信息，不卡死 |
| 后端挂了 | 后端未启动时登录 | 显示错误，不白屏 |

### 7.2 TodayScreen 四种状态

**Loading 状态：**
- 页面显示 loading 动画 + "加载中..."
- 不应闪现错误或空白

**Empty 状态：**
- 无家庭 / 无设备 / 无 RoomState 数据
- 显示 "暂无数据" + 说明文字
- 显示 wifi 图标

**Error 状态：**
- API 请求失败 → 显示错误卡片
- 显示警告图标 + "加载失败" + 错误信息
- 不应白屏或崩溃

**OK 状态：**
- 显示 "今日状态" 标题
- 每个房间一张卡片
- 卡片显示：房间名、在线/离线状态、当前姿态、心率、呼吸率

### 7.3 姿态显示

| 姿态值 | 中文显示 | 卡片颜色 |
|--------|----------|----------|
| `stand` | 站立 | 正常（primary 左边框） |
| `sit` | 坐着 | 正常 |
| `lie` | 躺卧 | 正常 |
| `walk` | 走动 | 正常 |
| `fall` | 跌倒 | 告警（alert 红色） |
| `null` | 未检测 | 正常 |
| 未知值 | 原文显示 | 正常 |

### 7.4 WebSocket 实时更新验证

**手动测试流程：**

```bash
# 1. 启动后端 + AI placeholder
# 2. App 登录成功，看到 TodayScreen
# 3. 运行模拟器发帧：
python simulator/feed.py rad_e2e e2e_secret --mode walk

# 4. 观察 App 页面：
#    - 姿态应实时更新（~2s 延迟）
#    - 心率/呼吸应每 10s 更新
#    - 设备在线状态应为"在线"

# 5. 切换模式验证：
#    Ctrl+C 停模拟器 → 改 --mode lie → 重新启动
#    观察 App 姿态从"走动"变为"躺卧"
```

**网络异常测试：**

| 场景 | 操作 | 预期 |
|------|------|------|
| WS 断线 | 关闭 daphne | 页面不崩溃，数据保持最后状态 |
| WS 重连 | 重启 daphne | App 需要手动刷新或自动重连 |
| 慢网络 | 限速/弱网 | 不崩溃，数据不错误 |
| 后台切换 | App 切到后台再回来 | 连接恢复，数据应为最新 |

### 7.5 UI 组件验证

**StatusBadge：**
- `online=true` → 绿色圆点 + "在线"
- `online=false` → 灰色圆点 + "离线"

**VitalStat：**
- 心率：红色心形图标 + 数值 + "心率"标签
- 呼吸：脉冲图标 + 数值 + "呼吸"标签

**Card：**
- `tone="normal"` → primary 色左边框
- `tone="alert"` → alert（红）色左边框

### 7.6 退出登录

- 点击"退出登录"→ 关闭 WS 连接 → 返回 LoginScreen
- 不应残留 WebSocket 连接

### 7.7 当前缺失的 App 测试

App 端目前 **没有任何自动化测试**。建议至少补充：

| 优先级 | 测试类型 | 范围 |
|--------|----------|------|
| P0 | 手动冒烟 | 登录 → 看数据 → 实时更新 → 退出 |
| P1 | Jest 单元 | api.ts 的 fetch mock |
| P1 | Jest 单元 | mergeUpdate 合并逻辑 |
| P1 | Jest 单元 | POSTURE_CN 映射完整性 |
| P2 | Detox/E2E | 登录 → 等待 WS 推送 → 验证页面元素 |

---

## 第 8 层：端到端集成

### 8.1 自动化 E2E 脚本

```bash
# 一键全链路验证
bash simulator/e2e_verify.sh
```

**验证阶段：**

| 阶段 | 检查点 | 脚本输出 |
|------|--------|----------|
| 启动 | daphne + AI placeholder 正常启动 | `daphne running (PID=...)` |
| 灌数 | 3 帧点云 + 1 体征帧均 ack | `pc#0: {"ack": "buffered"}` |
| 落库 | RoomState 有数据 | `✅ RoomState found: posture=...` |
| Redis | Stream 有消息 | `redis housafe:pointcloud:ingest: N msgs` |

### 8.2 手动全链路验证 Checklist

每次发版前，按以下顺序完整走一遍：

```
□ 1. 基础设施
  □ docker-compose up -d（Postgres + Redis）
  □ createdb housafe
  □ python manage.py migrate
  
□ 2. 启动服务
  □ daphne -p 8000 housafe.asgi:application（前台，看日志）
  □ python ai/placeholder/main.py（前台，看推理）
  
□ 3. 准备数据
  □ POST /api/auth/register 注册用户
  □ POST /api/auth/login 获取 JWT
  □ POST /api/families 创建家庭
  □ POST /api/families/{id}/devices/bind 绑定设备（记录 device_id + secret）
  
□ 4. 数据链路
  □ python simulator/feed.py <device_id> <secret> --mode walk 启动模拟器
  □ 确认 AI placeholder 输出 [AI] posture ... ok
  □ 确认 RoomState 有数据
  □ curl GET /families/{id}/today 返回正确数据
  
□ 5. 推送链路
  □ wscat 连接 /ws/app
  □ curl POST /api/internal/decoder-output 注入数据
  □ wscat 收到 room.state 推送
  
□ 6. App 验证
  □ npx expo start
  □ 手机扫码 → 登录 → 看到房间卡片
  □ 模拟器切换 mode → App 实时更新
  □ 模拟器停掉 → App 显示设备离线
  
□ 7. 异常场景
  □ 错误密码登录 → 显示错误提示
  □ 停止 AI → 数据停止更新（App 显示旧数据）
  □ 停止 daphne → App 显示加载失败
  □ 重启 daphne → App 手动刷新恢复
```

### 8.3 性能基准

| 指标 | 本地预期 | 生产目标 |
|------|----------|----------|
| 设备帧 → Redis | < 10ms | < 50ms |
| Redis → AI 消费 | < 100ms | < 200ms |
| AI 推理（占位符） | 0.3~1.0s | < 2s（真实模型） |
| POST 回调 → RoomState 落库 | < 20ms | < 50ms |
| group_send → App 收到 | < 50ms（本地） | < 500ms（跨网） |
| 端到端（帧→App） | ~2s | < 5s |

---

## 缺失测试汇总（待补）

按优先级排列，这些是目前测试覆盖的空白：

### P0 - 必须补

| 项目 | 位置 | 测什么 |
|------|------|--------|
| App 冒烟测试 | `app/` | 登录 → 看数据 → WS 更新 |
| 模拟器数据分布验证 | `simulator/` | 各 mode 点云 z 轴分布是否合理 |
| AI 推理边界用例 | `ai/tests/` | 空点云、单点、边界高度 |

### P1 - 应该补

| 项目 | 位置 | 测什么 |
|------|------|--------|
| Ingest 背压/高并发 | `backend/ingest/tests/` | 10 设备同时发帧 |
| WS 断线重连 | `backend/realtime/tests/` | App WS 断后重连收到最新状态 |
| API 权限隔离 | `backend/events/tests/` | 用户 A 不能看用户 B 的 family |
| App mergeUpdate 单测 | `app/` | WS 消息到达后状态合并正确性 |

### P2 - 锦上添花

| 项目 | 位置 | 测什么 |
|------|------|--------|
| App E2E（Detox） | `app/e2e/` | 完整用户旅程自动化 |
| AI placeholder 回调重试 | `ai/tests/` | Backend 挂了后的重试行为 |
| Redis 故障恢复 | `backend/tests/` | Redis 挂掉重启后 Consumer 恢复 |
| Timescale 压缩策略 | `backend/events/tests/` | 快照表压缩后查询正确性 |

---

## 日常开发测试流程

### 日常改动（小）

```bash
# 跑相关模块的测试
cd backend
python -m pytest ingest/tests/ events/tests/ realtime/tests/ -q

# 如果改了 contracts
python -m pytest ../contracts/tests/ -q

# 手动跑一次 e2e
bash ../simulator/e2e_verify.sh
```

### 数据流相关改动（中）

```bash
# 1. 跑全量测试
cd backend && python -m pytest . -q

# 2. 手动启动全链路，逐段验证
#    (见第 8.2 节 checklist)

# 3. 用 wscat 验证推送到 App 的数据格式正确
```

### 发版前（大）

```bash
# 完整 checklist（第 8.2 节）
# + 真机 App 验证（登录 → 看数据 → 实时更新 → 异常场景）
# + 至少发 50 帧验证无内存泄漏/连接泄漏
```

---

## 三终端调试布局

开发/测试时推荐的终端布局：

```
┌──────────────────┬──────────────────┬──────────────────┐
│ Terminal 1       │ Terminal 2       │ Terminal 3       │
│ daphne (前台)    │ AI 占位符 (前台)  │ 交互调试         │
│                  │                  │                  │
│ 看 WS 连接/断开   │ 看每帧推理结果    │ curl / wscat     │
│ 看 consumer 错误  │ 看 HTTP 回调状态  │ 手动灌数据       │
│ 看 SQL 查询       │ 看延迟时间        │ 查 DB / Redis    │
└──────────────────┴──────────────────┴──────────────────┘

               Terminal 4 (可选)
               expo start / 手机 App
               看前端实际渲染效果
```

---

## 快速参考：各层关键命令

```bash
# ── 基础设施 ──
docker-compose up -d                              # 启动 PG + Redis
lsof -i :5432 -i :6379 -P | grep LISTEN           # 确认服务在跑

# ── 后端 ──
cd backend && daphne -p 8000 housafe.asgi:application  # 启动服务

# ── AI ──
python ai/placeholder/main.py                     # 启动推理

# ── 模拟器 ──
python simulator/feed.py rad_e2e e2e_secret --mode walk  # 灌数

# ── 测试 ──
cd backend && python -m pytest . -q               # 全量测试
cd backend && python -m pytest ingest/tests/ -q   # 单模块
bash simulator/e2e_verify.sh                      # E2E 一键

# ── 调试 ──
redis-cli XLEN housafe:pointcloud:ingest          # Stream 长度
redis-cli XREVRANGE housafe:pointcloud:ingest + - COUNT 1  # 最新帧
wscat -c "ws://localhost:8000/ws/app?token=<JWT>&family=1"  # WS 监听
curl -s http://localhost:8000/api/families/1/today -H "Authorization: Bearer <JWT>"  # API 查询

# ── 查库 ──
python -c "
import os,sys;sys.path.insert(0,'backend');os.environ['DJANGO_SETTINGS_MODULE']='housafe.settings'
import django;django.setup()
from events.models import RoomState,Alert
for rs in RoomState.objects.all(): print(f'{rs.device_id} | {rs.room} | posture={rs.posture} | hr={rs.heart_rate}')
print(f'alerts: {Alert.objects.count()}')
"
```
