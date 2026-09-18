# 家安 housafe

非接触式居家养老 AI 健康监测系统。毫米波雷达组网 → 点云上云 → 世界模型推理 (Encoder 4.7M + LatentBaseline) → 子女 App 实时掌握父母状态。

**核心能力：** 规则引擎做确定性检测（跌倒/体征/静止），世界模型做概率检测（行为模式偏离个人常轨）——后者是竞品做不到的。

**一句话：让异地子女 30 秒看懂父母今天在家过得好不好，异常时第一时间知道。**

---

## 架构总览

```
雷达设备（TI IWR6843）              云端                 子女 App（Expo / RN）
  │                                  │                        │
  ├─ WiFi ──► IngestConsumer ──► Redis Stream ──► AI Encoder  │
  │  (ws/ingest)  点云帧/体征帧     环形缓冲        世界模型Worker │
  │                                  │                        │
  │                            AIBridgeView ◄── POST          │
  │                           (REST callback)                 │
  │                                  │                        │
  │                          RoomState / Alert                │
  │                          (PostgreSQL 16)                  │
  │                                  │                        │
  │                          AppConsumer ──── WS ────────────►│
  │                          (ws/app)    room.state 推送      │
```

| 组件 | 技术栈 | 职责 |
|------|--------|------|
| **contracts** | Pydantic ≥ 2.6 | 全局契约定义（帧模型、Decoder 输出、查询形状），唯一来源 |
| **backend/ingest** | Django Channels 4 | 设备 WebSocket 接入：鉴权、帧校验、XADD Redis Stream |
| **backend/events** | Django + DRF | 房间状态 upsert、异常告警存储、AI 回调接口、Today 查询 |
| **backend/realtime** | Django Channels 4 | App WebSocket：JWT 鉴权、家庭隔离、room.state 实时推送 |
| **backend/accounts** | DRF + SimpleJWT | 用户注册/登录/JWT 签发与刷新 |
| **backend/families** | Django ORM | 家庭、被测者、联系人 CRUD，owner 隔离 |
| **backend/devices** | Django ORM | 雷达设备绑定/列表/重命名、credential 签发与鉴权 |
| **ai/main** | Python + PyTorch + redis-py | 世界模型 Worker：XREAD Redis Stream → Encoder+Predictor+LatentBaseline → POST 回调 |
| **simulator** | Python + websockets | 设备端模拟器：发送合成点云帧 + 体征帧 + 心跳 |
| **app** | Expo SDK 57 (React Native) | 子女端：JWT 登录、今日房间状态、实时推送 |

---

## 前置条件

- **Python** ≥ 3.12
- **PostgreSQL** 16+（推荐 TimescaleDB，非必须）
- **Redis** 7
- **Node.js** 20+（App 端）
- **Docker**（可选，用 docker-compose 一键起基础设施）

---

## 快速启动

### 1. 基础设施

```bash
# Docker 方式（推荐）
docker-compose up -d          # 启动 Postgres + Redis

# 或者用本地 Homebrew
brew install postgresql@16 redis
brew services start postgresql@16 redis
```

创建数据库：

```bash
createdb housafe -U postgres
```

### 2. 后端

```bash
cd backend
python3.12 -m venv .venv && source .venv/bin/activate

# 安装依赖
pip install -e ".[dev]"
pip install -e ../contracts
pip install redis>=5.0 websockets

# 数据库迁移
python manage.py migrate

# 启动 ASGI 服务器
daphne -p 8000 housafe.asgi:application
```

### 3. AI 世界模型 Worker

```bash
cd ../
python ai/main.py
# 默认连接 redis://localhost:6379/0，回调 http://localhost:8000
# 自动加载 Encoder + Predictor + LatentBaseline
```

### 4. 仿真灌数（测试用）

```bash
# 先通过 API 注册用户、创建家庭、绑定设备
# 然后运行模拟器发送点云帧
python simulator/feed.py <device_id> <secret> --mode walk
```

### 5. App（前端的端）

```bash
cd app
npm install

# 编辑 src/api.ts 修改 API_BASE / WS_BASE 为本机 IP
npx expo start
# 手机 Expo Go 扫码，或按 i（iOS）/ a（Android）模拟器
```

---

## 项目结构

```
housafe/
├── contracts/                 # Pydantic 契约（唯一数据定义源）
│   └── housafe_contracts/
│       └── events.py          # 帧模型、Decoder 输出、RoomState、parse_frame()
├── backend/                   # Django ASGI 后端
│   ├── housafe/               # 项目配置（settings, asgi, urls）
│   ├── accounts/              # 用户注册/登录/JWT
│   ├── families/              # 家庭/被测者管理
│   ├── devices/               # 雷达设备管理
│   ├── ingest/                # 设备 WebSocket 接入网关
│   ├── events/                # 状态存储、Decoder 回调、查询 API
│   ├── realtime/              # App WebSocket 实时推送
│   ├── tests/                 # e2e 冒烟测试
│   └── pyproject.toml
├── ai/
│   ├── main.py                # WorldModelWorker：消费 Redis → 推理 → POST backend
│   ├── encoder/               # Encoder (PointNet++ + TCN, 4.7M)
│   ├── predictor/             # Predictor (GRU, 0.3M) + 训练脚本
│   ├── baseline/              # LatentBaseline (GMM on 256-d S_t) + PersonalBaseline
│   ├── latent_space/          # SpatialGraph (房间拓扑)
│   ├── decoders/              # AnomalyDecoder / NotificationDecider / ReportDecoder
│   ├── fallback/              # FallbackEngine (确定性规则)
│   ├── shared/                # 共享类型 (FeatureVector, AnomalyResult, ...) + Redis client
│   └── checkpoints/           # 模型 checkpoint（encoder/predictor/latent_baseline）
├── simulator/
│   ├── feed.py                # 设备端模拟器
│   └── e2e_verify.sh          # 端到端验证脚本
├── app/
│   ├── App.tsx                # 入口
│   ├── src/
│   │   ├── theme.ts           # 设计 token
│   │   ├── api.ts             # API 端点配置
│   │   ├── LoginScreen.tsx    # JWT 登录
│   │   └── TodayScreen.tsx    # 今日状态总览 + WS 实时推送
│   └── package.json
└── docker-compose.yml         # Postgres 16 (TimescaleDB) + Redis 7
```

---

## API 概览

### REST API

| 方法 | 路径 | 鉴权 | 说明 |
|------|------|------|------|
| POST | `/api/auth/register` | - | 用户注册 |
| POST | `/api/auth/login` | - | JWT 登录 |
| POST | `/api/auth/refresh` | JWT | Token 刷新 |
| GET | `/api/families` | JWT | 我的家庭列表 |
| POST | `/api/families` | JWT | 创建家庭 |
| POST | `/api/families/{id}/devices/bind` | JWT | 绑定雷达设备 |
| GET | `/api/families/{id}/devices` | JWT | 设备列表 |
| GET | `/api/families/{id}/today` | JWT | 今日房间状态 |
| POST | `/api/internal/decoder-output` | Internal Token | AI 推理结果回调 |

### WebSocket

| 路径 | 协议 | 鉴权 | 说明 |
|------|------|------|------|
| `/ws/ingest` | JSON frames | device_id + secret | 设备端点云/体征/心跳上报 |
| `/ws/app` | JSON push | JWT (query string) | App 端实时状态推送 |

#### Ingest 帧格式

```json
// 鉴权（仅首帧）
{"device_id": "rad_xxx", "secret": "..."}
// → {"ack": "auth"}

// 点云帧
{"kind": "point_cloud", "payload": {"ts": 1700000000000, "radar_id": "rad_xxx", "room": "bedroom", "frame_id": "f-001", "points": [{"x":1.0, "y":0.5, "z":1.5, "velocity":0.1, "intensity":0.9}]}}
// → {"ack": "buffered", "frame_id": "f-001"}

// 体征帧
{"kind": "vital", "payload": {"ts": 1700000000000, "radar_id": "rad_xxx", "room": "bedroom", "quiet": true, "resp_rate": 16.0, "heart_rate": 72.0, "quality": 0.85}}
// → {"ack": "buffered", "frame_id": "1700000000000"}

// 心跳帧
{"kind": "heartbeat", "payload": {"ts": 1700000000000, "radar_id": "rad_xxx", "status": "online", "fw_version": "1.2.3"}}
// → {"ack": "heartbeat"}
```

#### App 推送格式

```json
// v3.0 room.state（当前协议）
{"type": "room.state", "room_name": "bedroom", "device_online": true, "posture": "walk", "posture_ts": 1700000000000, "heart_rate": 72.0, "breath_rate": 16.0, "anomaly_score": 0.02}

// v2.x 离散事件（向后兼容）
{"kind": "posture", "payload": {"room": "bedroom", "posture": "walk", "ts": 1700000000000}}
{"kind": "vital", "payload": {"room": "bedroom", "heart_rate": 72.0, "resp_rate": 16.0}}
```

---

## 数据模型

### 合同定义的姿态/告警常量

```python
POSTURES = ("stand", "sit", "lie", "walk", "fall")
ALERT_TYPES = ("fall", "stillness", "vital_anomaly", "pattern_deviation", "offline")
SEVERITY_LEVELS = ("info", "warning", "critical")
```

### 数据库表（events app）

| 表 | 模式 | 说明 |
|----|------|------|
| `RoomState` | per-device upsert | 房间最新状态（姿态、体征、异常分、人数） |
| `Alert` | append-only | 异常通知事件 |
| `StateSnapshot` | ~1/min 快照 | 用于趋势报告 |

---

## 数据流

```
设备 point_cloud 帧
  → IngestConsumer (WS 鉴权 + Pydantic 校验)
    → XADD housafe:pointcloud:ingest (Redis Stream, MAXLEN ~1000)
      → WorldModelWorker XREAD → Encoder(点云→S_t) → LatentBaseline(密度异常分)
        → FallbackEngine(规则) → AnomalyDecoder(融合) → NotificationDecider
        → POST /api/internal/decoder-output {token, outputs: [...]}
          → AIBridgeView (Internal Token 鉴权)
            → store_decoder_output() → RoomState upsert / Alert create
            → channel_layer.group_send(family_{id}, room.state) → AppConsumer → 手机
```

---

## 配置与环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DATABASE_URL` | `postgres://housafe:housafe@localhost:5432/housafe` | PG 连接串 |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis 连接串 |
| `SECRET_KEY` | (开发默认值) | Django SECRET_KEY，**生产必改** |
| `INTERNAL_SERVICE_TOKEN` | `dev-internal-token` | AI 占位符 → 后端 M2M 鉴权，**生产必改** |

---

## 调试与产品测试

### 单元/集成测试

```bash
cd backend

# 全量测试
python -m pytest . -q

# 单模块测试
python -m pytest ingest/tests/ -q
python -m pytest events/tests/ -q
python -m pytest realtime/tests/ -q

# contracts 测试
python -m pytest ../contracts/tests/ -q
```

### 端到端自动化验证

一条命令跑通全链路：启动服务 → 灌数 → AI 推理 → 落库校验。

```bash
bash simulator/e2e_verify.sh
```

验证内容：

| 阶段 | 检查点 | 脚本输出标记 |
|------|--------|-------------|
| ingest 接入 | WS 鉴权 + 帧到达 | `auth: {"ack": "auth"}` |
| Redis 缓冲 | 点云帧写入 Stream | `pc#0: {"ack": "buffered"}` |
| AI 推理 | 占位符消费 + 回调 | daphne 无 1011 错误 |
| 落库 | RoomState 更新 | `✅ RoomState found: posture=...` |
| 实时推送 | group_send → AppConsumer | daphne 无 `No handler` 错误 |

### 手动逐段调试

调试问题时，不要一把梭——按数据流方向逐环节排查。

#### 第 1 段：基础设施

```bash
# 确认 PG 和 Redis 都在跑
lsof -i :5432 -i :6379 -P | grep LISTEN

# 检查数据库表是否存在
python manage.py dbshell
> \dt events_*
> SELECT * FROM events_roomstate;
```

#### 第 2 段：ingest WebSocket 接入

```bash
# 前台启动 daphne（看实时日志）
daphne -p 8000 housafe.asgi:application

# 另开终端，手动发一帧测试
python -c "
import asyncio, json
import websockets

async def test():
    async with websockets.connect('ws://localhost:8000/ws/ingest') as ws:
        await ws.send(json.dumps({'device_id':'rad_e2e','secret':'e2e_secret'}))
        print('auth:', await ws.recv())
        await ws.send(json.dumps({'kind':'point_cloud','payload':{
            'ts':1700000000000,'radar_id':'rad_e2e','room':'living',
            'frame_id':'debug-1',
            'points':[{'x':1,'y':0.5,'z':1.5,'velocity':0.1,'intensity':0.9}]
        }}))
        print('ack:', await ws.recv())
asyncio.run(test())
"
# 预期：auth → {"ack": "auth"}，然后 → {"ack": "buffered", "frame_id": "debug-1"}
```

#### 第 3 段：Redis Stream 缓冲

```bash
# 查看 Stream 中有多少待处理帧
python -c "
import redis
r = redis.Redis.from_url('redis://localhost:6379/0')
for s in r.keys('housafe:*'):
    info = r.xinfo_stream(s)
    print(f'{s.decode()}: {info[\"length\"]} msgs, first={info[\"first-entry\"][0].decode()}, last={info[\"last-entry\"][0].decode()}')
"

# 读取最新一帧的内容（看 AI 收到了什么）
python -c "
import redis, json
r = redis.Redis.from_url('redis://localhost:6379/0')
msgs = r.xrevrange('housafe:pointcloud:ingest', count=1)
for mid, data in msgs:
    print(f'id={mid.decode()}')
    for k,v in data.items():
        val = v.decode() if k != b'points' else f'{len(json.loads(v))} points'
        print(f'  {k.decode()}={val}')
"

# 清空 Stream（重置测试环境）
python -c "import redis; r=redis.Redis.from_url('redis://localhost:6379/0'); r.delete('housafe:pointcloud:ingest','housafe:vital:ingest'); print('cleared')"
```

#### 第 4 段：AI 世界模型推理

```bash
# 前台启动 AI Worker（看每次推理的输入输出）
python ai/main.py
# 输出示例：
# [WorldModel] Models loaded: encoder (4.7M) + predictor (0.3M)
# [WorldModel] LatentBaseline loaded: 6 contexts, ready=True
# [WorldModel] 12:00:00 bedroom walk → pattern_deviation/warning (score=0.85, latent=0.92)

# 用 curl 直接调 AI 回调接口（跳过 AI，手动注入推理结果）
curl -s -X POST http://localhost:8000/api/internal/decoder-output \
  -H "Content-Type: application/json" \
  -d '{
    "token": "dev-internal-token",
    "outputs": [{
      "kind": "posture",
      "family_id": 1,
      "payload": {"ts": 1700000000000, "device_id": "rad_e2e", "room": "living", "posture": "walk", "confidence": 0.95}
    }]
  }' | python -m json.tool
# 预期：{"ack": "stored", "count": 1}
```

#### 第 5 段：数据库落库

```bash
# 直接查 RoomState
python -c "
import os, sys
sys.path.insert(0, 'backend')
os.environ['DJANGO_SETTINGS_MODULE'] = 'housafe.settings'
import django; django.setup()
from events.models import RoomState, Alert
rs = RoomState.objects.first()
if rs:
    print(f'device={rs.device_id} room={rs.room}')
    print(f'posture={rs.posture} confidence={rs.confidence}')
    print(f'heart_rate={rs.heart_rate} resp_rate={rs.resp_rate}')
    print(f'anomaly_score={rs.anomaly_score}')
    print(f'ts={rs.ts}')
else:
    print('RoomState is empty — AI 可能还没消费')
alerts = Alert.objects.all()
print(f'alerts: {alerts.count()} rows')
"

# 或者进 Django shell
python manage.py shell
>>> from events.models import RoomState, Alert
>>> RoomState.objects.values()
>>> Alert.objects.all()
```

#### 第 6 段：App 实时推送

```bash
# 用 wscat 模拟 App 连接，观察推送内容
npm install -g wscat
wscat -c "ws://localhost:8000/ws/app?token=<JWT>&family=1"

# 然后在另一个终端触发一次推理回调（curl 调 AIBridgeView，上一步的 curl 命令）
# wscat 终端应该立即收到：
# {"type":"room.state","room_name":"living","device_online":true,"posture":"walk",...}
```

### 典型调试场景

**"模拟器发帧成功，但 App 看不到数据"**

```bash
# 1. 查 Stream 有没有帧
redis-cli XLEN housafe:pointcloud:ingest
# 2. 查 AI placeholder 是否在跑
ps aux | grep "ai/placeholder"
# 3. 查 RoomState 有没有落库（上面第 5 段的命令）
# 4. 查 App WebSocket token 是否有效
python -c "
from rest_framework_simplejwt.tokens import AccessToken
token = AccessToken('你的JWT')
print(token['user_id'], token.payload)
"
```

**"daphne 报 1011 internal error"**

通常是 consumer 某个 handler 缺失。检查 daphne 前台日志中的 traceback：

```bash
# 前台跑 daphne，重现操作，看错误栈
daphne -p 8000 housafe.asgi:application
# 常见原因：
# - IngestConsumer 或 AppConsumer 缺少 room_state / event_push 等 handler
# - Redis 连不上（检查 REDIS_URL）
# - DB schema 未迁移（python manage.py migrate）
```

**"Redis Stream 堆积但 AI 没处理"**

```bash
# 确认 AI placeholder 用正确的 stream 名和 $ 起始 ID
# 检查 INTERNAL_SERVICE_TOKEN 是否匹配
echo $INTERNAL_SERVICE_TOKEN   # AI placeholder 用的
# 对比 backend events/views.py 中的 INTERNAL_SERVICE_TOKEN 默认值
grep INTERNAL_SERVICE_TOKEN backend/events/views.py
```

### 监控面板

开发时用三终端布局观察全链路：

```
┌──────────────────┬──────────────────┬──────────────────┐
│ Terminal 1       │ Terminal 2       │ Terminal 3       │
│ daphne (前台)    │ AI 占位符 (前台)  │ 交互调试         │
│                  │                  │                  │
│ 看 WS 连接/断开   │ 看每帧推理结果    │ curl / wscat     │
│ 看 consumer 错误  │ 看 HTTP 回调状态  │ 手动灌数据       │
│ 看 SQL 查询       │ 看延迟时间        │ 查 DB / Redis    │
└──────────────────┴──────────────────┴──────────────────┘
```



---

## 设计约束

- **独居假设**：MVP 按单人建模，不做身份识别
- **无摄像头**：纯毫米波雷达，隐私硬边界
- **点云不落库**：原始点云帧处理后将数据丢弃，仅存推理结果（合规）
- **事件时间轴**：`ts`（UTC ms）全程由设备端提供，服务器仅记录 `ts_recv`
- **契约唯一来源**：`contracts/` 包定义所有数据结构，backend/ai/simulator 共同依赖
- **模块解耦**：ingest/realtime 是仅有的编排层；其余模块单一职责

---

## 相关文档

| 文档 | 说明 |
|------|------|
| `docs/ai-system-status-2026-08-01.md` | **AI 系统当前状态** — 架构、实验记录、指标、决策 |
| `家安_PRD_总纲_v1.md` | 产品全局定位、功能边界、分期、算法总览 |
| `家安_PRD_后端_v1.md` | 后端子系统 PRD |
| `家安_PRD_端侧_v1.md` | 雷达设备端 PRD |
| `家安_PRD_AI服务_v1.md` | AI 服务 PRD |
| `家安_PRD_App_v1.md` | 子女 App PRD |
| `家安_核心算法设计文档_v2.md` | 算法技术方案 |
| `家安_战略论证_v3.md` | 商业模式与战略论证 |
| `bp/家安housafe_BP内容_v1.md` | 商业计划书内容底稿 |
