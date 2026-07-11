# 家安 housafe · 子项目①「地基」设计文档

> 版本 v1.0 | 2026-07-11
> 范围：整个软件产品的第一个子项目——数据契约 + 后端骨架 + 最小 App 壳。
> 定位来源：本阶段无硬件，用「仿真器」长期扮演硬件层，软件全部做成**产品级成品**（非 demo）。
> 上层文档：《家安_PRD_总纲_v1.md》《家安_PRD_后端_v1.md》《家安_核心算法设计文档_v2.md》。

---

## 0. 背景与总策略

家安是 4 子系统产品（端侧点云采集 / 后端 / AI 服务 ALG-0~6 / 手机 App）。当前无硬件，决定**从仿真开始，把硬件之外的整个软件做成可用好用的产品**。

**唯一诚实约束**：ALG-0（点云→姿态/跌倒的感知模型）必须用真实雷达点云训练验证，此阶段做不成真成品。解法是把 ALG-0 定义为**感知适配层（防腐层）**：

- 现在：仿真器直接产出符合「事件契约」的感知事件，等价于一个"完美的 ALG-0"。
- 硬件到位后：用真实点云模型替换适配层，**上层（ALG-1~6 / 后端 / App）一行不改**。

于是仿真雷达与真实雷达对系统**可插拔**；这不是妥协，是正确解耦。

**构建顺序（各子项目独立 spec→plan→build）**：
1. **地基**（本文）：契约 + 后端骨架 + 最小 App 壳
2. 仿真器：场景引擎，产出事件流，支持时间加速
3. AI 服务 ALG-1~6
4. 通知系统：分级下发 + 升级
5. App：完整子女观测端

---

## 1. 技术栈（终局锁定）

| 层 | 选型 | 理由 |
|---|---|---|
| App | **React Native + Expo** | 一套码出 iOS+Android 原生，手机为主、多端可扩，产品级体验 |
| 后端 | **Django + DRF + Channels** | 电池全含（账户/权限/Admin/迁移）；Channels 同时解决"仿真器灌事件进"与"后端推状态给 App" |
| AI 服务 | **Python，同仓独立 worker** | ALG-1~6 与后端同为 Python，同仓不拆微服务（仿真期拆分=过度设计） |
| 存储 | **PostgreSQL + TimescaleDB** | 事件是时序数据，Timescale 是 Postgres 扩展，Django ORM 直接可用 |
| 通道层/缓存 | **Redis** | Channels channel-layer + AI/实时的事件分发 |

> 高吞吐点云接入是 Django 短板，但**仿真接缝在"事件契约"而非原始点云**——仿真器灌的是结构化事件（量很小），Django/Channels 完全够。真硬件 20fps 点云接入将来单独做成可替换 ingest 模块，不动上层。

---

## 2. 仓库形态（monorepo）

```
housafe/
├── contracts/        ★ 事件契约唯一来源（Pydantic + JSON Schema），sim/backend/ai 共用同一份
├── backend/          Django 项目（产品后端）
│   ├── accounts/     子女账户 注册/登录/鉴权
│   ├── families/     家庭 / 被监测老人 / 紧急联系人 / 通知顺序
│   ├── devices/      雷达设备 注册/绑定/在线状态/心跳/OTA触发(桩)
│   ├── ingest/       ★ 接入网关（Channels WS）：收事件→鉴权→校验契约→落库→分发
│   ├── events/       时序事件存储（Timescale hypertable）+ 查询 API
│   └── realtime/     向 App 推「今日状态/事件」的 WS
├── ai/               ALG-1~6（子项目③，本阶段仅留空目录+接口占位）
├── simulator/        场景引擎（子项目②，本阶段仅留 1 个最小灌数脚本用于自测）
├── app/              React Native + Expo（本阶段仅最小壳）
├── docs/
└── docker-compose.yml   Postgres+Timescale / Redis 本地一键起
```

---

## 3. 事件契约（防腐层，本子项目的技术核心）

契约分两层，**单一事实来源在 `contracts/`**，三方（sim/backend/ai）import 同一份，杜绝各写各的。

**Tier-0 原始点云（本阶段不产不消费，仅定义占位）**
```
点云帧 {ts, radar_id, room, frame_id, points:[{x,y,z,velocity,intensity}]}
```
> 只有真实 ALG-0 存在时才用到。定义在此是为了将来接硬件时契约已就位。

**Tier-1 感知事件（仿真器现在产出、ALG-1~6 将来消费）—— 就是"完美 ALG-0"的输出**
```
存在事件 {ts, ts_recv, radar_id, room, seq, presence:bool, moving:bool}
姿态事件 {ts, ts_recv, radar_id, room, seq, posture:enum[stand|sit|lie|walk|fall], confidence}
生命体征 {ts, ts_recv, radar_id, room, seq, quiet:bool, resp_rate, heart_rate, quality}
人数估计 {ts, ts_recv, radar_id, room, seq, count}
心跳     {ts, radar_id, status:enum[online|offline], fw_version}
```

**关键设计（为将来 ALG-1 多雷达对齐 / 仿真时间加速一次到位）：**
- `ts`：事件源权威时间（UTC 毫秒），**由 sim/设备打**，非服务器当前时间 → 支持回放与时间加速。
- `ts_recv`：服务器接收时间 → 供 ALG-1 估计时钟漂移。
- `seq`：单雷达自增序号 → 供去重与乱序恢复。
- 落库/查询一律以 `ts` 为时间轴，服务器绝不用"现在"覆盖。

---

## 4. 数据模型（骨架）

- **accounts**：`User`（子女，Django 用户 + JWT 鉴权，simplejwt）。
- **families**：`Family`、`Elder`（被监测者，无登录）、`Contact`（紧急联系人，含 `order` 通知顺序）、`Membership`（User↔Family，含角色）。
- **devices**：`RadarDevice`（`device_id`、`secret`、`room`、`family` FK、`online`、`last_heartbeat`、`fw_version`）。
- **events**：`PresenceEvent` / `PostureEvent` / `VitalEvent` / `OccupancyEvent` 各一张表，迁移中 `RunSQL` 转 Timescale hypertable（按 `ts` 分片）。

> 老人无账户、雷达无身份识别——严格遵守 PRD"独居假设"，模型不预留身份识别字段。

---

## 5. 接口（DRF REST + Channels WS）

**REST（App 用）**
- 鉴权：注册 / 登录 / 刷新（JWT）
- 家庭：Family CRUD、Elder/Contact 增删改、通知顺序
- 设备：绑定（扫码配网 mock——返回 device_id+secret）、列表、重命名、在线状态、指示灯开关(桩)、OTA 触发(桩)
- 事件：今日状态聚合（最小）、事件历史查询（按房间/时间/类型）

**WS（Channels）**
- `ingest`：设备侧接入。`device_id`+`secret` 鉴权 → 收事件 → 校验契约 → 落库 → 发 Redis（供 AI 与 realtime）
- `realtime`：App 侧。JWT 鉴权 → 订阅本家庭 → 实时收「存在/姿态/生命体征」推送

**数据流**
```
simulator ──契约──▶ ingest(WS) ──▶ events(Timescale) ──▶ realtime(WS) ──▶ App
                        └────────▶ Redis ──▶ ai worker（子项目③接，本阶段占位）
```

---

## 6. 最小 App 壳（地基交付内的 App 部分）

只做到"证明端到端打通"，完整 App 是子项目⑤：
- 登录
- 设备列表 + 在线状态
- 「今日状态」单屏：实时显示仿真器灌进来的存在/姿态/生命体征（走 realtime WS）

产品级要求体现在：真实鉴权、真实 WS、真实错误处理与加载态，不是假数据写死。

---

## 7. 本子项目「做完你能看到什么」

1. 注册登录、建家庭、加老人与紧急联系人（真实可用）
2. "绑定"一台仿真雷达，App 里看到它在线/离线
3. 跑一个最小灌数脚本，App「今日状态」屏**实时**跳出"卧室有人、站立→走动、呼吸 16/心率 72"
4. 数据落时序库，可按时间/房间查历史
5. 一套真实、有鉴权、有错误处理的**产品地基**——AI 告警、通知、完整 App 都往这上面长

---

## 8. 明确不在本子项目范围（YAGNI / 留给后续）

- Tier-0 原始点云的产/消费、真实 ALG-0（等硬件）
- ALG-1~6 算法逻辑（子项目③；本阶段仅占位接口 + Redis 分发通道）
- 通知短信/电话下发、升级策略（子项目④）
- 完整 App 的趋势/报告/告警中心（子项目⑤）
- OTA 真实固件流、MQTT 边缘协议（真硬件阶段）
- 报告 PDF、LLM 报告（PRD 标 P2）

---

## 9. 测试与验收

- **contracts**：契约 schema 校验测试（合法/非法样本）
- **backend**：DRF API 测试（accounts/families/devices/events 查询）；ingest WS 测试（灌样本事件→断言落库正确、以 `ts` 为轴）
- **端到端冒烟**：最小脚本以"设备"身份连 ingest 发若干事件 → 断言经 realtime/查询 API 能被 App 侧取到
- **验收标准**：§7 的 5 条能真实跑通；关键路径有自动化测试；`docker-compose up` 后本地一键复现

---

## 10. 非功能与约束

- 设备接入 `device_id`+`secret` over WSS；App 侧 JWT
- 事件以 `ts` 为权威时间轴（支撑回放/时间加速/多雷达对齐）
- 事件带 `seq`，ingest 做最小去重与乱序容忍
- 合规对齐 PRD：点云不落盘（本阶段无点云）；事件保留 90 天（保留期策略占位，删除/导出接口留到子项目④细化）
- 传输 TLS、存储加密：生产部署项，本地开发不阻塞

---

## 11. 开放项（不阻塞开工，实现时定）

- OQ-F1：JWT 具体库与刷新策略（倾向 simplejwt）
- OQ-F2：ingest 用纯 Channels WS 还是加一层轻量鉴权握手协议——实现时定
- OQ-F3：今日状态聚合的最小字段集，随最小 App 壳一起定
- OQ-F4：Timescale 保留期/压缩策略参数（占位，非本阶段调优）
