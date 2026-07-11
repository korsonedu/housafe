### Task 2: 事件契约包（contracts）

**Files:**
- Create: `contracts/pyproject.toml`, `contracts/housafe_contracts/__init__.py`
- Create: `contracts/housafe_contracts/events.py`
- Test: `contracts/tests/test_events.py`

**Interfaces:**
- Produces:
  - `POSTURES = ("stand","sit","lie","walk","fall")`
  - Pydantic 模型：`PresenceEvent`、`PostureEvent`、`VitalEvent`、`OccupancyEvent`、`Heartbeat`
  - 每个事件含 `ts:int`（UTC ms）、`radar_id:str`、`room:str`、`seq:int`（心跳无 seq/room）
  - `parse_event(kind:str, payload:dict) -> BaseModel`：按 `kind` 分发校验，非法抛 `ValidationError`
  - `EVENT_KINDS = ("presence","posture","vital","occupancy","heartbeat")`

- [ ] **Step 1: 写失败测试**

`contracts/tests/test_events.py`:
```python
import pytest
from pydantic import ValidationError
from housafe_contracts.events import parse_event, PostureEvent, POSTURES

def test_posture_valid():
    e = parse_event("posture", {"ts":1700000000000,"radar_id":"r1","room":"bedroom","seq":5,"posture":"walk","confidence":0.9})
    assert isinstance(e, PostureEvent) and e.posture == "walk"

def test_posture_rejects_bad_enum():
    with pytest.raises(ValidationError):
        parse_event("posture", {"ts":1,"radar_id":"r1","room":"b","seq":1,"posture":"jump","confidence":0.5})

def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        parse_event("nope", {})

def test_postures_frozen():
    assert POSTURES == ("stand","sit","lie","walk","fall")
```

- [ ] **Step 2: 运行确认失败**

Run: `cd contracts && pip install -e . && pytest -q`
Expected: FAIL（`housafe_contracts.events` 不存在）。

- [ ] **Step 3: 实现契约**

`contracts/housafe_contracts/events.py`:
```python
from typing import Literal
from pydantic import BaseModel, Field

POSTURES = ("stand","sit","lie","walk","fall")
EVENT_KINDS = ("presence","posture","vital","occupancy","heartbeat")

class _Base(BaseModel):
    ts: int = Field(..., description="UTC ms, source-authoritative")
    radar_id: str
    room: str
    seq: int

class PresenceEvent(_Base):
    presence: bool
    moving: bool

class PostureEvent(_Base):
    posture: Literal[POSTURES]
    confidence: float = Field(..., ge=0, le=1)

class VitalEvent(_Base):
    quiet: bool
    resp_rate: float | None = None
    heart_rate: float | None = None
    quality: float = Field(..., ge=0, le=1)

class OccupancyEvent(_Base):
    count: int = Field(..., ge=0)

class Heartbeat(BaseModel):
    ts: int
    radar_id: str
    status: Literal["online","offline"]
    fw_version: str

_MODELS = {"presence":PresenceEvent,"posture":PostureEvent,"vital":VitalEvent,
           "occupancy":OccupancyEvent,"heartbeat":Heartbeat}

def parse_event(kind: str, payload: dict) -> BaseModel:
    if kind not in _MODELS:
        raise ValueError(f"unknown event kind: {kind}")
    return _MODELS[kind](**payload)
```

`contracts/pyproject.toml`:
```toml
[project]
name = "housafe-contracts"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["pydantic>=2.6"]
```

> 注：`Literal[POSTURES]` 需 `Literal[*POSTURES]`（Python 3.12 语法）。`Literal[tuple]` 是 3.12 支持的正确写法 — 因为 `POSTURES` 是 tuple 字面量。若报错 `Literal[...] parameters must be literal types`，改为逐字写 `Literal["stand","sit","lie","walk","fall"]` 并保留 `POSTURES` 常量。

- [ ] **Step 4: 运行确认通过**

Run: `cd contracts && pytest -q`
Expected: PASS（4 passed）。

- [ ] **Step 5: 后端引用契约并提交**

在 `backend/pyproject.toml` dependencies 加 `"housafe-contracts"`，并 `pip install -e ../contracts`。

```bash
git add -A && git commit -m "feat: add event contract package (tier-1 perception events)"
```

## Global Constraints (relevant)

- Python ≥ 3.12；Pydantic ≥ 2.6。
- 事件契约唯一来源是 `contracts/` 包，backend / simulator / ai 三方 import 同一份，禁止各自复制字段定义。
- 姿态枚举固定为 `stand | sit | lie | walk | fall`；房间为自由字符串 label。
- 事件时间轴一律用事件自带的 `ts`（UTC 毫秒 int），服务器**绝不**用"当前时间"覆盖；服务器接收时间单独存 `ts_recv`。
- 每个任务结束必须 commit；提交信息用 `feat:` / `test:` / `chore:` 前缀。
