### Task 7: 接入网关 WS（ingest）

**Files:**
- Create: `backend/ingest/__init__.py`, `apps.py`, `consumers.py`, `routing.py`
- Create: `backend/housafe/asgi.py`（改造为 ProtocolTypeRouter）
- Modify: settings（加 `ingest`）
- Test: `backend/ingest/tests/test_ingest.py`

**Interfaces:**
- Consumes: `RadarDevice.verify`（Task 5）、`parse_event`（Task 2）、`store_event`（Task 6）
- Produces:
  - WS 端点 `ws/ingest`：首帧 `{device_id, secret}` 鉴权；后续帧 `{kind, payload}`；校验契约→`store_event`→`group_send("family_<id>", {...})` 供 realtime
  - 鉴权失败 close code 4401；契约非法回 `{error:"invalid"}` 不落库

- [ ] **Step 1: 写失败测试（Channels Communicator）**

`backend/ingest/tests/test_ingest.py`:
```python
import pytest
from channels.testing import WebsocketCommunicator
from housafe.asgi import application
pytestmark = pytest.mark.django_db(transaction=True)

async def _device(db):
    from django.contrib.auth.models import User
    from families.models import Family
    from devices.models import RadarDevice
    u = await User.objects.acreate_user("k","","pw12345678")
    fam = await Family.objects.acreate(name="f", owner=u)
    return await RadarDevice.objects.acreate(device_id="d1", secret="s1", family=fam, room="bed"), fam

@pytest.mark.asyncio
async def test_bad_secret_closes(db):
    await _device(db)
    c = WebsocketCommunicator(application, "/ws/ingest")
    await c.connect()
    await c.send_json_to({"device_id":"d1","secret":"WRONG"})
    msg = await c.receive_output()
    assert msg["type"] == "websocket.close"

@pytest.mark.asyncio
async def test_valid_event_persists(db):
    dev, _ = await _device(db)
    c = WebsocketCommunicator(application, "/ws/ingest")
    await c.connect()
    await c.send_json_to({"device_id":"d1","secret":"s1"})
    await c.receive_json_from()  # ack
    await c.send_json_to({"kind":"posture","payload":{"ts":1700000000000,"radar_id":"d1","room":"bed","seq":1,"posture":"walk","confidence":0.9}})
    await c.receive_json_from()  # stored ack
    from events.models import PostureRow
    assert await PostureRow.objects.filter(device_id="d1").acount() == 1
    await c.disconnect()
```

- [ ] **Step 2: 运行确认失败**

Run: `cd backend && pytest ingest -q`
Expected: FAIL（asgi/consumer 缺失）。

- [ ] **Step 3: 实现 consumer + routing + asgi**

`backend/ingest/consumers.py`:
```python
import time
from channels.generic.websocket import AsyncJsonWebsocketConsumer
from channels.db import database_sync_to_async
from pydantic import ValidationError
from housafe_contracts.events import parse_event
from devices.models import RadarDevice
from events.store import store_event

class IngestConsumer(AsyncJsonWebsocketConsumer):
    async def connect(self):
        self.device = None
        await self.accept()

    @database_sync_to_async
    def _verify(self, did, secret): return RadarDevice.verify(did, secret)
    @database_sync_to_async
    def _family_id(self): return self.device.family_id
    @database_sync_to_async
    def _store(self, kind, model): store_event(self.device.device_id, kind, model, ts_recv=int(time.time()*1000))

    async def receive_json(self, content):
        if self.device is None:
            self.device = await self._verify(content.get("device_id"), content.get("secret"))
            if self.device is None:
                await self.close(code=4401); return
            self.group = f"family_{await self._family_id()}"
            await self.channel_layer.group_add(self.group, self.channel_name)
            await self.send_json({"ack":"auth"}); return
        try:
            model = parse_event(content["kind"], content["payload"])
        except (ValidationError, ValueError, KeyError):
            await self.send_json({"error":"invalid"}); return
        await self._store(content["kind"], model)
        await self.channel_layer.group_send(self.group,
            {"type":"event.push","kind":content["kind"],"payload":content["payload"]})
        await self.send_json({"ack":"stored","seq":model.seq})

    async def disconnect(self, close_code):
        if hasattr(self, "group"):
            await self.channel_layer.group_discard(self.group, self.channel_name)
```
`backend/ingest/routing.py`:
```python
from django.urls import path
from .consumers import IngestConsumer
websocket_urlpatterns = [path("ws/ingest", IngestConsumer.as_asgi())]
```
`backend/housafe/asgi.py`:
```python
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE","housafe.settings")
from django.core.asgi import get_asgi_application
django_asgi = get_asgi_application()
from channels.routing import ProtocolTypeRouter, URLRouter
import ingest.routing
application = ProtocolTypeRouter({
    "http": django_asgi,
    "websocket": URLRouter(ingest.routing.websocket_urlpatterns),
})
```
> `realtime.routing` 在 Task 8 建。先只引 `ingest.routing`，Task 8 再加 `realtime`。

settings 加 `"ingest"`；`apps.py` 用标准 Django app config（name="ingest"）。

- [ ] **Step 4: 运行确认通过并提交**

Run: `cd backend && pytest ingest -q`
Expected: PASS（2 passed）。

```bash
git add -A && git commit -m "feat: ingest ws gateway (device auth, contract validate, persist, fanout)"
```

## Global Constraints (relevant)

- 所有 WS 鉴权失败必须关闭连接并返回明确 close code，不得静默接受。
- 模块解耦契约：ingest 是编排层，允许组合调用下游接口（contracts、devices、events、channel_layer）。
- `ts_recv` 使用 `int(time.time()*1000)` 记录服务器接收时间，不覆盖事件的 `ts` 字段。
