### Task 8: 实时推送 WS（realtime）

**Files:**
- Create: `backend/realtime/__init__.py`, `apps.py`, `consumers.py`, `routing.py`
- Modify: `backend/housafe/asgi.py`（加 realtime 路由），settings（加 `realtime`）
- Test: `backend/realtime/tests/test_realtime.py`

**Interfaces:**
- Consumes: ingest 的 `group_send("family_<id>", {type:"event.push",...})`；JWT
- Produces: WS 端点 `ws/app?token=<jwt>&family=<id>`：JWT 鉴权 + 校验用户属该家庭 → 加入 `family_<id>` 组 → 收到 `event.push` 转发给 App

- [ ] **Step 1: 写失败测试（端到端：ingest 灌 → app 端收到）**

`backend/realtime/tests/test_realtime.py`:
```python
import pytest
from channels.testing import WebsocketCommunicator
from housafe.asgi import application
pytestmark = pytest.mark.django_db(transaction=True)

@pytest.mark.asyncio
async def test_ingest_to_app_flow(db):
    from django.contrib.auth.models import User
    from families.models import Family
    from devices.models import RadarDevice
    from rest_framework_simplejwt.tokens import AccessToken
    u = await User.objects.acreate_user("k","","pw12345678")
    fam = await Family.objects.acreate(name="f", owner=u)
    await RadarDevice.objects.acreate(device_id="d1", secret="s1", family=fam, room="bed")
    token = str(AccessToken.for_user(u))

    app = WebsocketCommunicator(application, f"/ws/app?token={token}&family={fam.id}")
    assert (await app.connect())[0]
    ing = WebsocketCommunicator(application, "/ws/ingest")
    await ing.connect()
    await ing.send_json_to({"device_id":"d1","secret":"s1"}); await ing.receive_json_from()
    await ing.send_json_to({"kind":"posture","payload":{"ts":1,"radar_id":"d1","room":"bed","seq":1,"posture":"walk","confidence":0.9}})
    pushed = await app.receive_json_from()
    assert pushed["kind"] == "posture" and pushed["payload"]["room"] == "bed"
    await app.disconnect(); await ing.disconnect()
```

- [ ] **Step 2: 运行确认失败**

Run: `cd backend && pytest realtime -q`
Expected: FAIL。

- [ ] **Step 3: 实现 realtime consumer + routing**

`backend/realtime/consumers.py`:
```python
from urllib.parse import parse_qs
from channels.generic.websocket import AsyncJsonWebsocketConsumer
from channels.db import database_sync_to_async

class AppConsumer(AsyncJsonWebsocketConsumer):
    async def connect(self):
        qs = parse_qs(self.scope["query_string"].decode())
        token = qs.get("token",[None])[0]; family_id = qs.get("family",[None])[0]
        user = await self._auth(token)
        if user is None or not await self._owns(user, family_id):
            await self.close(code=4401); return
        self.group = f"family_{family_id}"
        await self.channel_layer.group_add(self.group, self.channel_name)
        await self.accept()

    @database_sync_to_async
    def _auth(self, token):
        if not token: return None
        try:
            from rest_framework_simplejwt.tokens import AccessToken
            from django.contrib.auth.models import User
            return User.objects.get(id=AccessToken(token)["user_id"])
        except Exception:
            return None
    @database_sync_to_async
    def _owns(self, user, family_id):
        from families.models import Family
        return Family.objects.filter(id=family_id, owner=user).exists()

    async def event_push(self, event):
        await self.send_json({"kind":event["kind"],"payload":event["payload"]})

    async def disconnect(self, code):
        if hasattr(self,"group"):
            await self.channel_layer.group_discard(self.group, self.channel_name)
```
`backend/realtime/routing.py`:
```python
from django.urls import path
from .consumers import AppConsumer
websocket_urlpatterns = [path("ws/app", AppConsumer.as_asgi())]
```
修改 `housafe/asgi.py` — 把 URLRouter 改成同时包含 ingest 和 realtime 路由：
```python
application = ProtocolTypeRouter({
    "http": django_asgi,
    "websocket": URLRouter(ingest.routing.websocket_urlpatterns + realtime.routing.websocket_urlpatterns),
})
```
settings 加 `"realtime"`；`apps.py` 用标准配置（name="realtime"）。

- [ ] **Step 4: 运行确认通过并提交**

Run: `cd backend && pytest realtime -q`
Expected: PASS。
```bash
git add -A && git commit -m "feat: realtime app ws (jwt auth, family scoping, event fanout)"
```

## Global Constraints (relevant)

- 所有 WS 鉴权失败必须关闭连接并返回明确 close code（4401），不得静默接受。
- 模块解耦契约：realtime 是编排层，允许组合调用（accounts JWT、families 所有权校验、channel_layer）。
- realtime 不得 import ingest 内部实现，只通过 channel_layer 的 group 消息通信。
