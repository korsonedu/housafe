### Task 9: 最小仿真灌数脚本 + 端到端冒烟

**Files:**
- Create: `simulator/__init__.py`, `simulator/feed.py`, `simulator/README.md`
- Test: `backend/tests/test_e2e_smoke.py`

**Interfaces:**
- Consumes: `ws/ingest`（Task 7）、契约（Task 2）
- Produces: `feed.py`：给定 `device_id/secret/ws_url`，按脚本发一串事件（presence→posture 序列 + vital），用于手动演示；`test_e2e_smoke.py` 断言全链路。

- [ ] **Step 1: 写端到端冒烟测试**

`backend/tests/test_e2e_smoke.py`:
```python
import pytest
from channels.testing import WebsocketCommunicator
from housafe.asgi import application
pytestmark = pytest.mark.django_db(transaction=True)

@pytest.mark.asyncio
async def test_full_chain_device_to_query(db):
    from django.contrib.auth.models import User
    from families.models import Family
    from devices.models import RadarDevice
    u = await User.objects.acreate_user("k","","pw12345678")
    fam = await Family.objects.acreate(name="f", owner=u)
    await RadarDevice.objects.acreate(device_id="d1", secret="s1", family=fam, room="bed")
    ing = WebsocketCommunicator(application, "/ws/ingest")
    await ing.connect()
    await ing.send_json_to({"device_id":"d1","secret":"s1"}); await ing.receive_json_from()
    for seq,p in [(1,"stand"),(2,"walk"),(3,"lie")]:
        await ing.send_json_to({"kind":"posture","payload":{"ts":1700000000000+seq,"radar_id":"d1","room":"bed","seq":seq,"posture":p,"confidence":0.9}})
        await ing.receive_json_from()
    from events.models import PostureRow
    assert await PostureRow.objects.filter(device_id="d1").acount() == 3
    await ing.disconnect()
```

- [ ] **Step 2: 运行确认通过**

Run: `cd backend && pytest tests/test_e2e_smoke.py -q`
Expected: PASS。

- [ ] **Step 3: 写演示用 feed 脚本**

`simulator/feed.py`:
```python
"""最小灌数脚本：python feed.py <device_id> <secret> [ws_url]"""
import asyncio, sys, time, json, websockets

async def main(device_id, secret, url="ws://localhost:8000/ws/ingest"):
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"device_id":device_id,"secret":secret}))
        print(await ws.recv())
        seq = 0
        script = [("presence",{"presence":True,"moving":False}),
                  ("posture",{"posture":"stand","confidence":0.95}),
                  ("posture",{"posture":"walk","confidence":0.9}),
                  ("vital",{"quiet":True,"resp_rate":16.0,"heart_rate":72.0,"quality":0.8}),
                  ("posture",{"posture":"lie","confidence":0.92})]
        for kind, fields in script:
            seq += 1
            payload = {"ts":int(time.time()*1000),"radar_id":device_id,"room":"bedroom","seq":seq, **fields}
            await ws.send(json.dumps({"kind":kind,"payload":payload}))
            print(await ws.recv()); await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))
```
`simulator/README.md`：说明先 `bind` 拿 device_id/secret，再 `pip install websockets && python feed.py <id> <secret>`，配合 App 观察实时刷新。

- [ ] **Step 4: 提交**

```bash
git add -A && git commit -m "feat: minimal simulator feed script + e2e smoke test"
```

## Global Constraints (relevant)

- 事件契约唯一来源是 `contracts/` 包，simulator 不得复制字段定义。
- `ts` 由脚本生成（`int(time.time()*1000)` 毫秒 UTC），符合事件自带时间轴原则。
