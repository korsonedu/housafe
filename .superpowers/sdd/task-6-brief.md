### Task 6: 时序事件存储与查询（events）

**Files:**
- Create: `backend/events/__init__.py`, `apps.py`, `models.py`, `serializers.py`, `views.py`, `urls.py`, `store.py`
- Create: `backend/events/migrations/0002_hypertables.py`（RunSQL 转 hypertable）
- Modify: settings（加 `events`），`housafe/urls.py`
- Test: `backend/events/tests/test_store.py`, `backend/events/tests/test_query.py`

**Interfaces:**
- Consumes: 契约模型（Task 2）、`RadarDevice`
- Produces:
  - 表 `PostureRow`、`PresenceRow`、`VitalRow`、`OccupancyRow`（列含 `ts:bigint`, `ts_recv`, `device_id`, `room`, `seq` + 各自字段）
  - `store_event(device_id:str, kind:str, model:BaseModel, ts_recv:int) -> None`（幂等：`(device_id, kind, seq)` 冲突则忽略）
  - REST：`GET /api/families/{id}/events?kind=&room=&since=&until=`（按 `ts` 升序）；`GET /api/families/{id}/today`（每房间最新 presence/posture/vital）

- [ ] **Step 1: 写失败测试（存储幂等 + 以 ts 为轴）**

`backend/events/tests/test_store.py`:
```python
import pytest
pytestmark = pytest.mark.django_db
from events.store import store_event
from events.models import PostureRow
from housafe_contracts.events import parse_event

def test_store_and_idempotent():
    m = parse_event("posture", {"ts":1700000000000,"radar_id":"d1","room":"bed","seq":1,"posture":"walk","confidence":0.9})
    store_event("d1","posture",m,ts_recv=1700000000050)
    store_event("d1","posture",m,ts_recv=1700000000060)  # 重复 seq
    rows = PostureRow.objects.filter(device_id="d1")
    assert rows.count() == 1 and rows.first().ts == 1700000000000
```

- [ ] **Step 2: 运行确认失败**

Run: `cd backend && pytest events/tests/test_store.py -q`
Expected: FAIL。

- [ ] **Step 3: 实现 models + store**

`backend/events/models.py`:
```python
from django.db import models

class _Row(models.Model):
    ts = models.BigIntegerField()
    ts_recv = models.BigIntegerField()
    device_id = models.CharField(max_length=40)
    room = models.CharField(max_length=50)
    seq = models.BigIntegerField()
    class Meta:
        abstract = True
        constraints = []
        indexes = [models.Index(fields=["device_id","ts"])]

class PostureRow(_Row):
    posture = models.CharField(max_length=10); confidence = models.FloatField()
    class Meta(_Row.Meta):
        constraints = [models.UniqueConstraint(fields=["device_id","seq"], name="uq_posture_seq")]
class PresenceRow(_Row):
    presence = models.BooleanField(); moving = models.BooleanField()
    class Meta(_Row.Meta):
        constraints = [models.UniqueConstraint(fields=["device_id","seq"], name="uq_presence_seq")]
class VitalRow(_Row):
    quiet = models.BooleanField(); resp_rate = models.FloatField(null=True)
    heart_rate = models.FloatField(null=True); quality = models.FloatField()
    class Meta(_Row.Meta):
        constraints = [models.UniqueConstraint(fields=["device_id","seq"], name="uq_vital_seq")]
class OccupancyRow(_Row):
    count = models.IntegerField()
    class Meta(_Row.Meta):
        constraints = [models.UniqueConstraint(fields=["device_id","seq"], name="uq_occ_seq")]
```
`backend/events/store.py`:
```python
from django.db import IntegrityError
from .models import PostureRow, PresenceRow, VitalRow, OccupancyRow

_MAP = {"posture":PostureRow,"presence":PresenceRow,"vital":VitalRow,"occupancy":OccupancyRow}
_FIELDS = {
    "posture": lambda m: {"posture":m.posture,"confidence":m.confidence},
    "presence": lambda m: {"presence":m.presence,"moving":m.moving},
    "vital": lambda m: {"quiet":m.quiet,"resp_rate":m.resp_rate,"heart_rate":m.heart_rate,"quality":m.quality},
    "occupancy": lambda m: {"count":m.count},
}
def store_event(device_id, kind, model, ts_recv):
    Row = _MAP[kind]
    try:
        Row.objects.create(ts=model.ts, ts_recv=ts_recv, device_id=device_id,
                           room=model.room, seq=model.seq, **_FIELDS[kind](model))
    except IntegrityError:
        pass  # 幂等：重复 (device_id, seq) 忽略
```

- [ ] **Step 4: 运行确认通过**

Run: `python manage.py makemigrations events && python manage.py migrate && pytest events/tests/test_store.py -q`
Expected: PASS。

- [ ] **Step 5: Timescale hypertable 迁移**

`backend/events/migrations/0002_hypertables.py`:
```python
from django.db import migrations
SQL = """
SELECT create_hypertable('events_posturerow','ts',chunk_time_interval=>86400000,if_not_exists=>TRUE,migrate_data=>TRUE);
SELECT create_hypertable('events_presencerow','ts',chunk_time_interval=>86400000,if_not_exists=>TRUE,migrate_data=>TRUE);
SELECT create_hypertable('events_vitalrow','ts',chunk_time_interval=>86400000,if_not_exists=>TRUE,migrate_data=>TRUE);
SELECT create_hypertable('events_occupancyrow','ts',chunk_time_interval=>86400000,if_not_exists=>TRUE,migrate_data=>TRUE);
"""
class Migration(migrations.Migration):
    dependencies = [("events","0001_initial")]
    operations = [migrations.RunSQL(SQL, reverse_sql=migrations.RunSQL.noop)]
```
> hypertable 要求时间列在主键内。若 `create_hypertable` 报唯一约束错误，将各表唯一约束改为 `("device_id","seq","ts")` 并重生成 0001 迁移。
>
> **如果用的是 SQLite（无 Timescale），跳过 Step 5** — 0002 迁移仅在 TimescaleDB 上有效。

Run: `python manage.py migrate`
Expected: 无错（hypertable 建成）或在 SQLite 上静默跳过（因 create_hypertable 不存在）。

- [ ] **Step 6: 查询 API（today + events）**

`backend/events/tests/test_query.py`:
```python
import pytest
pytestmark = pytest.mark.django_db
from housafe_contracts.events import parse_event
from events.store import store_event

def _setup(api):
    api.post("/api/auth/register", {"username":"k","password":"pw12345678"}, format="json")
    t = api.post("/api/auth/login", {"username":"k","password":"pw12345678"}, format="json").data["access"]
    api.credentials(HTTP_AUTHORIZATION=f"Bearer {t}")
    fid = api.post("/api/families", {"name":"f"}, format="json").data["id"]
    did = api.post(f"/api/families/{fid}/devices/bind", {"room":"bed"}, format="json").data["device_id"]
    return fid, did

def test_today_returns_latest_per_room(api):
    fid, did = _setup(api)
    for seq,p in [(1,"stand"),(2,"walk")]:
        store_event(did,"posture",parse_event("posture",
            {"ts":1700000000000+seq,"radar_id":did,"room":"bed","seq":seq,"posture":p,"confidence":0.9}),ts_recv=0)
    r = api.get(f"/api/families/{fid}/today")
    assert r.status_code == 200 and r.data["bed"]["posture"]["posture"] == "walk"
```
`backend/events/views.py`:
```python
from rest_framework import views, response
from families.models import Family
from .models import PostureRow, PresenceRow, VitalRow

class TodayView(views.APIView):
    def get(self, request, family_id):
        fam = Family.objects.get(id=family_id, owner=request.user)
        dids = list(fam.devices.values_list("device_id", flat=True))
        out = {}
        for did_room in fam.devices.values("device_id","room"):
            room = did_room["room"]; did = did_room["device_id"]
            out.setdefault(room, {})
            for key, Row, fields in [
                ("posture",PostureRow,["posture","confidence"]),
                ("presence",PresenceRow,["presence","moving"]),
                ("vital",VitalRow,["quiet","resp_rate","heart_rate","quality"])]:
                row = Row.objects.filter(device_id=did).order_by("-ts").first()
                if row: out[room][key] = {"ts":row.ts, **{f:getattr(row,f) for f in fields}}
        return response.Response(out)
```
`backend/events/urls.py`:
```python
from django.urls import path
from .views import TodayView
urlpatterns = [path("families/<int:family_id>/today", TodayView.as_view())]
```
settings 加 `"events"`；`housafe/urls.py` 加 `path("api/", include("events.urls"))`。

- [ ] **Step 7: 测试并提交**

Run: `cd backend && pytest events -q`
Expected: PASS。
```bash
git add -A && git commit -m "feat: timescale event storage + today/history query"
```

## Global Constraints (relevant)

- 事件时间轴一律用事件自带的 `ts`（UTC 毫秒 int），服务器**绝不**用"当前时间"覆盖；服务器接收时间单独存 `ts_recv`。
- 模块解耦契约：events 依赖 contracts、devices（仅 device_id 字符串），不得跨 app import 对方内部实现。
- `store_event(device_id, kind, model, ts_recv)` 签名固定，幂等原则不可破。
