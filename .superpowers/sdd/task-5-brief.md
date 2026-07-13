### Task 5: 雷达设备管理（devices）

**Files:**
- Create: `backend/devices/__init__.py`, `apps.py`, `models.py`, `serializers.py`, `views.py`, `urls.py`
- Modify: settings（加 `devices`），`housafe/urls.py`
- Test: `backend/devices/tests/test_devices.py`

**Interfaces:**
- Consumes: `families.Family`（设备归属家庭）；JWT 用户
- Produces:
  - 模型 `RadarDevice(device_id:str uniq, secret:str, family, room, online:bool, last_heartbeat, fw_version)`
  - `RadarDevice.verify(device_id, secret) -> RadarDevice | None`（classmethod，供 Task 7 ingest 鉴权用）
  - REST：`POST /api/families/{id}/devices/bind`（生成 device_id+secret 返回一次）、`GET /api/families/{id}/devices`、`PATCH /api/devices/{device_id}`（重命名 room）、`GET /api/devices/{device_id}/status`

- [ ] **Step 1: 写失败测试**

`backend/devices/tests/test_devices.py`:
```python
import pytest
pytestmark = pytest.mark.django_db

def auth_and_family(api):
    api.post("/api/auth/register", {"username":"k","password":"pw12345678"}, format="json")
    t = api.post("/api/auth/login", {"username":"k","password":"pw12345678"}, format="json").data["access"]
    api.credentials(HTTP_AUTHORIZATION=f"Bearer {t}")
    return api.post("/api/families", {"name":"我家"}, format="json").data["id"]

def test_bind_returns_credentials_and_lists(api):
    fid = auth_and_family(api)
    r = api.post(f"/api/families/{fid}/devices/bind", {"room":"bedroom"}, format="json")
    assert r.status_code == 201 and r.data["device_id"] and r.data["secret"]
    r = api.get(f"/api/families/{fid}/devices")
    assert len(r.data) == 1 and r.data[0]["room"] == "bedroom"

def test_verify_credentials():
    from devices.models import RadarDevice
    from families.models import Family
    from django.contrib.auth.models import User
    u = User.objects.create_user("k","","pw12345678")
    fam = Family.objects.create(name="f", owner=u)
    d = RadarDevice.objects.create(device_id="d1", secret="s1", family=fam, room="bath")
    assert RadarDevice.verify("d1","s1") == d
    assert RadarDevice.verify("d1","bad") is None
```

- [ ] **Step 2: 运行确认失败**

Run: `cd backend && pytest devices -q`
Expected: FAIL。

- [ ] **Step 3: 实现 models**

`backend/devices/models.py`:
```python
import secrets
from django.db import models
from families.models import Family

class RadarDevice(models.Model):
    device_id = models.CharField(max_length=40, unique=True)
    secret = models.CharField(max_length=64)
    family = models.ForeignKey(Family, on_delete=models.CASCADE, related_name="devices")
    room = models.CharField(max_length=50)
    online = models.BooleanField(default=False)
    last_heartbeat = models.DateTimeField(null=True, blank=True)
    fw_version = models.CharField(max_length=20, default="sim-0.1")

    @classmethod
    def issue(cls, family, room):
        return cls.objects.create(
            device_id="rad_"+secrets.token_hex(6),
            secret=secrets.token_hex(16), family=family, room=room)

    @classmethod
    def verify(cls, device_id, secret):
        return cls.objects.filter(device_id=device_id, secret=secret).first()
```

- [ ] **Step 4: serializers / views / urls**

`backend/devices/serializers.py`:
```python
from rest_framework import serializers
from .models import RadarDevice
class DeviceSerializer(serializers.ModelSerializer):
    class Meta:
        model = RadarDevice
        fields = ["device_id","room","online","last_heartbeat","fw_version"]
class BindResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = RadarDevice
        fields = ["device_id","secret","room"]
```
`backend/devices/views.py`:
```python
from rest_framework import viewsets, views, response, status
from families.models import Family
from .models import RadarDevice
from .serializers import DeviceSerializer, BindResultSerializer

class DeviceBindView(views.APIView):
    def post(self, request, family_id):
        fam = Family.objects.get(id=family_id, owner=request.user)
        d = RadarDevice.issue(fam, request.data.get("room","room"))
        return response.Response(BindResultSerializer(d).data, status=status.HTTP_201_CREATED)

class DeviceViewSet(viewsets.ModelViewSet):
    serializer_class = DeviceSerializer
    lookup_field = "device_id"
    def get_queryset(self):
        qs = RadarDevice.objects.filter(family__owner=self.request.user)
        fid = self.kwargs.get("family_id")
        return qs.filter(family_id=fid) if fid else qs
```
`backend/devices/urls.py`:
```python
from django.urls import path
from .views import DeviceBindView, DeviceViewSet
dev_list = DeviceViewSet.as_view({"get":"list"})
dev_detail = DeviceViewSet.as_view({"patch":"partial_update","get":"retrieve"})
urlpatterns = [
    path("families/<int:family_id>/devices/bind", DeviceBindView.as_view()),
    path("families/<int:family_id>/devices", dev_list),
    path("devices/<str:device_id>", dev_detail),
    path("devices/<str:device_id>/status", dev_detail),
]
```
settings 加 `"devices"`；`housafe/urls.py` 加 `path("api/", include("devices.urls"))`。

- [ ] **Step 5: 迁移、测试、提交**

Run: `python manage.py makemigrations devices && python manage.py migrate && pytest devices -q`
Expected: PASS（2 passed）。
```bash
git add -A && git commit -m "feat: radar device bind/list/rename + credential verify"
```

## Global Constraints (relevant)

- 模块解耦契约：devices 只允许依赖 families（仅 FK），对外暴露 `RadarDevice.verify()` / REST。不得跨 app 查对方的表。
- `RadarDevice.verify(device_id, secret)` classmethod 供 ingest（Task 7）鉴权用，签名必须一致。
- 每个任务结束必须 commit；提交信息用 `feat:`。
