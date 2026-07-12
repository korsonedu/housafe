### Task 4: 家庭 / 老人 / 紧急联系人（families）

**Files:**
- Create: `backend/families/__init__.py`, `apps.py`, `models.py`, `serializers.py`, `views.py`, `urls.py`
- Modify: `backend/housafe/settings.py`（INSTALLED_APPS 加 `families`），`backend/housafe/urls.py`
- Test: `backend/families/tests/test_families.py`

**Interfaces:**
- Consumes: `accounts` 的 JWT 用户
- Produces:
  - 模型 `Family(id, name, owner→User)`、`Membership(user, family, role)`、`Elder(id, family, name, note)`、`Contact(id, family, name, phone, order)`
  - REST（均需鉴权，仅本人所属家庭可见）：`GET/POST /api/families`、`GET/PATCH/DELETE /api/families/{id}`、`GET/POST /api/families/{id}/elders`、`GET/POST /api/families/{id}/contacts`（contacts 按 `order` 升序）

- [ ] **Step 1: 写失败测试**

`backend/families/tests/test_families.py`:
```python
import pytest
pytestmark = pytest.mark.django_db

def auth(api):
    api.post("/api/auth/register", {"username":"k","password":"pw12345678"}, format="json")
    t = api.post("/api/auth/login", {"username":"k","password":"pw12345678"}, format="json").data["access"]
    api.credentials(HTTP_AUTHORIZATION=f"Bearer {t}")

def test_create_family_and_list_own_only(api):
    auth(api)
    r = api.post("/api/families", {"name":"我家"}, format="json")
    assert r.status_code == 201
    fid = r.data["id"]
    r = api.get("/api/families")
    assert r.status_code == 200 and len(r.data) == 1
    r = api.post(f"/api/families/{fid}/contacts", {"name":"儿子","phone":"13800000000","order":1}, format="json")
    assert r.status_code == 201

def test_family_requires_auth(api):
    assert api.get("/api/families").status_code == 401
```

- [ ] **Step 2: 运行确认失败**

Run: `cd backend && pytest families -q`
Expected: FAIL（404/路由缺失）。

- [ ] **Step 3: 实现 models**

`backend/families/models.py`:
```python
from django.conf import settings
from django.db import models

class Family(models.Model):
    name = models.CharField(max_length=100)
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="owned_families")

class Membership(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    family = models.ForeignKey(Family, on_delete=models.CASCADE, related_name="members")
    role = models.CharField(max_length=20, default="member")

class Elder(models.Model):
    family = models.ForeignKey(Family, on_delete=models.CASCADE, related_name="elders")
    name = models.CharField(max_length=50)
    note = models.CharField(max_length=200, blank=True)

class Contact(models.Model):
    family = models.ForeignKey(Family, on_delete=models.CASCADE, related_name="contacts")
    name = models.CharField(max_length=50)
    phone = models.CharField(max_length=20)
    order = models.PositiveIntegerField(default=0)
    class Meta:
        ordering = ["order"]
```

- [ ] **Step 4: 实现 serializers / views / urls**

`backend/families/serializers.py`:
```python
from rest_framework import serializers
from .models import Family, Elder, Contact

class FamilySerializer(serializers.ModelSerializer):
    class Meta: model = Family; fields = ["id","name"]
class ElderSerializer(serializers.ModelSerializer):
    class Meta: model = Elder; fields = ["id","name","note"]
class ContactSerializer(serializers.ModelSerializer):
    class Meta: model = Contact; fields = ["id","name","phone","order"]
```
`backend/families/views.py`:
```python
from rest_framework import viewsets
from .models import Family, Elder, Contact
from .serializers import FamilySerializer, ElderSerializer, ContactSerializer

class FamilyViewSet(viewsets.ModelViewSet):
    serializer_class = FamilySerializer
    def get_queryset(self):
        return Family.objects.filter(owner=self.request.user)
    def perform_create(self, serializer):
        serializer.save(owner=self.request.user)

class _ChildViewSet(viewsets.ModelViewSet):
    child_model = None
    def _family(self):
        return Family.objects.get(id=self.kwargs["family_id"], owner=self.request.user)
    def get_queryset(self):
        return self.child_model.objects.filter(family=self._family())
    def perform_create(self, serializer):
        serializer.save(family=self._family())

class ElderViewSet(_ChildViewSet):
    child_model = Elder; serializer_class = ElderSerializer
class ContactViewSet(_ChildViewSet):
    child_model = Contact; serializer_class = ContactSerializer
```
`backend/families/urls.py`:
```python
from django.urls import path
from .views import FamilyViewSet, ElderViewSet, ContactViewSet
fam = FamilyViewSet.as_view({"get":"list","post":"create"})
fam_d = FamilyViewSet.as_view({"get":"retrieve","patch":"partial_update","delete":"destroy"})
elders = ElderViewSet.as_view({"get":"list","post":"create"})
contacts = ContactViewSet.as_view({"get":"list","post":"create"})
urlpatterns = [
    path("families", fam),
    path("families/<int:pk>", fam_d),
    path("families/<int:family_id>/elders", elders),
    path("families/<int:family_id>/contacts", contacts),
]
```
`apps.py` 同 Task 3 模式（name="families"）。settings 加 `"families"`；`housafe/urls.py` 加 `path("api/", include("families.urls"))`。

- [ ] **Step 5: 迁移、测试、提交**

Run: `python manage.py makemigrations families && python manage.py migrate && pytest families -q`
Expected: PASS（2 passed）。
```bash
git add -A && git commit -m "feat: families/elders/contacts with per-owner scoping"
```

## Global Constraints (relevant)

- 严守 PRD「独居假设 + 无身份识别」：数据模型不得出现"识别是谁"的字段。（此任务的数据模型正确：Family/Elder/Contact 均无身份识别字段）
- 每个任务结束必须 commit；提交信息用 `feat:` / `test:` / `chore:` 前缀。
- 模块解耦契约：families 只允许依赖 accounts（仅 `request.user`），对外暴露模型 + REST。禁止跨 app 查对方的表。
