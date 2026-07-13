# Task 5: 雷达设备管理（devices） - 报告

## TDD Evidence

1. **写失败测试** - 创建 `devices/tests/test_devices.py`，包含 `test_bind_returns_credentials_and_lists` 和 `test_verify_credentials`
2. **运行确认失败** - `pytest devices -q` → ModuleNotFoundError: No module named 'devices.models' (1 failed)
3. **实现 models** - `devices/models.py` 完成，含 `RadarDevice` 模型、`issue()` 和 `verify()` classmethod
4. **实现 serializers/views/urls** - 完成 DeviceSerializer, BindResultSerializer, DeviceBindView, DeviceViewSet
5. **迁移、测试、提交** - 迁移成功，`pytest devices -q` → 2 passed，已 commit

## Files Changed

### Created (8 files)
- `backend/devices/__init__.py` - 空文件
- `backend/devices/apps.py` - AppConfig
- `backend/devices/models.py` - RadarDevice 模型
- `backend/devices/serializers.py` - DeviceSerializer, BindResultSerializer
- `backend/devices/views.py` - DeviceBindView, DeviceViewSet
- `backend/devices/urls.py` - 4 个 URL 路由
- `backend/devices/tests/__init__.py` - 空文件
- `backend/devices/tests/test_devices.py` - 2 个测试用例

### Modified (2 files)
- `backend/housafe/settings.py` - INSTALLED_APPS 添加 "devices"
- `backend/housafe/urls.py` - 添加 `path("api/", include("devices.urls"))`

## Coupling Compliance Check

- ✅ `from families.models import Family` — ONLY for FK (models.py, views.py)
- ❌ 未从 `families.views`, `families.serializers`, `families.urls` 导入
- ❌ 未直接查询 families 表

## Self-Review Findings

- `RadarDevice.verify(device_id, secret)` 签名正确，返回 `RadarDevice | None`
- `DeviceViewSet.lookup_field = "device_id"` 正确设置
- `device_id` 格式: `rad_` + `secrets.token_hex(6)` = 16 字符
- `secret`: `secrets.token_hex(16)` = 32 字符（max_length=64 范围内）
- URL 路由与 brief 完全一致
  - `POST /api/families/{id}/devices/bind`
  - `GET /api/families/{id}/devices`
  - `PATCH /api/devices/{device_id}`
  - `GET /api/devices/{device_id}/status`
