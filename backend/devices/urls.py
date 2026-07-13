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
