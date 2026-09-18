from django.urls import path
from .views import TodayView, AIBridgeView

urlpatterns = [
    path("families/<int:family_id>/today", TodayView.as_view()),
    path("internal/decoder-output", AIBridgeView.as_view()),
]
