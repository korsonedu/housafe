from django.urls import path
from .views import TodayView
urlpatterns = [path("families/<int:family_id>/today", TodayView.as_view())]
