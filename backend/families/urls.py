from django.urls import path
from .views import FamilyViewSet, ElderViewSet, ContactViewSet

fam = FamilyViewSet.as_view({"get": "list", "post": "create"})
fam_d = FamilyViewSet.as_view({"get": "retrieve", "patch": "partial_update", "delete": "destroy"})
elders = ElderViewSet.as_view({"get": "list", "post": "create"})
contacts = ContactViewSet.as_view({"get": "list", "post": "create"})

urlpatterns = [
    path("families", fam),
    path("families/<int:pk>", fam_d),
    path("families/<int:family_id>/elders", elders),
    path("families/<int:family_id>/contacts", contacts),
]
