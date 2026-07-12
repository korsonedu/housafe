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
    child_model = Elder
    serializer_class = ElderSerializer

class ContactViewSet(_ChildViewSet):
    child_model = Contact
    serializer_class = ContactSerializer
