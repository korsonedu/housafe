from rest_framework import viewsets, views, response, status
from django.shortcuts import get_object_or_404
from families.models import Family
from .models import RadarDevice
from .serializers import DeviceSerializer, BindResultSerializer

class DeviceBindView(views.APIView):
    def post(self, request, family_id):
        fam = get_object_or_404(Family, id=family_id, owner=request.user)
        d = RadarDevice.issue(fam, request.data.get("room","room"))
        return response.Response(BindResultSerializer(d).data, status=status.HTTP_201_CREATED)

class DeviceViewSet(viewsets.ModelViewSet):
    serializer_class = DeviceSerializer
    lookup_field = "device_id"
    def get_queryset(self):
        qs = RadarDevice.objects.filter(family__owner=self.request.user)
        fid = self.kwargs.get("family_id")
        return qs.filter(family_id=fid) if fid else qs
