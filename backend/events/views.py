from django.shortcuts import get_object_or_404
from rest_framework import views, response
from families.models import Family
from .models import PostureRow, VitalRow

class TodayView(views.APIView):
    def get(self, request, family_id):
        fam = get_object_or_404(Family, id=family_id, owner=request.user)
        rooms = []
        for did_room in fam.devices.values("device_id", "room", "online"):
            room = did_room["room"]
            did = did_room["device_id"]

            posture_label = None
            posture_ts = None
            row = PostureRow.objects.filter(device_id=did).order_by("-ts").first()
            if row:
                posture_label = row.posture
                posture_ts = row.ts

            heart_rate = None
            breath_rate = None
            vrow = VitalRow.objects.filter(device_id=did).order_by("-ts").first()
            if vrow:
                heart_rate = vrow.heart_rate
                breath_rate = vrow.resp_rate

            rooms.append({
                "room_name": room,
                "device_online": did_room["online"],
                "posture": posture_label,
                "posture_ts": posture_ts,
                "heart_rate": heart_rate,
                "breath_rate": breath_rate,
            })
        return response.Response({"rooms": rooms})
