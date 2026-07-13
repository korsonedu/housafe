from rest_framework import views, response
from families.models import Family
from .models import PostureRow, PresenceRow, VitalRow

class TodayView(views.APIView):
    def get(self, request, family_id):
        fam = Family.objects.get(id=family_id, owner=request.user)
        dids = list(fam.devices.values_list("device_id", flat=True))
        out = {}
        for did_room in fam.devices.values("device_id","room"):
            room = did_room["room"]; did = did_room["device_id"]
            out.setdefault(room, {})
            for key, Row, fields in [
                ("posture",PostureRow,["posture","confidence"]),
                ("presence",PresenceRow,["presence","moving"]),
                ("vital",VitalRow,["quiet","resp_rate","heart_rate","quality"])]:
                row = Row.objects.filter(device_id=did).order_by("-ts").first()
                if row: out[room][key] = {"ts":row.ts, **{f:getattr(row,f) for f in fields}}
        return response.Response(out)
