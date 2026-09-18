import os

from django.shortcuts import get_object_or_404
from rest_framework import views, response, status, permissions
from families.models import Family
from .models import RoomState
from .store import store_decoder_output


class TodayView(views.APIView):
    def get(self, request, family_id):
        fam = get_object_or_404(Family, id=family_id, owner=request.user)
        rooms = []
        for d in fam.devices.values("device_id", "room", "online"):
            did = d["device_id"]
            room_data = {
                "room_name": d["room"],
                "device_online": d["online"],
                "posture": None,
                "posture_ts": None,
                "heart_rate": None,
                "breath_rate": None,
            }
            try:
                rs = RoomState.objects.get(device_id=did)
                room_data["posture"] = rs.posture
                room_data["posture_ts"] = rs.ts
                room_data["heart_rate"] = rs.heart_rate
                room_data["breath_rate"] = rs.resp_rate
            except RoomState.DoesNotExist:
                pass
            rooms.append(room_data)
        return response.Response({"rooms": rooms})


INTERNAL_SERVICE_TOKEN = os.environ.get("INTERNAL_SERVICE_TOKEN", "dev-internal-token")


class AIBridgeView(views.APIView):
    """AI 占位符回调：POST /api/internal/decoder-output"""
    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    def post(self, request):
        token = request.data.get("token")
        if token != INTERNAL_SERVICE_TOKEN:
            return response.Response({"error": "unauthorized"}, status=status.HTTP_401_UNAUTHORIZED)

        outputs = request.data.get("outputs", [])
        if not outputs:
            return response.Response({"error": "empty outputs"}, status=status.HTTP_400_BAD_REQUEST)

        from housafe_contracts.events import DecoderPosture, DecoderVital, DecoderAlert, DecoderOccupancy, DecoderAnomaly
        from channels.layers import get_channel_layer
        from asgiref.sync import async_to_sync

        _MAP = {
            "posture": DecoderPosture,
            "vital": DecoderVital,
            "alert": DecoderAlert,
            "occupancy": DecoderOccupancy,
            "anomaly": DecoderAnomaly,
        }

        channel = get_channel_layer()
        stored = 0
        for item in outputs:
            kind = item.get("kind")
            payload = item.get("payload", {})
            family_id = item.get("family_id")
            model_cls = _MAP.get(kind)
            if model_cls is None:
                continue
            output = model_cls(**payload)
            store_decoder_output(output.device_id, output.room, output)

            # 推送 room.state 到对应 family group
            if family_id:
                try:
                    rs = RoomState.objects.get(device_id=output.device_id)
                    room_payload = {
                        "type": "room.state",
                        "room_name": rs.room,
                        "device_online": True,
                        "posture": rs.posture,
                        "posture_ts": rs.ts,
                        "heart_rate": rs.heart_rate,
                        "breath_rate": rs.resp_rate,
                        "anomaly_score": rs.anomaly_score,
                    }
                    async_to_sync(channel.group_send)(
                        f"family_{family_id}",
                        {"type": "room.state", "payload": room_payload},
                    )
                except RoomState.DoesNotExist:
                    pass
            stored += 1

        return response.Response({"ack": "stored", "count": stored})
