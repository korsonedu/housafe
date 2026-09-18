import time
from housafe_contracts.events import (
    DecoderPosture,
    DecoderVital,
    DecoderAlert,
    DecoderOccupancy,
    DecoderAnomaly,
    DecoderOutput,
)
from .models import RoomState, Alert


def store_decoder_output(device_id: str, room: str, output: DecoderOutput, ts_recv: int | None = None):
    """接收 Decoder 输出，更新 RoomState / 创建 Alert"""
    if ts_recv is None:
        ts_recv = int(time.time() * 1000)

    if isinstance(output, DecoderPosture):
        RoomState.objects.update_or_create(
            device_id=device_id,
            defaults={
                "room": room,
                "ts": output.ts,
                "ts_recv": ts_recv,
                "posture": output.posture,
                "confidence": output.confidence,
                "presence": output.presence,
                "moving": output.moving,
            },
        )
    elif isinstance(output, DecoderVital):
        RoomState.objects.update_or_create(
            device_id=device_id,
            defaults={
                "room": room,
                "ts": output.ts,
                "ts_recv": ts_recv,
                "resp_rate": output.resp_rate,
                "heart_rate": output.heart_rate,
                "quality": output.quality,
            },
        )
    elif isinstance(output, DecoderAlert):
        Alert.objects.create(
            device_id=device_id,
            room=room,
            ts=output.ts,
            ts_recv=ts_recv,
            alert_type=output.alert_type,
            severity=output.severity,
            payload=output.payload,
        )
    elif isinstance(output, DecoderOccupancy):
        RoomState.objects.update_or_create(
            device_id=device_id,
            defaults={
                "room": room,
                "ts": output.ts,
                "ts_recv": ts_recv,
                "occupancy_count": output.count,
            },
        )
    elif isinstance(output, DecoderAnomaly):
        RoomState.objects.update_or_create(
            device_id=device_id,
            defaults={
                "room": room,
                "ts": output.ts,
                "ts_recv": ts_recv,
                "anomaly_score": output.anomaly_score,
            },
        )
