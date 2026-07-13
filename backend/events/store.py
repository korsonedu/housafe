from django.db import IntegrityError, transaction
from .models import PostureRow, PresenceRow, VitalRow, OccupancyRow

_MAP = {"posture":PostureRow,"presence":PresenceRow,"vital":VitalRow,"occupancy":OccupancyRow}
_FIELDS = {
    "posture": lambda m: {"posture":m.posture,"confidence":m.confidence},
    "presence": lambda m: {"presence":m.presence,"moving":m.moving},
    "vital": lambda m: {"quiet":m.quiet,"resp_rate":m.resp_rate,"heart_rate":m.heart_rate,"quality":m.quality},
    "occupancy": lambda m: {"count":m.count},
}
def store_event(device_id, kind, model, ts_recv):
    Row = _MAP[kind]
    try:
        with transaction.atomic():
            Row.objects.create(ts=model.ts, ts_recv=ts_recv, device_id=device_id,
                               room=model.room, seq=model.seq, **_FIELDS[kind](model))
    except IntegrityError:
        pass  # 幂等：重复 (device_id, seq) 忽略
