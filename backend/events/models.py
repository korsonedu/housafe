from django.db import models


class RoomState(models.Model):
    """per-room 最新状态，upsert 模式：每个 device_id 只有一行"""
    device_id = models.CharField(max_length=40, unique=True)
    room = models.CharField(max_length=50)
    ts = models.BigIntegerField(help_text="latest decoder output ts, UTC ms")
    ts_recv = models.BigIntegerField()
    posture = models.CharField(max_length=10, null=True, blank=True)
    confidence = models.FloatField(null=True, blank=True)
    presence = models.BooleanField(default=True)
    moving = models.BooleanField(default=False)
    resp_rate = models.FloatField(null=True, blank=True)
    heart_rate = models.FloatField(null=True, blank=True)
    quality = models.FloatField(null=True, blank=True)
    anomaly_score = models.FloatField(default=0.0)
    occupancy_count = models.IntegerField(default=0)

    class Meta:
        indexes = [models.Index(fields=["device_id"])]


class Alert(models.Model):
    """通知事件，append-only"""
    device_id = models.CharField(max_length=40)
    room = models.CharField(max_length=50)
    ts = models.BigIntegerField()
    ts_recv = models.BigIntegerField()
    alert_type = models.CharField(max_length=30)
    severity = models.CharField(max_length=10)
    payload = models.JSONField(default=dict)

    class Meta:
        indexes = [models.Index(fields=["device_id", "ts"])]


class StateSnapshot(models.Model):
    """定时快照，~1/min，用于趋势报告"""
    device_id = models.CharField(max_length=40)
    room = models.CharField(max_length=50)
    ts = models.BigIntegerField()
    state_blob = models.JSONField()

    class Meta:
        indexes = [models.Index(fields=["device_id", "ts"])]
