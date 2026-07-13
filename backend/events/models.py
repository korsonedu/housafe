from django.db import models

class _Row(models.Model):
    ts = models.BigIntegerField()
    ts_recv = models.BigIntegerField()
    device_id = models.CharField(max_length=40)
    room = models.CharField(max_length=50)
    seq = models.BigIntegerField()
    class Meta:
        abstract = True
        constraints = []
        indexes = [models.Index(fields=["device_id","ts"])]

class PostureRow(_Row):
    posture = models.CharField(max_length=10); confidence = models.FloatField()
    class Meta(_Row.Meta):
        constraints = [models.UniqueConstraint(fields=["device_id","seq"], name="uq_posture_seq")]
class PresenceRow(_Row):
    presence = models.BooleanField(); moving = models.BooleanField()
    class Meta(_Row.Meta):
        constraints = [models.UniqueConstraint(fields=["device_id","seq"], name="uq_presence_seq")]
class VitalRow(_Row):
    quiet = models.BooleanField(); resp_rate = models.FloatField(null=True)
    heart_rate = models.FloatField(null=True); quality = models.FloatField()
    class Meta(_Row.Meta):
        constraints = [models.UniqueConstraint(fields=["device_id","seq"], name="uq_vital_seq")]
class OccupancyRow(_Row):
    count = models.IntegerField()
    class Meta(_Row.Meta):
        constraints = [models.UniqueConstraint(fields=["device_id","seq"], name="uq_occ_seq")]
