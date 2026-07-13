import secrets
from django.db import models
from families.models import Family

class RadarDevice(models.Model):
    device_id = models.CharField(max_length=40, unique=True)
    secret = models.CharField(max_length=64)
    family = models.ForeignKey(Family, on_delete=models.CASCADE, related_name="devices")
    room = models.CharField(max_length=50)
    online = models.BooleanField(default=False)
    last_heartbeat = models.DateTimeField(null=True, blank=True)
    fw_version = models.CharField(max_length=20, default="sim-0.1")

    @classmethod
    def issue(cls, family, room):
        return cls.objects.create(
            device_id="rad_"+secrets.token_hex(6),
            secret=secrets.token_hex(16), family=family, room=room)

    @classmethod
    def verify(cls, device_id, secret):
        return cls.objects.filter(device_id=device_id, secret=secret).first()
