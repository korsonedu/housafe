from rest_framework import serializers
from .models import RadarDevice
class DeviceSerializer(serializers.ModelSerializer):
    class Meta:
        model = RadarDevice
        fields = ["device_id","room","online","last_heartbeat","fw_version"]
class BindResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = RadarDevice
        fields = ["device_id","secret","room"]
