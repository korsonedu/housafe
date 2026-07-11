from django.contrib.auth.models import User
from rest_framework import serializers

class RegisterSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ["username","password"]
        extra_kwargs = {"password":{"write_only":True,"min_length":8}}
    def create(self, data):
        return User.objects.create_user(**data)
