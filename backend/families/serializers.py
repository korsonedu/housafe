from rest_framework import serializers
from .models import Family, Elder, Contact

class FamilySerializer(serializers.ModelSerializer):
    class Meta:
        model = Family
        fields = ["id", "name"]

class ElderSerializer(serializers.ModelSerializer):
    class Meta:
        model = Elder
        fields = ["id", "name", "note"]

class ContactSerializer(serializers.ModelSerializer):
    class Meta:
        model = Contact
        fields = ["id", "name", "phone", "order"]
