from django.conf import settings
from django.db import models

class Family(models.Model):
    name = models.CharField(max_length=100)
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="owned_families")

class Membership(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    family = models.ForeignKey(Family, on_delete=models.CASCADE, related_name="members")
    role = models.CharField(max_length=20, default="member")

class Elder(models.Model):
    family = models.ForeignKey(Family, on_delete=models.CASCADE, related_name="elders")
    name = models.CharField(max_length=50)
    note = models.CharField(max_length=200, blank=True)

class Contact(models.Model):
    family = models.ForeignKey(Family, on_delete=models.CASCADE, related_name="contacts")
    name = models.CharField(max_length=50)
    phone = models.CharField(max_length=20)
    order = models.PositiveIntegerField(default=0)
    class Meta:
        ordering = ["order"]
