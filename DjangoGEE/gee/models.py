from django.db import models

class SatelliteImage(models.Model):
    image_id = models.CharField(max_length=100, unique=True)
    date = models.DateTimeField()
    metadata = models.JSONField()

    def __str__(self):
        return self.image_id
