from django.db import models

class Messages(models.Model):
    message = models.TextField()
    answer = models.TextField(default='')
