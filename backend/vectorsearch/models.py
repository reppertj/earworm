import uuid

from django.db import models

from .utils.json_tools import NumpyEncoder


class License(models.Model):
    abbreviation = models.CharField(max_length=20, editable=False)
    name = models.CharField(max_length=50, editable=False)
    description = models.TextField(editable=False)


class Artist(models.Model):
    name = models.CharField(max_length=50, editable=False)


class Album(models.Model):
    name = models.CharField(max_length=50, editable=False)
    artist = models.ForeignKey(Artist, on_delete=models.SET_NULL, null=True, editable=False)


class Embedder(models.Model):
    name = models.CharField(max_length=50, editable=False)
    vector_length = models.PositiveSmallIntegerField(editable=False)
    precision = models.PositiveSmallIntegerField(editable=False)


class BaseVector(models.Model):
    vector = models.JSONField(encoder=NumpyEncoder)

    class Meta:
        abstract = True


class Song(models.Model):
    external_url = models.URLField(max_length=200, editable=False)
    licenses = models.ManyToManyField(License, related_name="songs", editable=False)
    artist = models.ForeignKey(Artist, on_delete=models.SET_NULL, null=True, editable=False)
    album = models.ForeignKey(Album, on_delete=models.SET_NULL, null=True, editable=False)


class ResultVector(BaseVector):
    song = models.ForeignKey(Song, on_delete=models.CASCADE, editable=False)
    embedder = models.ForeignKey(Embedder, on_delete=models.CASCADE, editable=False)


class QueryVector(BaseVector):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    spectrogram_hash = models.CharField(max_length=200, null=True)
    embedder = models.ForeignKey(Embedder, on_delete=models.PROTECT)
