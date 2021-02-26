import uuid

from django.db import models


class License(models.Model):
    abbreviation = models.CharField(max_length=20)
    name = models.CharField(max_length=50)
    description = models.TextField()


class Artist(models.Model):
    name = models.CharField(max_length=50, on_delete=models.SET_NULL)


class Album(models.Model):
    name = models.CharField(max_length=50)
    artist = models.ForeignKey(Artist, on_delete=models.SET_NULL)


class Embedder(models.Model):
    name = models.CharField(max_length=50)
    vector_length = models.PositiveSmallIntegerField()
    precision = models.PositiveSmallIntegerField()


class BaseVector(models.Model):
    vector = models.JSONField(encoder=None, decoder=None)

    class Meta:
        abstract = True


class Song(models.Model):
    external_url = models.URLField(max_length=200)
    licenses = models.ManytoMany(License, related_name="songs")
    artist = models.ForeignKey(Artist, on_delete=models.SET_NULL)
    album = models.ForeignKey(Album, on_delete=models.SET_NULL)


class ResultVector(BaseVector):
    song = models.ForeignKey(Song, on_delete=models.CASCADE)
    embedder = models.ForeignKey(Embedder, on_delete=models.CASCADE)


class QueryVector(BaseVector):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    embedder = models.ForeignKey(Embedder, on_delete=models.PROTECT)
