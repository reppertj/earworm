from rest_framework import serializers

from .models import ResultVector


class ResultVectorSerializer(serializers.ModelSerializer):
    class Meta:
        model = ResultVector
        fields = ("song", "embedder", "vector")
        depth = 2
