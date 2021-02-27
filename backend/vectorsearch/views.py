from django.shortcuts import render
from rest_framework import viewsets

from .models import ResultVector
from .serializers import ResultVectorSerializer


class ResultViewSet(viewsets.ModelViewSet):
    """
    API endpoint that returns nearest neighbors
    with metadata
    """

    queryset = ResultVector.objects.all()
    serializer_class = ResultVectorSerializer
