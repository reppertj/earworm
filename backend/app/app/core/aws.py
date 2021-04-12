from typing import Any, Optional

import boto3

from .config import settings


def s3_preview_bucket() -> Optional[Any]:
    """ Returns None if s3 not configured in settings """
    if all(
        [
            var is not None
            for var in [
                settings.AWS_S3_KEY,
                settings.AWS_S3_SECRET,
                settings.AWS_S3_PREVIEW_BUCKET,
            ]
        ]
    ):
        s3 = boto3.Session(
            aws_access_key_id=settings.AWS_S3_KEY,
            aws_secret_access_key=settings.AWS_S3_SECRET,
        ).client("s3")
        return s3.Bucket(settings.AWS_S3_PREVIEW_BUCKET)
