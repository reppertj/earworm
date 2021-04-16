from typing import Generator, Optional, Literal
import botocore

import boto3
from botocore.exceptions import ClientError
import logging

from .config import settings


if all((settings.S3_KEY, settings.S3_SECRET)):
    s3 = boto3.Session(
        aws_access_key_id=settings.S3_KEY,
        aws_secret_access_key=settings.S3_SECRET,
    ).client(
        service_name="s3",
        region_name=settings.S3_REGION_NAME,
        endpoint_url=settings.S3_ENDPOINT_URL,
    )
    s3_is_setup = True
else:
    s3 = None
    s3_is_setup = False


bucket = settings.S3_PREVIEW_BUCKET


def if_s3_setup(func):
    if s3_is_setup:
        return func
    else:

        def noop(*args, **kwargs):
            return None

        return noop


@if_s3_setup
def object_on_s3(key: str) -> Optional[bool]:
    try:
        s3.head_object(Bucket=bucket, Key=key)
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        else:
            raise e
    else:
        return True


@if_s3_setup
def upload_to_s3(key: str, file_path: str, check_first=False) -> None:
    if not check_first or not object_on_s3(key=key):
        s3.upload_file(Filename=file_path, Bucket=bucket, Key=key)

@if_s3_setup
def remove_from_s3(key: str):
    s3.delete_object(Bucket=bucket, Key=key)

@if_s3_setup
def all_keys_on_s3(
    prefix="/", delimiter="/", start_after=""
) -> Generator[str, None, None]:
    s3_paginator = s3.get_paginator("list_objects_v2")
    prefix = prefix[1:] if prefix.startswith(delimiter) else prefix
    start_after = (start_after or prefix) if prefix.endswith(delimiter) else start_after
    for page in s3_paginator.paginate(
        Bucket=bucket, Prefix=prefix, StartAfter=start_after
    ):
        for content in page.get("Contents", ()):
            yield content["Key"]


@if_s3_setup
def download_from_s3(key: str, file_path: str) -> Optional[Literal[True]]:
    """ Downloads `key` to `file_path` from the bucket specified in config """
    s3.download_file(Bucket=bucket, Key=key, Filename=file_path)
    return True


@if_s3_setup
def create_presigned_url(key: str, expiration=3600) -> Optional[str]:
    """Generate a presigned URL to share an S3 object from the preview bucket

    :param key: object name as string
    :param object_name: string
    :param expiration: Time in seconds for the presigned URL to remain valid
    :return: Presigned URL as string. If error, returns None.
    """

    # Generate a presigned URL for the S3 object
    try:
        response = s3.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expiration
        )
    except ClientError as e:
        logging.error(e)
        return None

    # The response contains the presigned URL
    return response
