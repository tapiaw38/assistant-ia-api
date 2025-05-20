import os
import mimetypes
from typing import BinaryIO
import boto3
from botocore.config import Config
from typing import Protocol
from src.core.platform.config.service import ConfigurationService

class StoreService(Protocol):
    def put_object(self, file: BinaryIO, user_file: str, filename: str, size: int, object_id: str) -> str:
        ...

    def generate_url(self, key_name: str) -> str:
        ...

    def delete_object(self, key_name: str) -> None:
        ...

    def get_object(self, key_name: str) -> bytes:
        ...


class StoreServiceImpl:
    def __init__(self, session: boto3.session.Session, config: ConfigurationService):
        self.session = session
        self.config = config
        self.client = session.client("s3", config=Config(signature_version="s3v4"))

    def put_object(self, file: BinaryIO, user_file: str, filename: str, size: int, object_id: str) -> str:
        name, ext = os.path.splitext(filename)
        temp_file_name = f"files/{user_file}/{name}{object_id}{ext}"
        buffer = file.read()

        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

        self.client.put_object(
            Bucket=self.config.aws_config.bucket_name,
            Key=temp_file_name,
            ACL="public-read",
            Body=buffer,
            ContentLength=size,
            ContentType=content_type,
            ContentDisposition="attachment",
            ServerSideEncryption="AES256",
            StorageClass="INTELLIGENT_TIERING",
        )

        return temp_file_name

    def generate_url(self, key_name: str) -> str:
        return f"https://{self.config.aws_config.bucket_name}.s3.{self.config.aws_config.region}.amazonaws.com/{key_name}"

    def delete_object(self, key_name: str) -> None:
        self.client.delete_object(
            Bucket=self.config.aws_config.bucket_name,
            Key=key_name
        )

    def get_object(self, key_name: str) -> bytes:
        response = self.client.get_object(
            Bucket=self.config.aws_config.bucket_name,
            Key=key_name
        )
        return response['Body'].read()
