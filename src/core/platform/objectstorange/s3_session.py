import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError
import logging
from src.core.platform.config.service import get_config_service

_session = None

def _init_s3_session():
    global _session

    try:
        _session = boto3.session.Session(
            aws_access_key_id=get_config_service().aws_config.access_key_id,
            aws_secret_access_key=get_config_service().aws_config.secret_access_key,
            region_name=get_config_service().aws_config.region,
        )
        return _session
    except (BotoCoreError, NoCredentialsError) as e:
        logging.error(f"objectstore: error initializing s3 session: {e}")
        raise e

def get_s3_session():
    global _session
    if _session is None:
        _init_s3_session()
    return _session
