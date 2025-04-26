from pymongo import MongoClient as PyMongoClient
from pymongo import collection
from dotenv import load_dotenv
from typing import Optional, Generator
import os

from src.core.platform.nosql.migrations import Migration
from src.core.platform.nosql.client import MongoDBClient

load_dotenv()


DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'conversation')

DB_URI = f"mongodb://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

if not DB_URI or not DB_NAME:
    raise ValueError("MONGO_URI and DB_NAME must be set in .env")

mongo_client: Optional[MongoDBClient] = MongoDBClient(
    uri=DB_URI,
    database=DB_NAME,
    collection_name=COLLECTION_NAME
)

def get_mongo_client() -> MongoDBClient:
    if not mongo_client.ping():
        raise ConnectionError("Failed to connect to MongoDB")
    return mongo_client

def get_mongo_collection() -> collection.Collection:
    return mongo_client.get_collection()

def get_db() -> Generator[MongoDBClient, None, None]:
    try:
        yield mongo_client
    except Exception as e:
        raise e
