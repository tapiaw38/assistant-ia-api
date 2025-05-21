from pymongo import MongoClient as PyMongoClient, collection, errors
from pymongo.results import InsertOneResult, UpdateResult, DeleteResult
from pymongo.cursor import Cursor
from typing import Protocol, Any, Optional, List
from src.core.platform.nosql.migrations import Migration
import logging


class Client(Protocol):
    def disconnect(self) -> None: ...
    def get_collection(self) -> collection.Collection: ...
    def insert_one(self, document: dict) -> InsertOneResult: ...
    def find(self, filter: dict) -> Cursor: ...
    def find_one(self, filter: dict) -> Optional[dict]: ...
    def update_many(self, filter: dict, update: dict) -> UpdateResult: ...
    def update_one(self, filter: dict, update: dict) -> UpdateResult: ...
    def delete_one(self, filter: dict) -> DeleteResult: ...
    def run_migrations(self, migration_collection: collection.Collection, migrations: List[Migration]) -> None: ...


class MongoDBClient(Client):
    def __init__(self, uri: str, database: str, collection_name: str):
        self.client = PyMongoClient(uri)
        self.db = self.client[database]
        self._collection = self.db[collection_name]

    def disconnect(self) -> None:
        self.client.close()

    def get_collection(self) -> collection.Collection:
        return self._collection

    def insert_one(self, document: dict) -> InsertOneResult:
        return self._collection.insert_one(document)

    def find(self, filter: dict) -> Cursor:
        return self._collection.find(filter)

    def find_one(self, filter: dict, projection: Optional[dict] = None) -> Optional[dict]:
        return self._collection.find_one(filter, projection)

    def update_many(self, filter: dict, update: dict) -> UpdateResult:
        return self._collection.update_many(filter, update)

    def update_one(self, filter: dict, update: dict) -> UpdateResult:
        return self._collection.update_one(filter, update)

    def delete_one(self, filter: dict) -> DeleteResult:
        return self._collection.delete_one(filter)

    def create_index(self, key: str, unique: bool = False) -> None:
        self._collection.create_index(key, unique=unique)

    def run_migrations(self, migration_collection: collection.Collection, migrations: List[Migration]) -> None:
        for migration in migrations:
            existing = migration_collection.find_one({
                "version": migration.version,
                "collection": self._collection.name,
            })

            if existing:
                logging.info(f"Migration version {migration.version} already exists, skipping")
                continue

            logging.info(f"Running migration version {migration.version}")
            migration.up(self.db)

            migration_collection.insert_one({
                "version": migration.version,
                "collection": self._collection.name,
            })

        logging.info("All migrations are up to date")
