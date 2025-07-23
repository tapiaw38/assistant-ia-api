from fastapi import APIRouter, status, Body
from src.core.platform.nosql.client import Client, MongoDBClient
from typing import Optional, List, Any
from src.core.platform.config.service import (
    get_config_service
)

def insert_one_handler(client: Client, document: dict) -> dict:
    result = client.insert_one(document)
    return {"inserted_id": str(result.inserted_id)}


def find_handler(client: Client, filter: dict) -> List[dict]:
    cursor = client.find(filter)
    return [doc for doc in cursor]


def find_one_handler(client: Client, filter: dict) -> Optional[dict]:
    return client.find_one(filter)


def update_one_handler(client: Client, filter: dict, update: dict) -> dict:
    result = client.update_one(filter, update)
    return {"matched_count": result.matched_count, "modified_count": result.modified_count}


def delete_one_handler(client: Client, filter: dict) -> dict:
    result = client.delete_one(filter)
    return {"deleted_count": result.deleted_count}


router = APIRouter(
    prefix="/nosql",
    tags=["nosql"],
)

def get_instance() -> get_config_service:
    return get_config_service()

def select_client(collection: str) -> Client:
    config = get_config_service().nosql_config
    if collection == "profiles":
        profiles = config.profiles
        return MongoDBClient(
            profiles.database_uri,
            profiles.database,
            profiles.collection
        )
    elif collection == "conversation":
        conversation = config.conversation
        return MongoDBClient(
            conversation.database_uri,
            conversation.database,
            conversation.collection
        )
    else:
        raise ValueError("Colección no soportada: " + str(collection))


@router.post("/search", status_code=status.HTTP_200_OK)
async def search_documents(
    body: dict = Body(...),
    skip: int = 0,
    limit: int = 10
):
    collection = body.get("collection")
    filter = body.get("filter", {})
    client = select_client(collection)
    cursor = client.find(filter).skip(skip).limit(limit)
    return [doc for doc in cursor]


@router.post("/upsert", status_code=status.HTTP_200_OK)
async def upsert_document(
    body: dict = Body(...)
):
    collection = body.get("collection")
    filter = body.get("filter", {})
    update = body.get("update", {})
    client = select_client(collection)
    result = client.update_one(filter, {"$set": update})
    if result.matched_count == 0:
        return insert_one_handler(client, update)
    return {"matched_count": result.matched_count, "modified_count": result.modified_count}


@router.post("/upsert-bulk", status_code=status.HTTP_200_OK)
async def upsert_bulk_documents(
    body: dict = Body(...)
):
    collection = body.get("collection")
    documents = body.get("documents", [])
    client = select_client(collection)
    inserted = []
    for doc in documents:
        result = client.update_one({"_id": doc.get("_id")}, {"$set": doc})
        if result.matched_count == 0:
            res = client.insert_one(doc)
            inserted.append(str(res.inserted_id))
    return {"inserted_ids": inserted}


@router.post("/delete", status_code=status.HTTP_200_OK)
async def delete_documents(
    body: dict = Body(...)
):
    collection = body.get("collection")
    filter = body.get("filter", {})
    client = select_client(collection)
    result = client.delete_one(filter)
    return {"deleted_count": result.deleted_count}


@router.post("/count", status_code=status.HTTP_200_OK)
async def count_documents(
    body: dict = Body(...)
):
    collection = body.get("collection")
    filter = body.get("filter", {})
    client = select_client(collection)
    count = client.get_collection().count_documents(filter)
    return {"count": count}


@router.post("/find-first", status_code=status.HTTP_200_OK)
async def find_first_document(
    body: dict = Body(...)
):
    collection = body.get("collection")
    filter = body.get("filter", {})
    client = select_client(collection)
    doc = client.find_one(filter)
    return doc if doc else {}
