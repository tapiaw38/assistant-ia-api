from typing import List
from uuid import uuid4


class Migration:
    def __init__(self, version: int, up: callable):
        self.version = version
        self.up = up


def execute_profile_migrations() -> List[Migration]:
    return [
        Migration(
            version=1,
            up=lambda db: db["profiles"].update_many(
                {},
                {"$set": {"is_active": True}}
            )
        ),
        Migration(
            version=2,
            up=lambda db: db["profiles"].update_many(
                {},
                {"$set": {"iteration_limit": 100}}
            )
        ),
        Migration(
            version=3,
            up=lambda db: db["profiles"].update_many(
                {},
                {"$set": {"files": []}}
            )
        ),
        Migration(
            version=4,
            up=lambda db: db["conversations"].update_many(
                {},
                {"$set": {"client_id": str(uuid4())}}
            )
        ),
        # Add more migrations here
    ]
