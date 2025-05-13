from typing import List


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
        # Add more migrations here
    ]
