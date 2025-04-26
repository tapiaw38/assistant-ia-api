from abc import ABC, abstractmethod
from pymongo.database import Database

class Migration(ABC):
    version: int

    @abstractmethod
    def up(self, db: Database) -> None:
        pass

    @abstractmethod
    def down(self, db: Database) -> None:
        pass
