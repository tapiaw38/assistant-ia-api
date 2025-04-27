from src.adapters.datasources.datasources import Datasources
from src.adapters.datasources.repositories.conversation.repository import Repository as ConversationRepository


class Repositories:
    def __init__(self, conversation: ConversationRepository):
        self.conversation = conversation

    @staticmethod
    def create_repositories(datasources: Datasources):
        return Repositories(
            conversation=ConversationRepository(datasources)
        )