from src.adapters.datasources.datasources import Datasources
from src.adapters.datasources.repositories.conversation.repository import Repository as ConversationRepository
from src.adapters.datasources.repositories.profile.repository import Repository as ProfileRepository

class Repositories:
    def __init__(
            self,
            conversation: ConversationRepository,
            profile: ProfileRepository,
    ):
        self.conversation = conversation
        self.profile = profile

    @staticmethod
    def create_repositories(datasources: Datasources):
        return Repositories(
            conversation=ConversationRepository(datasources),
            profile=ProfileRepository(datasources),
        )