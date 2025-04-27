from adapters.services.conversation.service import ConversationService

class Services:
    _instance = None

    def __init__(
        self,
        conversation: ConversationService,
    ):
        self.conversation = conversation

    @classmethod
    def create_services(
        cls,
        conversation: ConversationService,
    ) -> "Services":
        if cls._instance is None:
            cls._instance = cls(
                conversation=conversation,
            )
        return cls._instance

    @classmethod
    def get_instance(cls) -> "Services":
        if cls._instance is None:
            raise Exception("Services has not been initialized. Call `create_services` first.")
        return cls._instance