from dataclasses import dataclass
from src.core.use_cases.conversation.use_case import CreateUseCase, FindByUserIdUseCase, AddMessageUseCase
from src.core.platform.appcontext.appcontext import Factory

@dataclass
class Conversation:
    create_usecase: CreateUseCase
    find_by_user_id_usecase: FindByUserIdUseCase
    add_message_usecase: AddMessageUseCase

@dataclass
class Usecases:
    conversation: Conversation

def create_usecases(context_factory: Factory) -> Usecases:
    return Usecases(
        conversation=Conversation(
            create_usecase=CreateUseCase(context_factory),
            find_by_user_id_usecase=FindByUserIdUseCase(context_factory),
            add_message_usecase=AddMessageUseCase(context_factory),
        )
    )