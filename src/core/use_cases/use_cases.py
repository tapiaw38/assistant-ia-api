from dataclasses import dataclass
from src.core.platform.appcontext.appcontext import Factory
from src.core.use_cases.conversation import use_case as conversation_use_case
from src.core.use_cases.profile import use_case as profile_use_case


@dataclass
class Profile:
    create_usecase: profile_use_case.CreateUseCase
    find_by_user_id_usecase: profile_use_case.FindByUserIdUseCase
    update_usecase: profile_use_case.UpdateUseCase
    change_status_usecase: profile_use_case.ChangeStatusUseCase


@dataclass
class Conversation:
    create_usecase: conversation_use_case.CreateUseCase
    find_by_user_id_usecase: conversation_use_case.FindByUserIdUseCase
    add_message_usecase: conversation_use_case.AddMessageUseCase
    delete_all_messages_usecase: conversation_use_case.DeleteAllMessagesUseCase


@dataclass
class Usecases:
    profile: Profile
    conversation: Conversation


def create_usecases(context_factory: Factory) -> Usecases:
    return Usecases(
        profile=Profile(
            create_usecase=profile_use_case.CreateUseCase(context_factory),
            find_by_user_id_usecase=profile_use_case.FindByUserIdUseCase(context_factory),
            update_usecase=profile_use_case.UpdateUseCase(context_factory),
            change_status_usecase=profile_use_case.ChangeStatusUseCase(context_factory),
        ),
        conversation=Conversation(
            create_usecase=conversation_use_case.CreateUseCase(context_factory),
            find_by_user_id_usecase=conversation_use_case.FindByUserIdUseCase(context_factory),
            add_message_usecase=conversation_use_case.AddMessageUseCase(context_factory),
            delete_all_messages_usecase=conversation_use_case.DeleteAllMessagesUseCase(context_factory),
        )
    )