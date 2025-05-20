from typing import Callable, Optional
from src.core.platform.config.service import ConfigurationService
from src.adapters.datasources.datasources import Datasources
from src.adapters.web.integrations.integrations import Integrations
from src.adapters.datasources.repositories.repositories import Repositories
from src.adapters.storeservice.storeservice import StoreService


class Context:
    def __init__(
        self,
        repositories: Optional[Repositories] = None,
        integrations: Optional[Integrations] = None,
        store_service: Optional[StoreService] = None,
        config_service: Optional[ConfigurationService] = None,
    ):
        self.repositories = repositories
        self.integrations = integrations
        self.store_service = store_service
        self.config_service = config_service

Option = Callable[[Context], None]
Factory = Callable[..., Context]

def new_factory(
    datasources: Datasources,
    integrations: Integrations,
    store_service: StoreService,
    config_service: ConfigurationService
) -> Factory:
    def factory(*opts: Option) -> Context:
        context = Context(
            repositories=Repositories.create_repositories(datasources),
            integrations=integrations,
            store_service=store_service,
            config_service=config_service
        )
        for opt in opts:
            opt(context)
        return context
    return factory
