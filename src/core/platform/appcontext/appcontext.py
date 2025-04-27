from typing import Callable, Optional
from src.core.platform.config.service import ConfigurationService
from src.adapters.datasources.datasources import Datasources
from src.adapters.integrations.integrations import Integrations
from src.adapters.datasources.repositories.repositories import Repositories

class Context:
    def __init__(
        self,
        repositories: Optional[Repositories] = None,
        integrations: Optional[Integrations] = None,
        config_service: Optional[ConfigurationService] = None,
    ):
        self.repositories = repositories
        self.integrations = integrations
        self.config_service = config_service

Option = Callable[[Context], None]
Factory = Callable[..., Context]

def new_factory(
    datasources: Datasources,
    integrations: Integrations,
    config_service: ConfigurationService
) -> Factory:
    def factory(*opts: Option) -> Context:
        context = Context(
            repositories=Repositories.create_repositories(datasources),
            integrations=integrations,
            config_service=config_service
        )
        for opt in opts:
            opt(context)
        return context
    return factory
