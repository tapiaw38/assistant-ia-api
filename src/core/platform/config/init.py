from src.core.platform.config.service import (
    read_config,
    init_config_service,
)


def init_config() -> None:
    config_service = read_config()
    init_config_service(config_service)

    return config_service