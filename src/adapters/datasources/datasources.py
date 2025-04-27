from src.core.platform.nosql.client import Client

class Datasources:
    def __init__(
        self,
        no_sql_conversations_client: Client,
        no_sql_profiles_client: Client,
    ):
        self.no_sql_conversations_client = no_sql_conversations_client
        self.no_sql_profiles_client = no_sql_profiles_client

    @classmethod
    def create_datasources(
        cls,
        no_sql_hotel_client: Client,
        no_sql_locations_client: Client,
    ) -> "Datasources":
        return cls(
            no_sql_conversations_client=no_sql_hotel_client,
            no_sql_profiles_client=no_sql_locations_client,
        )