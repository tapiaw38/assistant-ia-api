import os

PROD_SCOPE = "prod"
TEST_SCOPE = "test"

def get_scope() -> str:
    return os.getenv("SCOPE", "")

def is_prod_scope() -> bool:
    return get_scope() == PROD_SCOPE

def is_test_scope() -> bool:
    return get_scope() == TEST_SCOPE

def is_local_scope() -> bool:
    return get_scope() == ""