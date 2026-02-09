import os

import pytest
from dotenv import find_dotenv, load_dotenv

# Cargar .env lo más temprano posible (antes de pytest_collection_modifyitems)
load_dotenv(find_dotenv(usecwd=True))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    api_key = os.getenv("POLLINATIONS_API_KEY")
    for item in items:
        if "integration" in item.keywords and not api_key:
            item.add_marker(pytest.mark.skip(reason="Falta POLLINATIONS_API_KEY en entorno/.env"))
