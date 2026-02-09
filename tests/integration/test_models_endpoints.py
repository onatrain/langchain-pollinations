import os

import pytest

from langchain_pollinations.models import ModelInformation


@pytest.mark.integration
def test_list_text_models() -> None:
    api_key = os.environ["POLLINATIONS_API_KEY"]
    mi = ModelInformation(api_key=api_key)
    data = mi.list_text_models()
    assert data is not None

