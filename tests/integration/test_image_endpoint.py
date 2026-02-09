import os

import pytest

from langchain_pollinations.image import ImagePollinations


@pytest.mark.integration
def test_generate_image_bytes() -> None:
    api_key = os.environ["POLLINATIONS_API_KEY"]
    im = ImagePollinations(api_key=api_key)
    content = im.generate("a minimal black and white icon of a bee")
    assert isinstance(content, (bytes, bytearray))
    assert len(content) > 10

