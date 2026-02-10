from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from dotenv import find_dotenv, load_dotenv


@pytest.fixture(scope="session", autouse=True)
def _load_env() -> None:
    # repo_root = Path(__file__).resolve().parents[1]
    # load_dotenv(repo_root / ".env", override=False)
    load_dotenv(find_dotenv(usecwd=True))



@pytest.fixture(scope="session", autouse=True)
def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if src.exists():
        sys.path.insert(0, str(src))


@pytest.fixture(scope="session")
def pollinations_api_key() -> str:
    key = os.getenv("POLLINATIONS_API_KEY")
    if not key:
        pytest.skip("Missing POLLINATIONS_API_KEY (load it from .env)")
    return key


@pytest.fixture()
def chat(pollinations_api_key: str):
    from langchain_pollinations.chat import ChatPollinations

    return ChatPollinations(
        api_key=pollinations_api_key,
        model="openai",
        temperature=0,
        max_tokens=512,
    )

