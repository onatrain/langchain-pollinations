from __future__ import annotations

import os
from dataclasses import dataclass

ENV_API_KEY = "POLLINATIONS_API_KEY"


@dataclass(frozen=True, slots=True)
class AuthConfig:
    api_key: str

    @staticmethod
    def from_env_or_value(api_key: str | None) -> "AuthConfig":
        key = api_key or os.getenv(ENV_API_KEY)
        if not key:
            raise ValueError("API key missing. Define POLLINATIONS_API_KEY in environment or pass api_key value")
        return AuthConfig(api_key=key)
