"""
This module manages authentication configuration for the Pollinations API.
It handles retrieval of API keys from environment variables or direct input.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

ENV_API_KEY = "POLLINATIONS_API_KEY"


@dataclass(frozen=True, slots=True)
class AuthConfig:
    """
    Configuration container for Pollinations API authentication credentials.
    Ensures the API key is present and provides a centralized access point.
    """

    api_key: str

    @staticmethod
    def from_env_or_value(api_key: str | None) -> AuthConfig:
        """
        Create an AuthConfig instance from a provided value or environment variable.

        Args:
            api_key: Optional API key string provided by the user.

        Returns:
            An initialized AuthConfig instance containing the validated API key.

        Raises:
            ValueError: If no API key is found in both the argument and environment.
        """
        key = api_key or os.getenv(ENV_API_KEY)

        if not key:
            raise ValueError(
                "API key missing. Define POLLINATIONS_API_KEY in environment or pass api_key value"
            )
        return AuthConfig(api_key=key)
