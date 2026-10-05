"""Runtime configuration. Read from environment variables only, never from code."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

DEFAULT_SITE_URL = "https://www.atlandex.app"
DEFAULT_TIMEOUT_SEC = 30.0


class ConfigError(Exception):
    """Required configuration is missing or malformed. The message is shown to the agent."""


@dataclass(frozen=True)
class Settings:
    api_url: str
    """Base of the backend's /api routes, without a trailing slash."""

    site_url: str = DEFAULT_SITE_URL
    """Public Atlandex site, used for /v/<video_id>?t=<sec> landing links."""

    timeout_sec: float = DEFAULT_TIMEOUT_SEC

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env

        api_url = (env.get("ATLANDEX_API_URL") or "").strip().rstrip("/")
        if not api_url:
            raise ConfigError(
                "ATLANDEX_API_URL is not set. The user needs to add it to this MCP server's "
                "environment (the backend's /api base URL; see the README)."
            )
        if not api_url.startswith(("http://", "https://")):
            raise ConfigError("ATLANDEX_API_URL must start with http:// or https://.")

        site_url = (env.get("ATLANDEX_SITE_URL") or DEFAULT_SITE_URL).strip().rstrip("/")

        raw_timeout = (env.get("ATLANDEX_TIMEOUT_SEC") or "").strip()
        try:
            timeout_sec = float(raw_timeout) if raw_timeout else DEFAULT_TIMEOUT_SEC
        except ValueError as exc:
            raise ConfigError("ATLANDEX_TIMEOUT_SEC must be a number of seconds.") from exc
        if timeout_sec <= 0:
            raise ConfigError("ATLANDEX_TIMEOUT_SEC must be greater than zero.")

        return cls(api_url=api_url, site_url=site_url, timeout_sec=timeout_sec)
