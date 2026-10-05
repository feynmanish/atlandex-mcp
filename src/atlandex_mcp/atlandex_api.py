"""Async client for the Atlandex backend routes this server uses.

- GET  /api/term/<term>                          cross-channel lookup of an extracted index term
- GET  /api/yt_videos/<video_id>/embeddings_ready  whether the video's transcript is searchable
- POST /api/yt_videos/<video_id>/query           dense retrieval over one video's transcript chunks
- POST /api/snippet-seek                         caption match for a phrase -> start_sec

The server only reads. It never calls routes that write, ingest or bill an LLM completion:
queries are sent with retrieve_only, so the backend embeds the question and skips its own
answer generation.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx

from . import __version__

_MAX_BACKEND_MESSAGE_CHARS = 200
_CHECK_URL = "; check that ATLANDEX_API_URL points at the backend's /api base"


class AtlandexError(Exception):
    """A backend failure. Subclass messages are written to be read by the agent."""


class NotIndexed(AtlandexError):
    """The video has no transcript embeddings in Atlandex."""


class RateLimited(AtlandexError):
    """The backend's per-client meter refused the request (HTTP 429)."""


class Unavailable(AtlandexError):
    """Timeout, connection failure, 5xx without a usable message, or a non-JSON body."""


class ServerError(Unavailable):
    """The route answered 5xx. Unlike a timeout, the backend is up and this one request failed."""


class BackendRejected(AtlandexError):
    """The backend refused the request and said why."""


def _backend_message(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    message = data.get("message") or data.get("error")
    if not isinstance(message, str) or not message.strip():
        return None
    message = " ".join(message.split())
    if len(message) > _MAX_BACKEND_MESSAGE_CHARS:
        message = message[: _MAX_BACKEND_MESSAGE_CHARS - 1] + "…"
    return message


class AtlandexAPI:
    def __init__(
        self,
        api_url: str,
        *,
        timeout_sec: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=api_url.rstrip("/") + "/",
            timeout=httpx.Timeout(timeout_sec, connect=10.0),
            headers={"User-Agent": f"atlandex-mcp/{__version__}", "Accept": "application/json"},
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def term_lookup(self, term: str, *, channel_id: str | None = None) -> dict[str, Any]:
        encoded = quote(term, safe="")
        if channel_id:
            path = f"channel/{quote(channel_id, safe='')}/term/{encoded}"
        else:
            path = f"term/{encoded}"
        return await self._request("GET", path)

    async def query_video(self, video_id: str, question: str, top_k: int) -> dict[str, Any]:
        return await self._request(
            "POST",
            f"yt_videos/{video_id}/query",
            json={"question": question, "top_k": top_k, "retrieve_only": True},
            not_found=NotIndexed(f"video {video_id} is not indexed"),
        )

    async def embeddings_ready(self, video_id: str) -> bool | None:
        """Whether the video has transcript embeddings; None if the answer is not a boolean."""
        data = await self._request("GET", f"yt_videos/{quote(video_id, safe='')}/embeddings_ready")
        ready = data.get("has_embeddings")
        return ready if isinstance(ready, bool) else None

    async def snippet_seek(self, video_id: str, phrase: str) -> dict[str, Any]:
        return await self._request("POST", "snippet-seek", json={"video_id": video_id, "query": phrase})

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        not_found: AtlandexError | None = None,
    ) -> dict[str, Any]:
        try:
            response = await self._client.request(method, path, json=json)
        except httpx.TimeoutException as exc:
            raise Unavailable("the request timed out") from exc
        except httpx.HTTPError as exc:
            raise Unavailable(f"could not connect ({type(exc).__name__})") from exc

        try:
            data: Any = response.json()
        except ValueError:
            data = None

        status = response.status_code
        message = _backend_message(data)
        # The query route's own 404 says "No embeddings found"; any other 404 means the
        # request missed the backend, usually a wrong ATLANDEX_API_URL.
        if status == 404 and not_found is not None and "embedding" in (message or "").lower():
            raise not_found
        if status == 404 and message is None:
            raise BackendRejected("HTTP 404" + _CHECK_URL)
        if status == 429:
            raise RateLimited("rate limit reached")
        if status >= 500:
            # snippet-seek answers 500 with {"status": "error", "message": ...} when it cannot
            # load captions; that message is useful. Other 5xx bodies can carry database
            # errors, which stay out of the agent's context.
            if isinstance(data, dict) and data.get("status") == "error" and message:
                raise BackendRejected(message)
            raise ServerError(f"HTTP {status}")
        if status >= 400:
            raise BackendRejected(message or f"HTTP {status}")
        if not isinstance(data, dict):
            # e.g. the public site's HTML instead of the backend's JSON
            raise Unavailable("the response was not a JSON object" + _CHECK_URL)
        return data
