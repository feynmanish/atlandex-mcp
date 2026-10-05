"""A stand-in Atlandex backend for tests, built from the real routes' response shapes.

The transcript text below is synthetic and written for these tests.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx

from atlandex_mcp.atlandex_api import AtlandexAPI
from atlandex_mcp.config import Settings

VIDEO_ID = "AbCdEfGhIjK"
API_URL = "https://backend.test/api"
SETTINGS = Settings(api_url=API_URL)

CHUNK_INTRO = (
    "welcome back to the channel today we are looking at a search system for long videos "
    "the idea is simple you paste a link and the system indexes the talk so that people can "
    "jump straight to the part they care about we will start with ingestion then talk about "
    "chunking and at the end we will look at numbers from a small test set that we built by "
    "hand over a few evenings because nobody else had one for this kind of content"
)

_EVAL_LEAD = (
    "so before we get to the results let me say a word about the setup we had twenty two "
    "questions each written by hand against transcripts we knew well and for every question we "
    "marked which passage a good answer should come from some questions had more than one good "
    "passage and we counted a hit if any of them showed up in the top three that matters later "
    "because it makes the metric easier to saturate than people expect when they hear the number "
    "anyway we ran the first version and honestly the coffee machine in that office was the "
    "highlight of the week it made a very loud noise every morning at nine"
)
_EVAL_CORE = (
    "here is the surprise the dense retrieval run lost to the plain TF-IDF baseline on these "
    "questions and the reason was vocabulary the questions used product names and acronyms that "
    "the embedding model had rarely seen so exact word overlap beat semantic similarity when we "
    "fused the two rankings with reciprocal rank fusion recall at three reached one point zero "
    "which sounds perfect but on twenty two questions it mostly tells you the test set is too "
    "easy now"
)
CHUNK_EVAL = f"{_EVAL_LEAD} {_EVAL_CORE}"

# Where each fixture chunk starts in the fake video, and a speaking rate for
# turning a phrase's word offset into a time, so seeks are deterministic.
_CHUNK_START_SEC = ((CHUNK_EVAL, 600), (CHUNK_INTRO, 0))
_WORDS_PER_SEC = 2


def expected_start(phrase: str) -> int | None:
    """The start_sec the fake seek route returns for `phrase`, or None if it falls back."""
    if not phrase:
        return None
    for chunk, chunk_start in _CHUNK_START_SEC:
        index = chunk.find(phrase)
        if index >= 0:
            return chunk_start + len(chunk[:index].split()) // _WORDS_PER_SEC
    return None


def query_body(chunks: list[str] | None = None) -> dict[str, Any]:
    chunks = [CHUNK_EVAL, CHUNK_INTRO] if chunks is None else chunks
    return {
        "videoid": VIDEO_ID,
        "question": "q",
        "answer": None,
        "chunks": [
            {"text": text, "score": 0.42 + i * 0.1, "rank": i + 1} for i, text in enumerate(chunks)
        ],
        "context": "\n\n".join(chunks),
        "embedding_model": "test-embedding-model",
        "chunk_count": 14,
        "retrieve_only": True,
    }


def seek_body(start_sec: int, match_source: str = "transcript") -> dict[str, Any]:
    return {
        "status": "success",
        "videoid": VIDEO_ID,
        "video_id": VIDEO_ID,
        "start_sec": start_sec,
        "match_source": match_source,
        "title": "Searching long videos",
        "channelTitle": "Test Channel",
        "thumbnail": f"https://i.ytimg.com/vi/{VIDEO_ID}/hqdefault.jpg",
        "url": f"https://www.youtube.com/watch?v={VIDEO_ID}",
        "landing_path": f"/v/{VIDEO_ID}?t={start_sec}",
        "terms_pending": True,
    }


def default_seek(phrase: str) -> tuple[int, dict[str, Any]]:
    """Locate phrases from the fixture chunks; anything else falls back like the real route."""
    start = expected_start(phrase)
    if start is None:
        return 200, seek_body(0, match_source="fallback")
    return 200, seek_body(start)


def term_body(rows: list[dict[str, Any]] | None = None, term: str = "reciprocal rank fusion") -> dict[str, Any]:
    if rows is None:
        rows = [
            term_row("AbCdEfGhIjK", "Searching long videos", 950, 754, "Evaluation"),
            term_row("ZyXwVuTsRqP", "Hybrid search in practice", 800, 3725, "Fusion"),
            term_row("MnOpQrStUvW", "Retrieval basics", 600, None, None),
        ]
    return {"term": term, "results": rows}


def term_row(
    video_id: str, title: str, relevance: int, seconds: int | None, chapter: str | None
) -> dict[str, Any]:
    return {
        "videoid": video_id,
        "title": title,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "channel_title": "Test Channel",
        "channel_thumb": None,
        "audio_url": None,
        "term_relevance": relevance,
        "seconds": seconds,
        "chapter_title": chapter,
        "user_relevance": 500,
        "matched_term_relevance": relevance,
    }


class FakeBackend:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.query_status = 200
        self.query_payload: Any = query_body()
        self.term_status = 200
        self.term_payload: Any = term_body()
        self.seek: Callable[[str], tuple[int, Any]] = default_seek
        # Videos absent from `ready` are searchable. `ready_status` overrides every check.
        self.ready: dict[str, bool] = {}
        self.ready_status = 200

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path.startswith("/api/yt_videos/") and path.endswith("/query"):
            return httpx.Response(self.query_status, json=self.query_payload)
        if path.startswith("/api/yt_videos/") and path.endswith("/embeddings_ready"):
            video_id = path.split("/")[-2]
            if self.ready_status != 200:
                return httpx.Response(self.ready_status, json={"error": "Internal server error"})
            return httpx.Response(
                200, json={"has_embeddings": self.ready.get(video_id, True), "videoid": video_id}
            )
        if path == "/api/snippet-seek":
            status, payload = self.seek(json.loads(request.content)["query"])
            return httpx.Response(status, json=payload)
        if "/term/" in path:
            return httpx.Response(self.term_status, json=self.term_payload)
        return httpx.Response(404, json={"error": "unknown route"})

    def api(self) -> AtlandexAPI:
        return AtlandexAPI(API_URL, transport=httpx.MockTransport(self.handler))

    def requests_to(self, suffix: str) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.path.endswith(suffix)]
