"""End-to-end through the MCP protocol: an in-process client calls the real server,
which talks to a fake Atlandex backend over a mocked HTTP transport."""

import json

import httpx
import pytest
from mcp import Client

from atlandex_mcp.atlandex_api import AtlandexAPI
from atlandex_mcp.passages import select_passage
from atlandex_mcp.server import create_server
from atlandex_mcp.youtube import format_timestamp

from fakes import (
    API_URL,
    CHUNK_EVAL,
    CHUNK_INTRO,
    SETTINGS,
    VIDEO_ID,
    FakeBackend,
    expected_start,
    query_body,
    term_body,
)

pytestmark = pytest.mark.anyio

QUESTION = "Why did dense retrieval lose to TF-IDF?"
NO_TRANSCRIPT = (
    f"Video {VIDEO_ID} has no searchable transcript in Atlandex. Do not retry search_video on "
    "it. Cite the chapter link you already have for it, or use locate_quote to time a "
    "specific phrase from its captions."
)
QUERY_FAILED = (
    "Atlandex failed searching this video; retry once, then tell the user this video "
    "can't be searched right now."
)


async def call(fake: FakeBackend, tool: str, arguments: dict):
    api = fake.api()
    try:
        async with Client(create_server(settings=SETTINGS, api=api)) as client:
            return await client.call_tool(tool, arguments)
    finally:
        await api.aclose()


async def test_lists_three_read_only_tools_with_output_schemas():
    api = FakeBackend().api()
    async with Client(create_server(settings=SETTINGS, api=api)) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        instructions = client.instructions
    await api.aclose()

    assert set(tools) == {"find_videos", "search_video", "locate_quote"}
    for tool in tools.values():
        assert tool.annotations.read_only_hint is True
        assert tool.annotations.destructive_hint is False
        assert tool.output_schema is not None
        assert tool.description
    assert "find_videos" in instructions and "search_video" in instructions


async def test_search_video_returns_passages_located_in_the_captions():
    fake = FakeBackend()
    result = await call(
        fake, "search_video", {"video": f"https://youtu.be/{VIDEO_ID}?si=x", "question": QUESTION, "top_k": 2}
    )
    assert result.is_error is False
    data = result.structured_content

    (query,) = fake.requests_to("/query")
    assert json.loads(query.content) == {"question": QUESTION, "top_k": 2, "retrieve_only": True}

    expected = select_passage(CHUNK_EVAL, QUESTION)
    start = expected_start(expected.anchor)
    first = data["passages"][0]
    assert "TF-IDF baseline" in first["text"]
    assert first["text"] == expected.text
    assert first["start_sec"] == start
    assert first["timestamp"] == format_timestamp(start)
    assert first["located_by"] == "transcript"
    assert first["youtube_url"] == f"https://www.youtube.com/watch?v={VIDEO_ID}&t={start}s"
    assert first["atlandex_url"] == f"https://www.atlandex.app/v/{VIDEO_ID}?t={start}"
    assert first["citation"] == (
        f"[Searching long videos, {format_timestamp(start)}]"
        f"(https://www.youtube.com/watch?v={VIDEO_ID}&t={start}s)"
    )
    assert first["distance"] == pytest.approx(0.42)
    assert [p["rank"] for p in data["passages"]] == [1, 2]
    assert data["title"] == "Searching long videos"
    assert data["channel"] == "Test Channel"
    assert data["chunks_in_video"] == 14
    assert data["note"] is None

    seek_phrases = sorted(json.loads(r.content)["query"] for r in fake.requests_to("/snippet-seek"))
    assert seek_phrases == sorted([expected.anchor, select_passage(CHUNK_INTRO, QUESTION).anchor])


async def test_passages_are_much_shorter_than_chunks():
    fake = FakeBackend()
    long_chunk = " ".join([CHUNK_EVAL] * 8)
    fake.query_payload = query_body([long_chunk])
    result = await call(fake, "search_video", {"video": VIDEO_ID, "question": QUESTION, "top_k": 1})
    text = result.structured_content["passages"][0]["text"]
    assert len(text) < len(long_chunk) / 5


async def test_search_still_answers_when_captions_cannot_be_read():
    fake = FakeBackend()
    fake.seek = lambda _phrase: (500, {"status": "error", "message": "Could not load captions from YouTube"})
    result = await call(fake, "search_video", {"video": VIDEO_ID, "question": QUESTION})
    assert result.is_error is False
    data = result.structured_content
    assert data["passages"]
    assert all(p["start_sec"] is None for p in data["passages"])
    assert all(p["youtube_url"] == f"https://www.youtube.com/watch?v={VIDEO_ID}" for p in data["passages"])
    assert "no timestamp" in data["note"]
    assert data["title"] is None
    # no title and no timestamp: the video ID is the label and the link is the plain one
    assert all(p["citation"] == f"[{VIDEO_ID}](https://www.youtube.com/watch?v={VIDEO_ID})" for p in data["passages"])


async def test_unmatched_passage_has_no_timestamp_but_others_keep_theirs():
    fake = FakeBackend()
    fake.query_payload = query_body([CHUNK_EVAL, "an unrelated passage nobody ever said in this video at all"])
    data = (await call(fake, "search_video", {"video": VIDEO_ID, "question": QUESTION})).structured_content
    located, unlocated = data["passages"]
    assert located["start_sec"] is not None
    assert unlocated["start_sec"] is None and unlocated["located_by"] is None
    assert located["citation"].startswith("[Searching long videos, ") and "&t=" in located["citation"]
    assert unlocated["citation"] == f"[Searching long videos](https://www.youtube.com/watch?v={VIDEO_ID})"
    assert data["note"] is not None


async def test_video_without_embeddings_is_not_retried_and_does_not_loop_to_find_videos():
    fake = FakeBackend()
    fake.query_status = 404
    fake.query_payload = {"error": "No embeddings found for this video"}
    result = await call(fake, "search_video", {"video": VIDEO_ID, "question": QUESTION})
    assert result.is_error is True
    text = result.content[0].text
    assert NO_TRANSCRIPT in text
    assert "find_videos" not in text
    assert fake.requests_to("/snippet-seek") == []
    assert fake.requests_to("/embeddings_ready") == []


@pytest.mark.parametrize(
    ("status", "payload", "expected", "absent"),
    [
        (429, {"error": "Daily limit reached"}, "rate limit", None),
        (
            500,
            {"error": "Database query failed: password authentication failed for user app"},
            "failed searching this video",
            "password",
        ),
    ],
)
async def test_backend_failures_become_actionable_errors(status, payload, expected, absent):
    fake = FakeBackend()
    fake.query_status = status
    fake.query_payload = payload
    result = await call(fake, "search_video", {"video": VIDEO_ID, "question": QUESTION})
    assert result.is_error is True
    text = result.content[0].text
    assert expected in text
    if absent:
        assert absent not in text


async def test_query_5xx_on_a_video_without_embeddings_is_a_missing_transcript():
    fake = FakeBackend()
    fake.query_status = 500
    fake.query_payload = {"error": "Internal server error"}
    fake.ready[VIDEO_ID] = False
    result = await call(fake, "search_video", {"video": VIDEO_ID, "question": QUESTION})
    assert result.is_error is True
    assert NO_TRANSCRIPT in result.content[0].text
    assert [r.method for r in fake.requests_to("/embeddings_ready")] == ["GET"]


@pytest.mark.parametrize("ready_status", [200, 500])
async def test_query_5xx_with_embeddings_or_unknown_readiness_asks_for_one_retry(ready_status):
    fake = FakeBackend()
    fake.query_status = 500
    fake.query_payload = {"error": "Internal server error"}
    fake.ready_status = ready_status  # 200: has_embeddings is true; 500: the check itself fails
    result = await call(fake, "search_video", {"video": VIDEO_ID, "question": QUESTION})
    assert result.is_error is True
    text = result.content[0].text
    assert QUERY_FAILED in text
    assert "backend" not in text.lower()


async def test_query_timeout_still_reports_the_backend_as_unavailable():
    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    api = AtlandexAPI(API_URL, transport=httpx.MockTransport(timeout))
    async with Client(create_server(settings=SETTINGS, api=api)) as client:
        result = await client.call_tool("search_video", {"video": VIDEO_ID, "question": QUESTION})
    await api.aclose()
    assert result.is_error is True
    assert "backend is unavailable" in result.content[0].text


async def test_rejects_invalid_video_before_calling_the_backend():
    fake = FakeBackend()
    result = await call(fake, "search_video", {"video": "https://example.com/watch", "question": QUESTION})
    assert result.is_error is True
    assert "not a YouTube URL" in result.content[0].text
    assert fake.requests == []


async def test_top_k_is_bounded_by_the_schema():
    fake = FakeBackend()
    result = await call(fake, "search_video", {"video": VIDEO_ID, "question": QUESTION, "top_k": 9})
    assert result.is_error is True
    assert fake.requests == []


async def test_find_videos_returns_timestamped_hits():
    fake = FakeBackend()
    result = await call(fake, "find_videos", {"term": "  reciprocal   rank fusion ", "limit": 2})
    data = result.structured_content
    assert fake.requests[0].url.raw_path == b"/api/term/reciprocal%20rank%20fusion"
    assert data["term"] == "reciprocal rank fusion"
    assert data["total_matches"] == 3
    assert [v["video_id"] for v in data["videos"]] == ["AbCdEfGhIjK", "ZyXwVuTsRqP"]
    second = data["videos"][1]
    assert (second["relevance"], second["chapter_title"], second["timestamp"]) == (800, "Fusion", "1:02:05")
    assert second["youtube_url"] == "https://www.youtube.com/watch?v=ZyXwVuTsRqP&t=3725s"
    assert second["citation"] == "[Hybrid search in practice, 1:02:05](https://www.youtube.com/watch?v=ZyXwVuTsRqP&t=3725s)"
    assert data["hint"] is None


async def test_find_videos_flags_which_videos_are_searchable():
    fake = FakeBackend()
    fake.ready["ZyXwVuTsRqP"] = False
    data = (await call(fake, "find_videos", {"term": "rrf"})).structured_content
    assert [v["searchable"] for v in data["videos"]] == [True, False, True]
    checked = sorted(r.url.path for r in fake.requests_to("/embeddings_ready"))
    assert checked == [
        "/api/yt_videos/AbCdEfGhIjK/embeddings_ready",
        "/api/yt_videos/MnOpQrStUvW/embeddings_ready",
        "/api/yt_videos/ZyXwVuTsRqP/embeddings_ready",
    ]


async def test_find_videos_checks_only_the_videos_it_returns():
    fake = FakeBackend()
    await call(fake, "find_videos", {"term": "rrf", "limit": 1})
    assert len(fake.requests_to("/embeddings_ready")) == 1


async def test_find_videos_searchable_is_null_when_the_check_fails():
    fake = FakeBackend()
    fake.ready_status = 500
    result = await call(fake, "find_videos", {"term": "rrf"})
    assert result.is_error is False
    data = result.structured_content
    assert [v["searchable"] for v in data["videos"]] == [None, None, None]
    assert data["total_matches"] == 3


async def test_find_videos_handles_rows_without_a_chapter():
    data = (await call(FakeBackend(), "find_videos", {"term": "rrf"})).structured_content
    third = data["videos"][2]
    assert third["start_sec"] is None and third["timestamp"] is None
    assert third["youtube_url"] == "https://www.youtube.com/watch?v=MnOpQrStUvW"
    assert third["citation"] == "[Retrieval basics](https://www.youtube.com/watch?v=MnOpQrStUvW)"


async def test_find_videos_can_scope_to_a_channel():
    fake = FakeBackend()
    await call(fake, "find_videos", {"term": "rrf", "channel_id": "UC123"})
    assert fake.requests[0].url.raw_path == b"/api/channel/UC123/term/rrf"


async def test_find_videos_without_matches_suggests_a_shorter_term():
    fake = FakeBackend()
    fake.term_payload = term_body(rows=[])
    data = (await call(fake, "find_videos", {"term": "retrieval augmented generation pipelines"})).structured_content
    assert data["videos"] == [] and data["total_matches"] == 0
    assert "shorter" in data["hint"]


async def test_locate_quote_finds_the_moment():
    fake = FakeBackend()
    phrase = "fused the two rankings with reciprocal rank fusion"
    data = (
        await call(fake, "locate_quote", {"video": f"https://www.youtube.com/watch?v={VIDEO_ID}", "phrase": phrase})
    ).structured_content
    start = expected_start(phrase)
    assert data["found"] is True
    assert data["start_sec"] == start
    assert data["youtube_url"] == f"https://www.youtube.com/watch?v={VIDEO_ID}&t={start}s"
    assert data["citation"] == (
        f"[Searching long videos, {format_timestamp(start)}]"
        f"(https://www.youtube.com/watch?v={VIDEO_ID}&t={start}s)"
    )
    assert json.loads(fake.requests[0].content) == {"video_id": VIDEO_ID, "query": phrase}


async def test_locate_quote_not_found_returns_a_hint():
    data = (
        await call(FakeBackend(), "locate_quote", {"video": VIDEO_ID, "phrase": "words that are never spoken here"})
    ).structured_content
    assert data["found"] is False and data["start_sec"] is None
    assert data["youtube_url"] == f"https://www.youtube.com/watch?v={VIDEO_ID}"
    assert data["citation"] == f"[Searching long videos](https://www.youtube.com/watch?v={VIDEO_ID})"
    assert "fewer" in data["hint"]


async def test_locate_quote_reports_unreadable_captions():
    fake = FakeBackend()
    fake.seek = lambda _phrase: (500, {"status": "error", "message": "Could not load captions from YouTube for this video"})
    result = await call(fake, "locate_quote", {"video": VIDEO_ID, "phrase": "anything at all"})
    assert result.is_error is True
    text = result.content[0].text
    assert "could not read the captions" in text and "Could not load captions" in text


async def test_missing_configuration_is_reported_to_the_agent(monkeypatch):
    monkeypatch.delenv("ATLANDEX_API_URL", raising=False)
    async with Client(create_server()) as client:
        result = await client.call_tool("find_videos", {"term": "rag"})
    assert result.is_error is True
    assert "ATLANDEX_API_URL is not set" in result.content[0].text
