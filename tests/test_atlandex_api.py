import json

import httpx
import pytest

from atlandex_mcp.atlandex_api import (
    AtlandexAPI,
    BackendRejected,
    NotIndexed,
    RateLimited,
    Unavailable,
)

from fakes import API_URL, VIDEO_ID, FakeBackend

pytestmark = pytest.mark.anyio


async def test_query_asks_for_retrieval_only():
    fake = FakeBackend()
    api = fake.api()
    await api.query_video(VIDEO_ID, "what is rrf", 3)
    await api.aclose()

    (request,) = fake.requests
    assert request.method == "POST"
    assert str(request.url) == f"{API_URL}/yt_videos/{VIDEO_ID}/query"
    assert json.loads(request.content) == {"question": "what is rrf", "top_k": 3, "retrieve_only": True}
    assert request.headers["user-agent"].startswith("atlandex-mcp/")


async def test_term_lookup_encodes_the_term_and_scopes_by_channel():
    fake = FakeBackend()
    api = fake.api()
    await api.term_lookup("reciprocal rank fusion")
    await api.term_lookup("C++/CLI", channel_id="UC123")
    await api.aclose()

    assert fake.requests[0].url.raw_path == b"/api/term/reciprocal%20rank%20fusion"
    assert fake.requests[1].url.raw_path == b"/api/channel/UC123/term/C%2B%2B%2FCLI"


async def test_snippet_seek_payload():
    fake = FakeBackend()
    api = fake.api()
    await api.snippet_seek(VIDEO_ID, "a phrase")
    await api.aclose()
    assert json.loads(fake.requests[0].content) == {"video_id": VIDEO_ID, "query": "a phrase"}


@pytest.mark.parametrize(
    ("status", "body", "error", "message_part"),
    [
        (404, {"error": "No embeddings found for this video"}, NotIndexed, "not indexed"),
        (429, {"error": "limit"}, RateLimited, "rate limit"),
        (400, {"error": "Embedding dimensionality mismatch"}, BackendRejected, "dimensionality mismatch"),
        (500, {"error": "Database query failed: password authentication failed"}, Unavailable, "HTTP 500"),
        (502, "<html>bad gateway</html>", Unavailable, "HTTP 502"),
    ],
)
async def test_query_error_mapping(status, body, error, message_part):
    fake = FakeBackend()
    fake.query_status = status
    fake.query_payload = body
    api = fake.api()
    with pytest.raises(error) as excinfo:
        await api.query_video(VIDEO_ID, "q", 3)
    await api.aclose()
    assert message_part in str(excinfo.value)
    assert "password" not in str(excinfo.value)


async def test_seek_caption_failure_keeps_the_backend_message():
    fake = FakeBackend()
    fake.seek = lambda _phrase: (500, {"status": "error", "message": "Could not load captions from YouTube"})
    api = fake.api()
    with pytest.raises(BackendRejected, match="Could not load captions"):
        await api.snippet_seek(VIDEO_ID, "x")
    await api.aclose()


async def test_timeouts_and_connection_errors_are_unavailable():
    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    def refused(request):
        raise httpx.ConnectError("refused", request=request)

    for handler, message in ((timeout, "timed out"), (refused, "ConnectError")):
        api = AtlandexAPI(API_URL, transport=httpx.MockTransport(handler))
        with pytest.raises(Unavailable, match=message):
            await api.term_lookup("rag")
        await api.aclose()


async def test_non_object_json_is_unavailable():
    fake = FakeBackend()
    fake.term_payload = ["not", "an", "object"]
    api = fake.api()
    with pytest.raises(Unavailable, match="not a JSON object"):
        await api.term_lookup("rag")
    await api.aclose()


async def test_wrong_base_url_is_not_mistaken_for_an_unindexed_video():
    """A misconfigured URL hits a generic 404 page or the public site's HTML."""

    def html(status):
        return lambda request: httpx.Response(status, text="<html>page</html>")

    for status, error in ((404, BackendRejected), (200, Unavailable)):
        api = AtlandexAPI(API_URL, transport=httpx.MockTransport(html(status)))
        with pytest.raises(error, match="ATLANDEX_API_URL"):
            await api.query_video(VIDEO_ID, "q", 3)
        await api.aclose()
