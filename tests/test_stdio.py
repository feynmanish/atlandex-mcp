"""Launch the real entry point as a subprocess and talk to it over stdio, the way
Claude Code and other MCP clients run it. No network beyond localhost."""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters

from atlandex_mcp.passages import select_passage

from fakes import CHUNK_EVAL, VIDEO_ID, default_seek, expected_start, query_body, term_body

pytestmark = pytest.mark.anyio

QUESTION = "Why did dense retrieval lose to TF-IDF?"
_PROXY_VARS = {"http_proxy", "https_proxy", "all_proxy"}


def _server_params(api_url: str) -> StdioServerParameters:
    env = {k: v for k, v in os.environ.items() if k.lower() not in _PROXY_VARS}
    env |= {"ATLANDEX_API_URL": api_url, "ATLANDEX_TIMEOUT_SEC": "5", "NO_PROXY": "*"}
    return StdioServerParameters(command=sys.executable, args=["-m", "atlandex_mcp"], env=env)


class _BackendHandler(BaseHTTPRequestHandler):
    """Serves the fake backend's payloads over real HTTP."""

    def _send(self, status: int, payload) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        data = json.loads(self.rfile.read(length) or b"{}")
        if self.path == f"/api/yt_videos/{VIDEO_ID}/query":
            self._send(200, query_body())
        elif self.path == "/api/snippet-seek":
            self._send(*default_seek(data["query"]))
        else:
            self._send(404, {"error": "unknown route"})

    def do_GET(self) -> None:
        if self.path.startswith("/api/term/"):
            self._send(200, term_body())
        else:
            self._send(404, {"error": "unknown route"})

    def log_message(self, *args) -> None:
        pass


@pytest.fixture
def local_backend():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _BackendHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/api"
    httpd.shutdown()
    httpd.server_close()


# "legacy" runs the initialize handshake (protocol 2024-11-05 to 2025-11-25) that most
# MCP clients in use speak; "auto" negotiates the newest version both sides support.
@pytest.mark.parametrize("mode", ["legacy", "auto"])
async def test_entry_point_serves_tools_and_reports_errors(mode):
    # Port 9 on localhost refuses connections, which exercises the backend-down path.
    async with Client(_server_params("http://127.0.0.1:9/api"), mode=mode) as client:
        assert client.server_info.name == "atlandex"
        names = {t.name for t in (await client.list_tools()).tools}
        assert names == {"find_videos", "search_video", "locate_quote"}

        invalid = await client.call_tool("search_video", {"video": "nope", "question": "q"})
        assert invalid.is_error is True
        assert "not a YouTube URL" in invalid.content[0].text

        down = await client.call_tool("find_videos", {"term": "rag"})
        assert down.is_error is True
        assert "did not respond properly" in down.content[0].text


async def test_full_search_over_stdio_and_real_http(local_backend):
    async with Client(_server_params(local_backend), mode="legacy") as client:
        search = await client.call_tool("search_video", {"video": VIDEO_ID, "question": QUESTION, "top_k": 2})
        found = await client.call_tool("find_videos", {"term": "reciprocal rank fusion"})

    assert search.is_error is False
    first = search.structured_content["passages"][0]
    assert first["start_sec"] == expected_start(select_passage(CHUNK_EVAL, QUESTION).anchor)
    assert "TF-IDF baseline" in first["text"]

    assert found.is_error is False
    assert found.structured_content["total_matches"] == 3
