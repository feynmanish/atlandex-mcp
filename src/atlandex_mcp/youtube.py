"""YouTube video-ID parsing and timestamped link building."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
# Path segments followed by the video ID: youtube.com/embed|shorts|live/<id>,
# and Atlandex's own video pages, atlandex.app/v/<id>.
_ID_PATH_MARKERS = ("embed", "shorts", "live", "v")


def extract_video_id(raw: str | None) -> str | None:
    """Return the 11-character video ID from a bare ID, a YouTube URL or an Atlandex /v/ link.

    Mirrors extract_youtube_video_id in the Atlandex backend, plus /v/<id> links.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if _ID_RE.match(s):
        return s

    with_scheme = s if re.match(r"^https?://", s, re.IGNORECASE) else f"https://{s}"
    try:
        url = urlparse(with_scheme)
    except ValueError:
        return None

    candidates = parse_qs(url.query).get("v") or []
    host = (url.hostname or "").removeprefix("www.")
    parts = [p for p in url.path.split("/") if p]
    if host == "youtu.be" and parts:
        candidates.append(parts[0])
    for marker in _ID_PATH_MARKERS:
        if marker in parts:
            i = parts.index(marker)
            if i + 1 < len(parts):
                candidates.append(parts[i + 1])

    for candidate in candidates:
        if _ID_RE.match(candidate):
            return candidate
    return None


def format_timestamp(seconds: int) -> str:
    """65 -> '1:05', 3725 -> '1:02:05'."""
    hours, rest = divmod(max(0, int(seconds)), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def youtube_url(video_id: str, start_sec: int | None = None) -> str:
    base = f"https://www.youtube.com/watch?v={video_id}"
    return base if start_sec is None else f"{base}&t={int(start_sec)}s"


def atlandex_url(site_url: str, video_id: str, start_sec: int | None = None) -> str:
    base = f"{site_url.rstrip('/')}/v/{video_id}"
    return base if start_sec is None else f"{base}?t={int(start_sec)}"
