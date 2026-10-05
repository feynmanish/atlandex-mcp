import pytest

from atlandex_mcp.youtube import atlandex_url, extract_video_id, format_timestamp, youtube_url

VID = "AbCdEfGhIjK"


@pytest.mark.parametrize(
    "raw",
    [
        VID,
        f"  {VID}  ",
        f"https://www.youtube.com/watch?v={VID}",
        f"https://www.youtube.com/watch?list=PL123&v={VID}&t=42s",
        f"youtube.com/watch?v={VID}",
        f"https://m.youtube.com/watch?v={VID}",
        f"https://youtu.be/{VID}?si=abc",
        f"https://www.youtube.com/embed/{VID}",
        f"https://www.youtube.com/shorts/{VID}",
        f"https://www.youtube.com/live/{VID}?feature=share",
        f"https://www.atlandex.app/v/{VID}?t=754",
    ],
)
def test_extracts_id(raw):
    assert extract_video_id(raw) == VID


@pytest.mark.parametrize(
    "raw",
    [None, "", "not a video", "https://www.youtube.com/watch?v=short", "https://example.com/page", "AbCdEfGhIj"],
)
def test_rejects_non_videos(raw):
    assert extract_video_id(raw) is None


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "0:00"), (65, "1:05"), (754, "12:34"), (3725, "1:02:05"), (-5, "0:00")],
)
def test_format_timestamp(seconds, expected):
    assert format_timestamp(seconds) == expected


def test_links_with_and_without_time():
    assert youtube_url(VID) == f"https://www.youtube.com/watch?v={VID}"
    assert youtube_url(VID, 754) == f"https://www.youtube.com/watch?v={VID}&t=754s"
    assert atlandex_url("https://www.atlandex.app/", VID) == f"https://www.atlandex.app/v/{VID}"
    assert atlandex_url("https://www.atlandex.app", VID, 754) == f"https://www.atlandex.app/v/{VID}?t=754"
