"""Structured tool results. Field descriptions end up in each tool's output schema."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class VideoHit(BaseModel):
    video_id: str
    title: str
    channel: str | None = None
    relevance: int | None = Field(
        default=None,
        description="How central the term is to the video, 1-1000, as scored at indexing time.",
    )
    chapter_title: str | None = Field(default=None, description="Chapter where the term comes up.")
    start_sec: int | None = Field(default=None, description="Start of that chapter, in seconds.")
    timestamp: str | None = Field(default=None, description="start_sec as m:ss or h:mm:ss.")
    youtube_url: str = Field(description="Link to the video, at start_sec when known.")
    atlandex_url: str = Field(description="Atlandex page for the video, at start_sec when known.")


class FindVideosResult(BaseModel):
    term: str
    total_matches: int = Field(description="Matching videos found (Atlandex returns at most 200).")
    videos: list[VideoHit] = Field(description="Best matches first, cut to the requested limit.")
    hint: str | None = Field(default=None, description="What to try next when nothing matched.")


class Passage(BaseModel):
    rank: int = Field(description="1 is the closest match.")
    text: str = Field(description="Excerpt from the transcript (speech-to-text, lightly punctuated).")
    distance: float | None = Field(
        default=None,
        description="L2 distance between question and chunk embeddings; lower is closer. "
        "Comparable only within one search.",
    )
    start_sec: int | None = Field(default=None, description="Where the excerpt starts; null if not located.")
    timestamp: str | None = Field(default=None, description="start_sec as m:ss or h:mm:ss.")
    located_by: Literal["transcript", "creator_chapter"] | None = Field(
        default=None, description="How start_sec was found; null when it could not be."
    )
    youtube_url: str = Field(description="Cite this. Opens the video at start_sec when known.")
    atlandex_url: str = Field(description="Atlandex page for the video, at start_sec when known.")


class SearchVideoResult(BaseModel):
    video_id: str
    title: str | None = None
    channel: str | None = None
    question: str
    chunks_in_video: int = Field(description="Transcript chunks indexed for this video.")
    passages: list[Passage]
    note: str | None = None


class Moment(BaseModel):
    video_id: str
    found: bool = Field(description="False when the phrase did not match the captions.")
    start_sec: int | None = None
    timestamp: str | None = Field(default=None, description="start_sec as m:ss or h:mm:ss.")
    located_by: Literal["transcript", "creator_chapter"] | None = None
    title: str | None = None
    channel: str | None = None
    youtube_url: str
    atlandex_url: str
    hint: str | None = None
