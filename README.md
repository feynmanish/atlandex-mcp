# atlandex-mcp

An [MCP](https://modelcontextprotocol.io) server that lets AI agents search the long-form video [Atlandex](https://www.atlandex.app) has indexed (talks, podcasts, webinars) and cite the exact moment something is said. Ask Claude what a speaker said about a topic and every claim comes back with a link to that point in the video.

## Tools

| Tool | What it returns | Atlandex route |
|---|---|---|
| `find_videos(term, limit, channel_id)` | Indexed videos where a concept, person or technology comes up, with the chapter and time, and a `searchable` flag saying whether `search_video` can read the video | `GET /api/term/<term>`, then `GET /api/yt_videos/<id>/embeddings_ready` per video |
| `search_video(video, question, top_k)` | The passages of one video that answer a question, each with a timestamped link | `POST /api/yt_videos/<id>/query`, then `POST /api/snippet-seek` per passage |
| `locate_quote(video, phrase)` | When a phrase is spoken, as a timestamped link | `POST /api/snippet-seek` |

`video` accepts a YouTube URL, an Atlandex `/v/` link or a bare video ID. All three tools are read-only.

`searchable` is `true` when the video has transcript embeddings, `false` when it is in the term index but has no searchable transcript, and `null` when the check failed. For `false` videos the agent cites the chapter link or uses `locate_quote`.

## How a search works

1. **Dense retrieval (Atlandex).** The backend ranks the video's transcript chunks, up to 2,000 tokens each, by embedding distance. The request sets `retrieve_only`, so the backend embeds the question and skips generating its own answer.
2. **Passage selection (this server).** In each chunk, the server picks the 110-word window that covers the most question terms and starts it a few words before the first match.
3. **Timestamp lookup (Atlandex).** The window's opening words go to the caption seek, which returns where they are spoken. Seeks run concurrently, and one that fails leaves that passage without a timestamp instead of failing the search.

Each hit costs the agent about 1,000 characters of context instead of the 8,000 of a full chunk. Output from the test fixtures (synthetic transcript):

```json
{
  "video_id": "AbCdEfGhIjK",
  "title": "Searching long videos",
  "question": "Why did dense retrieval lose to TF-IDF?",
  "chunks_in_video": 14,
  "passages": [
    {
      "rank": 1,
      "text": "nine here is the surprise the dense retrieval run lost to the plain TF-IDF baseline on these questions and the reason was vocabulary the questions used product names and acronyms that the embedding model had rarely seen ...",
      "distance": 0.42,
      "start_sec": 657,
      "timestamp": "10:57",
      "located_by": "transcript",
      "youtube_url": "https://www.youtube.com/watch?v=AbCdEfGhIjK&t=657s",
      "atlandex_url": "https://www.atlandex.app/v/AbCdEfGhIjK?t=657"
    }
  ]
}
```

## Setup

Requires Python 3.10+.

```bash
git clone https://github.com/<you>/atlandex-mcp && cd atlandex-mcp
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Configuration comes from environment variables only:

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `ATLANDEX_API_URL` | yes | | Base URL of the Atlandex backend's `/api` routes |
| `ATLANDEX_SITE_URL` | no | `https://www.atlandex.app` | Site used for `/v/<id>?t=<sec>` links |
| `ATLANDEX_TIMEOUT_SEC` | no | `30` | HTTP timeout per backend call |

### Claude Code

```bash
claude mcp add atlandex -e ATLANDEX_API_URL=https://<backend-host>/api -- "$PWD/.venv/bin/atlandex-mcp"
claude mcp list    # atlandex: ... - ✓ Connected
```

Add `-s user` to make it available in every project. Inside a session, `/mcp` shows the server and its tools.

Any other MCP client works the same way: it launches `atlandex-mcp` and talks to it over stdio.

## Design decisions

- **Passages, not chunks.** Full chunks would put about 25,000 characters into context for three hits. One passage per chunk keeps a search near 3,000 characters and leaves room for the agent to compare videos.
- **Timestamps resolved in the server.** The agent gets citable links in one call instead of having to remember a second lookup per passage. The cost is one caption seek per passage, run in parallel.
- **Three narrow, read-only tools.** Each tool maps onto existing routes and carries `readOnlyHint` and `idempotentHint`, so clients can safely approve them without prompting. The server never calls routes that write, ingest or generate a completion.
- **Errors written for the model.** Anticipated failures say what to do next: a video with no searchable transcript says not to retry it and to cite the chapter link or use `locate_quote` (it does not point back to `find_videos`, which would list the same video again), a rate limit says to wait and use what is already retrieved, and a response that isn't the backend's JSON points to `ATLANDEX_API_URL`. When the query route answers 5xx, the server asks `embeddings_ready`: `false` gets the no-transcript message, anything else gets "retry once, then tell the user this video can't be searched right now". Only a timeout or connection failure is reported as the backend being unavailable. Backend 5xx bodies are not passed through, because they can contain database errors. The one exception is the caption seek's own error message.
- **No backend changes.** The server works against the API as deployed.
- **Backend URL from the environment only.** The caption-seek route is unmetered and triggers proxied caption fetches, so this public repo does not ship a default URL anyone could point at it.
- **stdio first.** Remote deployment over Streamable HTTP with authentication is the next step.
- **MCP Python SDK 2.x.** Built on `MCPServer` (renamed from `FastMCP` in 2.0) and pinned below 3.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

- **Unit tests:** video-ID parsing, passage selection and HTTP error mapping.
- **Protocol tests:** an in-process MCP client calls the real server against a fake backend, built from the actual routes' response shapes.
- **Entry-point tests:** the installed entry point is launched as a subprocess over stdio, once with the initialize handshake most clients use (protocol 2024-11-05 to 2025-11-25) and once negotiating 2026-07-28.

CI runs the suite on Python 3.10 and 3.13.

## Checking it against the live backend

Run ten real questions through Claude Code and look for:

1. **Tool choice.** For topic questions, the agent should start with `find_videos`. Given a video, it should go straight to `search_video`.
2. **Search terms.** Short terms find matches. Whole questions passed to `find_videos` return nothing.
3. **Citations.** Every claim should carry a `youtube_url`. When `start_sec` is null, the link should have no timestamp.
4. **Timestamp accuracy.** Open three links and note how far before the line each one lands (see limits below).
5. **Recovery.** Ask `search_video` about a video with no searchable transcript. The agent should not retry it, and should cite the chapter link or use `locate_quote`.
6. **Searchable flag.** `find_videos` results marked `searchable: false` should not be passed to `search_video`.

Fix what goes wrong in the tool descriptions first. They are what the model reads.

## Known limits

- **Search scope.** Semantic search works within one video. Discovery across videos matches extracted index terms exactly. Searching the whole corpus by meaning would need a new backend route.
- **Term index and transcript index differ.** `find_videos` matches the term index, which can include videos that have no transcript embeddings. They come back with `searchable: false`: `search_video` cannot read them, so you get the chapter link and `locate_quote`. `embeddings_ready` can report `true` for a video whose query route still answers 500 (seen with an ID Atlandex has not ingested), so `searchable: true` is not a guarantee: `search_video` can still fail on it with a retry-once message.
- **Timestamp precision.** The caption seek returns the start of the earliest 45-second caption window that contains the phrase, minus 10 seconds of lead-in. Links can therefore land up to about a minute before the line.
- **`locate_quote` false positives.** `locate_quote` can report `found: true` for phrases that are not in the captions. The backend seek accepts any 45-second window that contains two of the query's words, including words like "on" and "the", and the server cannot tell such a hit from a real match. Treat a `locate_quote` time as a pointer to check, and use it only for words you already know were said.
