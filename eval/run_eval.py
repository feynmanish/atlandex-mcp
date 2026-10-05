"""Run the live evaluation questions through `claude -p` and summarise each run.

    python eval/run_eval.py --model sonnet --round r1 [--ids 1 2 3]

Each question runs in a fresh process from an empty folder so the agent cannot read this repo.
Raw stream-json goes to eval/runs/ (git-ignored); a one-page summary is printed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
TOOLS = "mcp__atlandex__find_videos,mcp__atlandex__search_video,mcp__atlandex__locate_quote"


def run_one(q: dict, model: str, round_: str) -> Path:
    workdir = Path(tempfile.gettempdir()) / "atlandex-eval"
    workdir.mkdir(exist_ok=True)
    out = RUNS / f"q{q['id']:02d}-{model}-{round_}.jsonl"
    cmd = [
        shutil.which("claude") or "claude", "-p", q["q"], "--model", model,
        "--output-format", "stream-json", "--verbose",
        "--allowedTools", TOOLS, "--disallowedTools", "Bash,WebSearch,WebFetch",
    ]
    with out.open("w", encoding="utf-8") as f:
        try:
            subprocess.run(cmd, cwd=workdir, stdout=f, stderr=subprocess.STDOUT, text=True,
                           encoding="utf-8", stdin=subprocess.DEVNULL, timeout=300)
        except subprocess.TimeoutExpired:
            f.write(json.dumps({"type": "result", "result": "RUN TIMED OUT after 300 s"}) + "\n")
    return out


def parse(path: Path) -> dict:
    calls, results, final = [], {}, ""
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        msg = ev.get("message") or {}
        content = msg.get("content")
        if ev.get("type") == "assistant" and isinstance(content, list):
            for b in content:
                if b.get("type") == "tool_use":
                    calls.append({"id": b["id"], "tool": b["name"].replace("mcp__atlandex__", ""),
                                  "args": b["input"]})
        if ev.get("type") == "user" and isinstance(content, list):
            for b in content:
                if b.get("type") == "tool_result":
                    c = b.get("content")
                    if isinstance(c, list):
                        c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
                    results[b["tool_use_id"]] = c or ""
        if ev.get("type") == "result":
            final = ev.get("result", "")
    return {"calls": calls, "results": results, "final": final}


def summarise(path: Path) -> str:
    p = parse(path)
    searchable: dict[str, object] = {}
    cited_ok: set[str] = set()
    null_start: set[str] = set()
    lines = []
    for c in p["calls"]:
        res = p["results"].get(c["id"], "")
        lines.append(f"  {c['tool']}({json.dumps(c['args'], ensure_ascii=False)})")
        lines.append(f"      -> {res[:160]!r}")
        if c["tool"] == "find_videos":
            try:
                for v in json.loads(res).get("videos", []):
                    searchable[v["video_id"]] = v.get("searchable")
            except ValueError:
                pass
        cited_ok.update(re.findall(r"https://www\.youtube\.com/watch\?v=[\w-]{11}(?:&t=\d+s)?", res))
    flags = []
    seen: dict[str, int] = {}
    for c in p["calls"]:
        key = c["tool"] + json.dumps(c["args"], sort_keys=True)
        seen[key] = seen.get(key, 0) + 1
    if any(n > 2 for n in seen.values()):
        flags.append("LOOP")
    for c in p["calls"]:
        if c["tool"] == "search_video":
            vid = re.search(r"[\w-]{11}", re.sub(r".*v=", "", c["args"]["video"]) or "")
            if vid and searchable.get(vid.group(0)) is False:
                flags.append(f"search_video-on-unsearchable:{vid.group(0)}")
        if c["tool"] == "find_videos" and len(c["args"].get("term", "").split()) > 4:
            flags.append("long-term")
    answer_links = re.findall(r"https://www\.youtube\.com/watch\?v=[\w-]{11}(?:&t=\d+s)?", p["final"])
    invented = [u for u in set(answer_links) if u not in cited_ok]
    if invented:
        flags.append(f"links-not-from-tools:{invented}")
    return (f"## {path.name}  calls={len(p['calls'])} flags={flags or 'none'}\n"
            + "\n".join(lines) + f"\n  ANSWER:\n{p['final']}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--round", default="r1")
    ap.add_argument("--ids", type=int, nargs="*")
    args = ap.parse_args()
    RUNS.mkdir(exist_ok=True)
    qs = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    for q in qs:
        if args.ids and q["id"] not in args.ids:
            continue
        path = run_one(q, args.model, args.round)
        print(f"Q{q['id']} [{q['kind']}] {q['q']}")
        print(summarise(path), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
