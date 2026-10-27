"""Shared blocking HTTP layer for vLLM chat calls (run in executor threads).

Every call STREAMS the response (SSE) under a hard wall-clock deadline.
Why streaming even though callers want the whole body: with a plain POST,
a client-side timeout abandons the request but vLLM keeps generating it to
completion — a zombie request burning GPU that starves every other loop
(this exact failure produced 10-24s commit patches). With streaming, closing
the connection makes the server notice the disconnect and abort generation,
so a timeout actually cancels the work.

Two failure types, deliberately distinct:
  - ChatHTTPError (has .status): the server ANSWERED and rejected the request
    — evidence about the request shape (e.g. unsupported structured-output
    parameter).
  - ChatTimeout: deadline exceeded / connection problems — evidence about
    load or availability, NEVER about the request shape. Callers must not
    respond to this by re-sending the request in a different shape.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


class ChatTimeout(Exception):
    pass


class ChatHTTPError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(f"HTTP {status}: {detail[:200]}")
        self.status = status
        self.detail = detail


def stream_chat_blocking(base_url: str, body: dict, deadline_s: float) -> str:
    """POST a chat completion, streaming; returns the full message content.

    Raises ChatTimeout when deadline_s elapses (connection is closed first,
    which aborts the generation server-side) and ChatHTTPError on HTTP
    error responses.
    """
    body = dict(body)
    body["stream"] = True
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    try:
        resp = urllib.request.urlopen(req, timeout=deadline_s)
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(e)
        raise ChatHTTPError(e.code, detail) from None
    except (TimeoutError, OSError) as e:
        raise ChatTimeout(f"connect failed: {e}") from None

    parts: list[str] = []
    try:
        with resp:
            for raw in resp:
                if time.monotonic() - t0 > deadline_s:
                    raise ChatTimeout(
                        f"deadline {deadline_s:.0f}s exceeded mid-stream")
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                for choice in chunk.get("choices", []):
                    content = (choice.get("delta") or {}).get("content")
                    if content:
                        parts.append(content)
    except ChatTimeout:
        raise
    except (TimeoutError, OSError) as e:
        raise ChatTimeout(f"read stalled: {e}") from None
    return "".join(parts)
