"""
In-memory (with disk snapshot) session store for CourtAssist.

A "session" == one hearing. It holds:
 - context: extracted text chunks from uploaded case files (motions, briefs,
   exhibits, prior orders, statutes, etc.) so the model has the case's facts.
 - transcript: rolling list of {speaker, text, ts} entries from the live
   mic-transcription stream on the phone.
 - suggestions: every counter-argument/citation packet produced so far, each
   tagged with an incrementing index so the app can poll "give me everything
   after index N".
 - drafts: any document/email drafts generated, for later retrieval.

This is intentionally simple (no external DB) so the whole backend can run
on a single small VPS or even a laptop. Swap for Redis/Postgres if you need
multi-instance scaling.
"""
import json
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

DATA_DIR = os.environ.get("COURTASSIST_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
os.makedirs(DATA_DIR, exist_ok=True)

_lock = threading.RLock()
_sessions: Dict[str, Dict[str, Any]] = {}


def _path(session_id: str) -> str:
    return os.path.join(DATA_DIR, f"{session_id}.json")


def _save(session_id: str) -> None:
    with open(_path(session_id), "w") as f:
        json.dump(_sessions[session_id], f)


def create_session(title: str = "") -> str:
    session_id = uuid.uuid4().hex[:12]
    with _lock:
        _sessions[session_id] = {
            "id": session_id,
            "title": title or f"Hearing {time.strftime('%Y-%m-%d %H:%M')}",
            "created_at": time.time(),
            "context": [],       # list of {name, text, added_at}
            "transcript": [],    # list of {speaker, text, ts}
            "suggestions": [],   # list of {idx, ts, trigger_text, counter_argument, citations, status}
            "drafts": [],        # list of {id, kind, content, ts}
        }
        _save(session_id)
    return session_id


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        if session_id in _sessions:
            return _sessions[session_id]
        # try loading from disk (e.g. after a backend restart)
        p = _path(session_id)
        if os.path.exists(p):
            with open(p) as f:
                _sessions[session_id] = json.load(f)
            return _sessions[session_id]
    return None


def require_session(session_id: str) -> Dict[str, Any]:
    s = get_session(session_id)
    if s is None:
        raise KeyError(f"No such session: {session_id}")
    return s


def add_context(session_id: str, name: str, text: str) -> None:
    with _lock:
        s = require_session(session_id)
        s["context"].append({"name": name, "text": text, "added_at": time.time()})
        _save(session_id)


def list_context(session_id: str) -> List[Dict[str, Any]]:
    return require_session(session_id)["context"]


def context_blob(session_id: str, max_chars: int = 60000) -> str:
    """Concatenate uploaded case files into one context blob for prompting."""
    s = require_session(session_id)
    parts = []
    total = 0
    for item in s["context"]:
        chunk = f"\n\n=== FILE: {item['name']} ===\n{item['text']}"
        total += len(chunk)
        if total > max_chars:
            parts.append(chunk[: max_chars - (total - len(chunk))])
            break
        parts.append(chunk)
    return "".join(parts)


def add_transcript_line(session_id: str, speaker: str, text: str) -> Dict[str, Any]:
    with _lock:
        s = require_session(session_id)
        entry = {"speaker": speaker, "text": text, "ts": time.time()}
        s["transcript"].append(entry)
        _save(session_id)
        return entry


def recent_transcript(session_id: str, n: int = 40) -> List[Dict[str, Any]]:
    return require_session(session_id)["transcript"][-n:]


def add_suggestion(session_id: str, trigger_text: str, counter_argument: str,
                    citations: List[Dict[str, Any]], status: str = "ready") -> Dict[str, Any]:
    with _lock:
        s = require_session(session_id)
        idx = len(s["suggestions"])
        item = {
            "idx": idx,
            "ts": time.time(),
            "trigger_text": trigger_text,
            "counter_argument": counter_argument,
            "citations": citations,
            "status": status,
        }
        s["suggestions"].append(item)
        _save(session_id)
        return item


def suggestions_since(session_id: str, since: int = -1) -> List[Dict[str, Any]]:
    s = require_session(session_id)
    return [x for x in s["suggestions"] if x["idx"] > since]


def add_draft(session_id: str, kind: str, content: Dict[str, Any]) -> Dict[str, Any]:
    with _lock:
        s = require_session(session_id)
        draft = {"id": uuid.uuid4().hex[:8], "kind": kind, "content": content, "ts": time.time()}
        s["drafts"].append(draft)
        _save(session_id)
        return draft
