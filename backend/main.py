"""
CourtAssist backend.

Run: uvicorn main:app --host 0.0.0.0 --port 8080

This is the server the Android app talks to. It never stores your
Perplexity API key -- the app sends it on every request via the
X-Perplexity-Key header, and it's only held in memory for the duration of
that one request.

IMPORTANT: this backend has NO built-in authentication. If you deploy it on
a public VPS, put it behind HTTPS + a shared bearer token (see README) so a
stranger can't hit your endpoints, upload into your session, or burn your
Perplexity API quota.
"""
from typing import List, Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import extraction
import perplexity
import store
import verify

app = FastAPI(title="CourtAssist Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def _key(x_perplexity_key: Optional[str]) -> str:
    if not x_perplexity_key:
        raise HTTPException(400, "Missing X-Perplexity-Key header. Add your Perplexity API key in the app's Settings screen.")
    return x_perplexity_key


# ---------- Sessions ----------

class CreateSessionBody(BaseModel):
    title: str = ""


@app.post("/sessions")
def create_session(body: CreateSessionBody):
    sid = store.create_session(body.title)
    return {"session_id": sid}


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    s = store.get_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    return {
        "id": s["id"], "title": s["title"], "created_at": s["created_at"],
        "context_files": [c["name"] for c in s["context"]],
        "transcript_lines": len(s["transcript"]),
        "suggestions": len(s["suggestions"]),
    }


# ---------- Case context (multi-file upload) ----------

@app.post("/sessions/{session_id}/context/file")
async def upload_context_file(session_id: str, file: UploadFile = File(...)):
    if not store.get_session(session_id):
        raise HTTPException(404, "Session not found")
    data = await file.read()
    text, note = extraction.extract_text(file.filename, data)
    if text:
        store.add_context(session_id, file.filename, text)
    return {"filename": file.filename, "chars_extracted": len(text), "note": note}


class TextContextBody(BaseModel):
    name: str
    text: str


@app.post("/sessions/{session_id}/context/text")
def upload_context_text(session_id: str, body: TextContextBody):
    if not store.get_session(session_id):
        raise HTTPException(404, "Session not found")
    store.add_context(session_id, body.name, body.text)
    return {"ok": True}


@app.get("/sessions/{session_id}/context")
def list_context(session_id: str):
    if not store.get_session(session_id):
        raise HTTPException(404, "Session not found")
    items = store.list_context(session_id)
    return [{"name": i["name"], "chars": len(i["text"]), "added_at": i["added_at"]} for i in items]


# ---------- Live transcript -> counter-argument + verified citations ----------

class TranscriptBody(BaseModel):
    speaker: str = "unknown"
    text: str


@app.post("/sessions/{session_id}/transcript")
async def post_transcript(session_id: str, body: TranscriptBody,
                           x_perplexity_key: Optional[str] = Header(None)):
    if not store.get_session(session_id):
        raise HTTPException(404, "Session not found")
    api_key = _key(x_perplexity_key)

    store.add_transcript_line(session_id, body.speaker, body.text)
    recent = store.recent_transcript(session_id, n=30)
    case_context = store.context_blob(session_id)

    try:
        analysis = await perplexity.analyze_transcript(
            api_key, case_context, recent[:-1], body.speaker, body.text,
        )
    except perplexity.PerplexityError as e:
        raise HTTPException(502, str(e))

    if not analysis.get("actionable"):
        return {"actionable": False, "trigger_text": analysis.get("trigger_text", body.text)}

    search_pool = (analysis.get("_model_search_results") or []) + [
        {"url": u, "title": ""} for u in (analysis.get("_model_citations") or [])
    ]
    raw_citations = analysis.get("citations") or []
    verified_citations = await verify.verify_all(api_key, raw_citations, search_pool)

    suggestion = store.add_suggestion(
        session_id,
        trigger_text=analysis.get("trigger_text", body.text),
        counter_argument=analysis.get("counter_argument", ""),
        citations=verified_citations,
    )
    return {"actionable": True, **suggestion}


@app.get("/sessions/{session_id}/suggestions")
def get_suggestions(session_id: str, since: int = -1):
    if not store.get_session(session_id):
        raise HTTPException(404, "Session not found")
    return store.suggestions_since(session_id, since)


# ---------- Drafting (documents & emails) ----------

class DraftDocBody(BaseModel):
    instruction: str


@app.post("/sessions/{session_id}/draft/document")
async def draft_document(session_id: str, body: DraftDocBody,
                          x_perplexity_key: Optional[str] = Header(None)):
    if not store.get_session(session_id):
        raise HTTPException(404, "Session not found")
    api_key = _key(x_perplexity_key)
    case_context = store.context_blob(session_id)
    try:
        content = await perplexity.draft_document(api_key, case_context, body.instruction)
    except perplexity.PerplexityError as e:
        raise HTTPException(502, str(e))
    draft = store.add_draft(session_id, "document", {"text": content, "instruction": body.instruction})
    return draft


class DraftEmailBody(BaseModel):
    instruction: str


@app.post("/sessions/{session_id}/draft/email")
async def draft_email(session_id: str, body: DraftEmailBody,
                       x_perplexity_key: Optional[str] = Header(None)):
    if not store.get_session(session_id):
        raise HTTPException(404, "Session not found")
    api_key = _key(x_perplexity_key)
    case_context = store.context_blob(session_id)
    try:
        content = await perplexity.draft_email(api_key, case_context, body.instruction)
    except perplexity.PerplexityError as e:
        raise HTTPException(502, str(e))
    draft = store.add_draft(session_id, "email", content)
    return draft
    # NOTE: sending is deliberately NOT done here. The Android app sends the
    # email itself via the Gmail API using the user's own signed-in account,
    # only after the user reviews this draft and taps Send. The backend never
    # has Gmail credentials and can never send on its own.


@app.get("/sessions/{session_id}/drafts")
def list_drafts(session_id: str):
    s = store.get_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    return s["drafts"]


@app.get("/healthz")
def healthz():
    return {"ok": True}
