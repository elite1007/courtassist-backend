# CourtAssist backend

FastAPI service the Android app talks to. See `/docs/SETUP.md` at the
project root for the full walkthrough.

Quick start:

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8080
```

Key endpoints (all under the base URL you configure in the app):

- `POST /sessions` -- create a hearing session -> `{ "session_id": "..." }`
- `POST /sessions/{id}/context/file` -- multipart file upload (PDF/DOCX/TXT/MD/CSV)
- `POST /sessions/{id}/context/text` -- add raw text context `{name, text}`
- `GET  /sessions/{id}/context` -- list uploaded context files
- `POST /sessions/{id}/transcript` -- `{speaker, text}` (header `X-Perplexity-Key` required)
  -> `{actionable, trigger_text, counter_argument, citations: [...]}` where
  each citation has `verified`, `verified_url`, `verification_reason`.
- `GET  /sessions/{id}/suggestions?since=<idx>` -- poll for new suggestions
- `POST /sessions/{id}/draft/document` -- `{instruction}` -> drafted text
- `POST /sessions/{id}/draft/email` -- `{instruction}` -> `{to, subject, body}`
  (drafting only -- the app sends via Gmail directly after you approve)

The `X-Perplexity-Key` header carries your own Perplexity API key on every
AI-backed request; it is read once per request and never written to disk.
Session data (context text, transcript, suggestions, drafts) is persisted
as JSON under `backend/data/` so a restart doesn't lose an in-progress
hearing -- delete that folder if you want a clean slate.
