"""
Thin wrapper around the Perplexity API (api.perplexity.ai), used for two
things:

 1. Reasoning + web/legal search to draft a counter-argument grounded in the
    user's uploaded case files.
 2. Independent re-search calls used purely to verify citations (see
    verify.py) -- kept separate from (1) so the verification pass is a truly
    independent lookup, not just trusting the first answer.

The caller must supply their own Perplexity API key (Settings screen in the
Android app -> sent as the X-Perplexity-Key header on every backend request,
never stored server-side). Get a key at https://www.perplexity.ai/settings/api
"""
import json
import re
from typing import Any, Dict, List, Optional

import httpx

# Perplexity retired the Sonar /chat/completions API in favor of the Agent
# API (POST /v1/agent, OpenAI-Responses-shaped request/response). See
# https://docs.perplexity.ai/docs/agent-api/migrate-from-sonar/overview
API_URL = "https://api.perplexity.ai/v1/agent"

# Presets replace the old model names. Each preset bundles a model, system
# prompt, and built-in web search -- the same "search-grounded answer with
# citations" behavior the old sonar/sonar-reasoning-pro models provided.
ANALYSIS_PRESET = "high"   # deeper reasoning w/ citations, replaces sonar-reasoning-pro
VERIFY_PRESET = "low"      # fast independent corroboration search, replaces sonar-pro
DRAFT_PRESET = "high"      # for document/email drafting, replaces sonar-reasoning-pro


class PerplexityError(RuntimeError):
    pass


def _parse_agent_response(resp: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten an Agent API ResponsesResponse into the shape the rest of this
    module expects: {content, citations, search_results}."""
    status = resp.get("status")
    if status not in (None, "completed"):
        err = resp.get("error") or {}
        raise PerplexityError(
            f"Perplexity Agent API response status={status}: {err.get('message', resp)}"[:500]
        )
    text_parts: List[str] = []
    search_results: List[Dict[str, Any]] = []
    citations: List[str] = []
    for item in resp.get("output") or []:
        item_type = item.get("type")
        if item_type == "message":
            for part in item.get("content") or []:
                if part.get("type") == "output_text" and part.get("text"):
                    text_parts.append(part["text"])
        elif item_type == "search_results":
            for r in item.get("results") or []:
                search_results.append({
                    "url": r.get("url", ""),
                    "title": r.get("title", ""),
                    "snippet": r.get("snippet", ""),
                })
                if r.get("url"):
                    citations.append(r["url"])
    return {"content": "".join(text_parts), "citations": citations, "search_results": search_results}


async def _chat(api_key: str, preset: str, system_prompt: str, user_content: str,
                 max_tokens: int = 1200) -> Dict[str, Any]:
    if not api_key:
        raise PerplexityError("Missing Perplexity API key. Add it in the app's Settings screen.")
    # Defensive: a stray newline/whitespace in the key (e.g. from an env var
    # pasted into a hosting dashboard) makes this an invalid HTTP header
    # value and httpx/httpcore raises LocalProtocolError on every request.
    headers = {"Authorization": f"Bearer {api_key.strip()}", "Content-Type": "application/json"}
    body = {
        "input": user_content,
        "instructions": system_prompt,
        "preset": preset,
        "tools": [{"type": "web_search"}],
        "max_output_tokens": max_tokens,
        "temperature": 0.2,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(API_URL, headers=headers, json=body)
    if r.status_code != 200:
        raise PerplexityError(f"Perplexity API error {r.status_code}: {r.text[:500]}")
    return _parse_agent_response(r.json())


def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """Model is instructed to answer in JSON; be defensive about stray prose/fences."""
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else None
    if candidate is None:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = brace.group(0) if brace else None
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


ANALYSIS_SYSTEM_PROMPT = """You are a real-time courtroom co-counsel assistant. \
You are given (a) the user's own uploaded case materials, (b) a rolling transcript \
of a live hearing, and (c) the newest line spoken. Your ONLY job right now is to \
decide whether the newest line is an argument, claim, or factual assertion made by \
the opposing side (or the judge raising a point) that the user could benefit from \
rebutting -- and if so, draft a concise, usable counter-argument with real, \
verifiable legal citations (case law, statutes, rules of procedure, or the user's \
own case documents).

Rules:
- If the newest line is small talk, procedural chatter, or nothing worth reacting \
to, say so plainly.
- NEVER invent a citation. Only cite sources you would stand behind if someone \
looked them up right now. If you are not certain a citation is real, omit it \
rather than guess.
- Prefer citing the user's own uploaded case materials when they are directly \
relevant, in addition to outside legal authority.
- Keep the counter-argument short enough to read in 10-15 seconds -- this is being \
read live, mid-hearing.

Respond with ONLY a JSON object, no other text, matching exactly:
{
  "actionable": true|false,
  "trigger_text": "<the newest line or the specific claim within it>",
  "counter_argument": "<short, usable rebuttal, or empty string if not actionable>",
  "citations": [
    {"label": "<short readable name, e.g. 'Fed. R. Civ. P. 56(a)' or case name>",
     "detail": "<what this source establishes, one sentence>",
     "source_hint": "<best URL or identifying string you have for this, may be empty>"}
  ]
}
"""


async def analyze_transcript(api_key: str, case_context: str, recent_lines: List[Dict[str, Any]],
                              newest_speaker: str, newest_text: str) -> Dict[str, Any]:
    transcript_str = "\n".join(f"{l.get('speaker', '?')}: {l['text']}" for l in recent_lines)
    user_prompt = (
        f"UPLOADED CASE MATERIALS (may be empty):\n{case_context or '(none uploaded yet)'}\n\n"
        f"RECENT TRANSCRIPT:\n{transcript_str}\n\n"
        f"NEWEST LINE ({newest_speaker}): {newest_text}\n\n"
        "Analyze the newest line per your instructions and respond with the JSON object only."
    )
    resp = await _chat(api_key, ANALYSIS_PRESET, ANALYSIS_SYSTEM_PROMPT, user_prompt)
    content = resp["content"]
    parsed = _extract_json_block(content)
    citations_meta = resp.get("citations") or []
    search_results = resp.get("search_results") or []
    if parsed is None:
        return {"actionable": False, "trigger_text": newest_text, "counter_argument": "",
                "citations": [], "_raw": content}
    parsed["_model_citations"] = citations_meta
    parsed["_model_search_results"] = search_results
    return parsed


DRAFT_SYSTEM_PROMPT = """You draft legal documents and correspondence for a \
self-represented litigant, grounded strictly in the case materials and instructions \
given to you. Never invent facts, dates, docket numbers, or citations that are not \
in the provided materials or independently verifiable. If information needed to \
complete the draft is missing, insert a clearly marked placeholder like \
[INSERT DOCKET NUMBER] rather than guessing."""


async def draft_document(api_key: str, case_context: str, instruction: str) -> str:
    resp = await _chat(
        api_key, DRAFT_PRESET, DRAFT_SYSTEM_PROMPT,
        f"CASE MATERIALS:\n{case_context}\n\nINSTRUCTION:\n{instruction}",
        max_tokens=2000,
    )
    return resp["content"]


EMAIL_SYSTEM_PROMPT = """You draft a single email on behalf of a self-represented \
litigant, grounded strictly in the case materials and instruction given. Output \
ONLY a JSON object: {"to": "...", "subject": "...", "body": "..."}. Leave "to" \
empty if no recipient is specified or inferable. Never invent a recipient email \
address."""


async def draft_email(api_key: str, case_context: str, instruction: str) -> Dict[str, str]:
    resp = await _chat(
        api_key, DRAFT_PRESET, EMAIL_SYSTEM_PROMPT,
        f"CASE MATERIALS:\n{case_context}\n\nINSTRUCTION:\n{instruction}",
        max_tokens=1200,
    )
    content = resp["content"]
    parsed = _extract_json_block(content)
    if parsed is None:
        return {"to": "", "subject": "Draft", "body": content}
    return parsed


ASK_SYSTEM_PROMPT = """You are a real-time courtroom co-counsel assistant answering a \
direct question typed by a self-represented litigant, either during hearing prep or \
mid-hearing. You are given (a) the user's own uploaded case materials and (b) a \
rolling transcript of the live hearing so far (may be empty). Answer the question \
directly and usefully, grounded in the case materials and real, verifiable legal \
authority (case law, statutes, rules of procedure) wherever relevant.

Rules:
- Always answer -- this was explicitly asked for, so never say "not actionable".
- NEVER invent a citation. Only cite sources you would stand behind if someone \
looked them up right now. If you are not certain a citation is real, omit it \
rather than guess.
- Prefer citing the user's own uploaded case materials when directly relevant, in \
addition to outside legal authority.
- Keep the answer concise enough to read in 15-20 seconds if this is mid-hearing.

Respond with ONLY a JSON object, no other text, matching exactly:
{
  "answer": "<direct, usable answer to the question>",
  "citations": [
    {"label": "<short readable name, e.g. 'Fed. R. Civ. P. 56(a)' or case name>",
     "detail": "<what this source establishes, one sentence>",
     "source_hint": "<best URL or identifying string you have for this, may be empty>"}
  ]
}
"""


async def answer_question(api_key: str, case_context: str, recent_lines: List[Dict[str, Any]],
                           question: str) -> Dict[str, Any]:
    transcript_str = "\n".join(f"{l.get('speaker', '?')}: {l['text']}" for l in recent_lines)
    user_prompt = (
        f"UPLOADED CASE MATERIALS (may be empty):\n{case_context or '(none uploaded yet)'}\n\n"
        f"RECENT TRANSCRIPT (may be empty):\n{transcript_str or '(none yet)'}\n\n"
        f"QUESTION FROM USER: {question}\n\n"
        "Answer per your instructions and respond with the JSON object only."
    )
    resp = await _chat(api_key, ANALYSIS_PRESET, ASK_SYSTEM_PROMPT, user_prompt)
    content = resp["content"]
    parsed = _extract_json_block(content)
    citations_meta = resp.get("citations") or []
    search_results = resp.get("search_results") or []
    if parsed is None:
        return {"answer": content, "citations": [], "_model_citations": citations_meta,
                "_model_search_results": search_results}
    parsed["_model_citations"] = citations_meta
    parsed["_model_search_results"] = search_results
    return parsed


async def independent_search(api_key: str, query: str) -> Dict[str, Any]:
    """A deliberately separate, narrow search call used only for verification."""
    resp = await _chat(
        api_key, VERIFY_PRESET, "Answer factually and briefly. Cite exact sources.", query,
        max_tokens=500,
    )
    return {"content": resp["content"], "citations": resp.get("citations") or [],
            "search_results": resp.get("search_results") or []}
