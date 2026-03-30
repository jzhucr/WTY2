#!/usr/bin/env python3
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import chat_runtime_openrouter as runtime


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_PATH = BASE_DIR / ".env"
DEFAULT_SESSION_STORE_PATH = BASE_DIR / "prepared_data" / "api_sessions.json"
DEFAULT_CHAT_PAGE_PATH = BASE_DIR / "web_chat.html"

load_dotenv(dotenv_path=DEFAULT_ENV_PATH, override=False)

app = FastAPI(title="Wang Tianyu Agent API", version="1.0.0")

_store_lock = threading.Lock()
_client = None


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    turns: int
    summary: str


class ResetSessionRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)


def _get_client():
    global _client
    if _client is None:
        _client = runtime._create_client()
    return _client


def _get_store_path() -> Path:
    return Path(os.getenv("API_SESSION_STORE_PATH", str(DEFAULT_SESSION_STORE_PATH)))


def _load_store() -> Dict[str, Dict[str, Any]]:
    path = _get_store_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    sessions = payload.get("sessions", {})
    if not isinstance(sessions, dict):
        return {}
    return sessions


def _save_store(sessions: Dict[str, Dict[str, Any]]) -> None:
    path = _get_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"sessions": sessions}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _get_or_create_session(sessions: Dict[str, Dict[str, Any]], session_id: str) -> Dict[str, Any]:
    if session_id not in sessions:
        sessions[session_id] = {"summary": "", "history": []}
    session = sessions[session_id]
    if not isinstance(session.get("summary", ""), str):
        session["summary"] = ""
    if not isinstance(session.get("history", []), list):
        session["history"] = []
    return session


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def chat_page() -> FileResponse:
    if not DEFAULT_CHAT_PAGE_PATH.exists():
        raise HTTPException(status_code=404, detail="chat page not found")
    return FileResponse(DEFAULT_CHAT_PAGE_PATH)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    compress_trigger_turns = int(os.getenv("MEMORY_COMPRESS_TRIGGER_TURNS", "8"))
    keep_recent_turns = int(os.getenv("MEMORY_KEEP_RECENT_TURNS", "4"))

    with _store_lock:
        sessions = _load_store()
        session = _get_or_create_session(sessions, req.session_id)
        history: List[Dict[str, str]] = session["history"]
        summary: str = session["summary"]

    try:
        reply = runtime.chat_once(
            client=_get_client(),
            user_message=req.message,
            history=history,
            session_summary=summary,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"chat failed: {exc}") from exc

    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": reply})
    summary, history = runtime._compact_history(
        summary=summary,
        history=history,
        compress_trigger_turns=compress_trigger_turns,
        keep_recent_turns=keep_recent_turns,
    )

    with _store_lock:
        sessions = _load_store()
        sessions[req.session_id] = {"summary": summary, "history": history}
        _save_store(sessions)

    return ChatResponse(
        session_id=req.session_id,
        reply=reply,
        turns=len(history) // 2,
        summary=summary,
    )


@app.post("/session/reset")
def reset_session(req: ResetSessionRequest) -> Dict[str, str]:
    with _store_lock:
        sessions = _load_store()
        sessions.pop(req.session_id, None)
        _save_store(sessions)
    return {"status": "ok", "session_id": req.session_id}


@app.get("/session/{session_id}")
def get_session(session_id: str) -> Dict[str, Any]:
    with _store_lock:
        sessions = _load_store()
        session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")
    return {"session_id": session_id, "summary": session.get("summary", ""), "history": session.get("history", [])}
