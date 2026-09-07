"""
app.py
======
FastAPI Server for Multilingual Voice Assistant with RAG & Nemotron 3.5 Flash Free.
Provides endpoints for Chat, Document RAG Ingestion, Audio STT/TTS, and System Status.

Fixes applied:
  - [Fix 1]  Removed circular import from chatbot.py; imports speech_config + whisper_loader instead.
  - [Fix 4]  Audio cache TTL cleanup (background task, max_age = 10 min).
  - [Fix 6]  CORS origins read from CORS_ORIGINS env var (comma-separated), default "*".
  - [Fix 7]  PACS write-operations protected by PACS_OPERATOR_KEY header check.
  - [Fix 11] STT temp file cleaned in finally block (no more file leaks on errors).
  - [Fix 12] Upload temp file cleaned in finally block (no more file leaks on errors).
"""

import sys
import os

# Force UTF-8 output on Windows (cp1252 terminals crash on emoji characters)
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import time
import tempfile
import asyncio
import uuid
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends, Header
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import edge_tts
from dotenv import load_dotenv

load_dotenv()

# ── shared stateless helpers (no heavy models loaded) ──────────────────────────
from speech_config import (
    determine_language,
    VOICE_MAP,
    LANG_NAMES,
    BASE_SYSTEM_PROMPT,
)

# ── lazy Whisper loader (GPU model loaded on first STT request) ────────────────
from whisper_loader import get_whisper_model

# ── service singletons ─────────────────────────────────────────────────────────
from llm_service import llm_service
from rag_service import rag_service
from db import db_service

# ──────────────────────────────────────────────────────────────────────────────
# APP SETUP
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Multilingual AI Assistant with RAG",
    description="Powered by Nemotron 3.5 Flash Free (OpenCode Zen) and Hybrid RAG",
    version="2.1.0",
)

# [Fix 6] CORS — read from env, default to "*" for dev convenience
_raw_origins = os.environ.get("CORS_ORIGINS", "").strip()
CORS_ORIGINS: List[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()] or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static assets & audio cache dirs
AUDIO_CACHE_DIR = os.path.join(os.path.dirname(__file__), "audio_cache")
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/audio", StaticFiles(directory=AUDIO_CACHE_DIR), name="audio")


# ──────────────────────────────────────────────────────────────────────────────
# [Fix 7] PACS OPERATOR AUTH
# ──────────────────────────────────────────────────────────────────────────────

_PACS_OPERATOR_KEY = os.environ.get("PACS_OPERATOR_KEY", "").strip()


def verify_pacs_key(x_pacs_key: Optional[str] = Header(default=None)):
    """
    FastAPI dependency that checks the X-PACS-Key header for write operations.
    If PACS_OPERATOR_KEY is not configured in the environment, auth is skipped
    (developer / local-only mode).
    """
    if not _PACS_OPERATOR_KEY:
        # No key configured → open access (local dev mode)
        return
    if x_pacs_key != _PACS_OPERATOR_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing PACS operator key. Set X-PACS-Key header.",
        )


# ──────────────────────────────────────────────────────────────────────────────
# PYDANTIC MODELS
# ──────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    language: Optional[str] = None
    chat_history: Optional[List[Dict[str, str]]] = []


class TTSRequest(BaseModel):
    text: str
    language: Optional[str] = "en"


class GrievanceStatusUpdate(BaseModel):
    status: str
    pacs_remarks: Optional[str] = ""


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

async def generate_speech_audio(text: str, lang_code: str) -> str:
    """Generate TTS MP3 with edge-tts and return the audio URL path."""
    voice = VOICE_MAP.get(lang_code, VOICE_MAP.get("en"))
    filename = f"tts_{uuid.uuid4().hex[:10]}.mp3"
    file_path = os.path.join(AUDIO_CACHE_DIR, filename)
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(file_path)
        return f"/audio/{filename}"
    except Exception as e:
        print(f"⚠️ TTS generation failed: {e}")
        return ""


# [Fix 4] Audio cache cleanup — delete MP3s older than max_age_seconds
def cleanup_old_audio_files(max_age_seconds: int = 600):
    """Remove audio cache files older than max_age_seconds (default 10 min)."""
    now = time.time()
    try:
        for fname in os.listdir(AUDIO_CACHE_DIR):
            if not fname.endswith(".mp3"):
                continue
            fpath = os.path.join(AUDIO_CACHE_DIR, fname)
            try:
                if now - os.path.getmtime(fpath) > max_age_seconds:
                    os.remove(fpath)
            except OSError:
                pass  # file may have already been removed
    except Exception as e:
        print(f"⚠️ Audio cache cleanup error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def get_index():
    """Serve main web user interface."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Multilingual RAG Assistant API is running."}


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest, background_tasks: BackgroundTasks):
    """
    Main chat route: processes user message, runs LLM tool loop (RAG / Web Search),
    and generates TTS audio response.
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty query string.")

    user_text = req.message.strip()

    # Determine response language
    target_lang = req.language
    if not target_lang or target_lang == "auto":
        target_lang = determine_language(user_text, whisper_lang="en")

    lang_instruction = f"\nUser language context: Respond in {LANG_NAMES.get(target_lang, 'English')}."

    # Build chat history context
    history = req.chat_history or []
    if not history or history[0].get("role") != "system":
        history.insert(0, {"role": "system", "content": BASE_SYSTEM_PROMPT})

    # Execute LLM Agent loop
    reply = llm_service.generate_response(
        user_text=user_text,
        base_system_prompt=BASE_SYSTEM_PROMPT,
        language_instruction=lang_instruction,
        chat_history=history,
    )

    # Generate speech output
    audio_url = await generate_speech_audio(reply, target_lang)

    # [Fix 4] Schedule audio cache purge in background
    background_tasks.add_task(cleanup_old_audio_files, 600)

    return {
        "reply": reply,
        "language": target_lang,
        "language_name": LANG_NAMES.get(target_lang, "English"),
        "audio_url": audio_url,
        "model": os.environ.get("OPENCODE_MODEL", "nemotron-3-ultra-free"),
    }


@app.post("/api/stt")
async def speech_to_text(file: UploadFile = File(...)):
    """Transcribe uploaded audio clip using Whisper (loaded lazily on first call)."""
    tmp_path = None
    try:
        suffix = os.path.splitext(file.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # [Fix 1] Whisper loaded lazily — not at server startup
        whisper_model = get_whisper_model()
        segments, info = whisper_model.transcribe(tmp_path, beam_size=5)
        text = " ".join([seg.text for seg in segments]).strip()
        detected_lang = info.language

        resolved_lang = determine_language(text, whisper_lang=detected_lang)

        return {
            "text": text,
            "detected_language": detected_lang,
            "resolved_language": resolved_lang,
            "language_name": LANG_NAMES.get(resolved_lang, "English"),
        }
    except Exception as e:
        print(f"⚠️ STT failed: {e}")
        raise HTTPException(status_code=500, detail=f"Speech recognition error: {str(e)}")
    finally:
        # [Fix 11] Always remove temp file, even on error
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload document / image into RAG Knowledge Base with OCR."""
    tmp_path = None
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        res = rag_service.add_document(tmp_path, custom_name=file.filename)

        if res.get("status") == "error":
            raise HTTPException(status_code=400, detail=res.get("message"))

        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document upload processing failed: {str(e)}")
    finally:
        # [Fix 12] Always remove temp file, even on error
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@app.get("/api/documents")
async def list_documents():
    """List indexed documents in RAG knowledge base."""
    docs = rag_service.list_documents()
    return {"documents": docs, "total_chunks": len(rag_service.documents)}


@app.delete("/api/documents/{filename}")
async def delete_document(
    filename: str,
    _: None = Depends(verify_pacs_key),  # [Fix 7] protected
):
    """Delete a document from RAG index."""
    success = rag_service.delete_document(filename)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"status": "success", "message": f"Document '{filename}' deleted."}


@app.delete("/api/documents")
async def clear_all_documents(
    _: None = Depends(verify_pacs_key),  # [Fix 7] protected
):
    """Clear entire vector index."""
    rag_service.clear_all()
    return {"status": "success", "message": "Knowledge Base cleared."}


@app.get("/pacs")
async def get_pacs_portal():
    """Serve PACS Operator Dashboard interface."""
    pacs_path = os.path.join(STATIC_DIR, "pacs.html")
    if os.path.exists(pacs_path):
        return FileResponse(pacs_path)
    return {"message": "PACS Operator Portal is ready."}


# ──────────────────────────────────────────────────────────────────────────────
# GRIEVANCE & PACS ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/api/grievances")
async def list_grievances(
    pacs_centre: Optional[str] = None,
    department: Optional[str] = None,
    status: Optional[str] = None,
):
    """Retrieve citizen grievances for PACS operator dashboard."""
    records = db_service.get_grievances(
        pacs_centre=pacs_centre, department=department, status=status
    )
    return {"grievances": records, "total": len(records)}


@app.get("/api/grievances/stats")
async def get_grievance_stats():
    """Return dashboard analytics metrics."""
    return db_service.get_pacs_stats()


@app.get("/api/grievances/{tracking_id}")
async def get_grievance_detail(tracking_id: str):
    """Fetch single grievance details."""
    record = db_service.get_grievance_by_tracking_id(tracking_id)
    if not record:
        raise HTTPException(status_code=404, detail="Grievance ticket not found.")
    return record


@app.patch("/api/grievances/{tracking_id}")
async def update_grievance(
    tracking_id: str,
    req: GrievanceStatusUpdate,
    _: None = Depends(verify_pacs_key),  # [Fix 7] protected write
):
    """Update grievance resolution status and PACS remarks."""
    updated = db_service.update_grievance_status(
        tracking_id, req.status, req.pacs_remarks or ""
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Grievance ticket not found.")
    return {"status": "success", "message": f"Grievance '{tracking_id}' updated to {req.status}."}


@app.get("/api/status")
async def get_system_status():
    """System health check and loaded configurations."""
    return {
        "status": "online",
        "model": os.environ.get("OPENCODE_MODEL", "nemotron-3-ultra-free"),
        "rag_documents": len(rag_service.list_documents()),
        "rag_chunks": len(rag_service.documents),
        "grievance_stats": db_service.get_pacs_stats(),
        "serper_active": bool(os.environ.get("SERPER_API_KEY")),
        "opencode_active": bool(os.environ.get("OPENCODE_API_KEY")),
        "cors_origins": CORS_ORIGINS,
        "pacs_auth_enabled": bool(_PACS_OPERATOR_KEY),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
