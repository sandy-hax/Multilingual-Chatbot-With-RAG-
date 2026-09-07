"""
app.py
======
FastAPI Server for Multilingual Voice Assistant with RAG & Nemotron 3.5 Flash Free.
Provides endpoints for Chat, Document RAG Ingestion, Audio STT/TTS, and System Status.
"""

import os
import tempfile
import asyncio
import uuid
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import edge_tts
from llm_service import llm_service, BASE_SYSTEM_PROMPT
from rag_service import rag_service
from chatbot import determine_language, VOICE_MAP, LANG_NAMES, whisper_model

app = FastAPI(
    title="Multilingual AI Assistant with RAG",
    description="Powered by Nemotron 3.5 Flash Free (OpenCode Zen) and Hybrid RAG",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static audio export dir
AUDIO_CACHE_DIR = os.path.join(os.path.dirname(__file__), "audio_cache")
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

# Mount Static assets
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/audio", StaticFiles(directory=AUDIO_CACHE_DIR), name="audio")


# ============================================================
# MODELS
# ============================================================

class ChatRequest(BaseModel):
    message: str
    language: Optional[str] = None
    chat_history: Optional[List[Dict[str, str]]] = []

class TTSRequest(BaseModel):
    text: str
    language: Optional[str] = "en"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

async def generate_speech_audio(text: str, lang_code: str) -> str:
    """Generate audio MP3 file from text using edge-tts and return audio filename."""
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


# ============================================================
# ROUTE ENDPOINTS
# ============================================================

@app.get("/")
async def get_index():
    """Serve main web user interface."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Multilingual RAG Assistant API is running."}


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
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
        chat_history=history
    )

    # Generate speech output asynchronously
    audio_url = await generate_speech_audio(reply, target_lang)

    return {
        "reply": reply,
        "language": target_lang,
        "language_name": LANG_NAMES.get(target_lang, "English"),
        "audio_url": audio_url,
        "model": os.environ.get("OPENCODE_MODEL", "nemotron-3.5-flash-free")
    }


@app.post("/api/stt")
async def speech_to_text(file: UploadFile = File(...)):
    """Transcribe uploaded audio clip using Whisper."""
    try:
        suffix = os.path.splitext(file.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        segments, info = whisper_model.transcribe(tmp_path, beam_size=5)
        text = " ".join([seg.text for seg in segments]).strip()
        detected_lang = info.language

        os.remove(tmp_path)

        resolved_lang = determine_language(text, whisper_lang=detected_lang)

        return {
            "text": text,
            "detected_language": detected_lang,
            "resolved_language": resolved_lang,
            "language_name": LANG_NAMES.get(resolved_lang, "English")
        }
    except Exception as e:
        print(f"⚠️ STT failed: {e}")
        raise HTTPException(status_code=500, detail=f"Speech recognition error: {str(e)}")


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload document / image into RAG Knowledge Base with OCR."""
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        res = rag_service.add_document(tmp_path, custom_name=file.filename)
        os.remove(tmp_path)

        if res.get("status") == "error":
            raise HTTPException(status_code=400, detail=res.get("message"))

        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document upload processing failed: {str(e)}")


@app.get("/api/documents")
async def list_documents():
    """List indexed documents in RAG knowledge base."""
    docs = rag_service.list_documents()
    return {"documents": docs, "total_chunks": len(rag_service.documents)}


@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    """Delete a document from RAG index."""
    success = rag_service.delete_document(filename)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"status": "success", "message": f"Document '{filename}' deleted."}


@app.delete("/api/documents")
async def clear_all_documents():
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


# ============================================================
# GRIEVANCE & PACS ENDPOINTS
# ============================================================

from db import db_service

class GrievanceStatusUpdate(BaseModel):
    status: str
    pacs_remarks: Optional[str] = ""

@app.get("/api/grievances")
async def list_grievances(pacs_centre: Optional[str] = None, department: Optional[str] = None, status: Optional[str] = None):
    """Retrieve citizen grievances for PACS operator dashboard."""
    records = db_service.get_grievances(pacs_centre=pacs_centre, department=department, status=status)
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
async def update_grievance(tracking_id: str, req: GrievanceStatusUpdate):
    """Update grievance resolution status and PACS remarks."""
    updated = db_service.update_grievance_status(tracking_id, req.status, req.pacs_remarks or "")
    if not updated:
        raise HTTPException(status_code=404, detail="Grievance ticket not found.")
    return {"status": "success", "message": f"Grievance '{tracking_id}' updated to {req.status}."}


@app.get("/api/status")
async def get_system_status():
    """System health check and loaded configurations."""
    return {
        "status": "online",
        "model": os.environ.get("OPENCODE_MODEL", "nemotron-3.5-flash-free"),
        "rag_documents": len(rag_service.list_documents()),
        "rag_chunks": len(rag_service.documents),
        "grievance_stats": db_service.get_pacs_stats(),
        "serper_active": bool(os.environ.get("SERPER_API_KEY")),
        "opencode_active": bool(os.environ.get("OPENCODE_API_KEY"))
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
