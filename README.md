# Multilingual AI Voice & Hybrid RAG Assistant

A state-of-the-art voice-to-voice and text multilingual assistant designed for citizen services, government schemes (PMFBY, PACS bye-laws, etc.), and document QA. Powered by **Nemotron 3.5 Flash Free** on OpenCode Zen, local **Hybrid RAG** (Dense Vector + BM25 Keyword Search), **Unlimited Local OCR**, and a modern **Glassmorphism Web UI**.

---

## 🌟 Key Features

1. **Nemotron 3.5 Flash Free LLM Core**:
   - Integrates with OpenCode Zen API (`https://opencode.ai/zen/v1`).
   - Translates multi-dialect intents (Tanglish, Hinglish, Tamil, Hindi, English) to precise English search queries.
   - Dual-tool execution loop: queries local RAG Knowledge Base and live Web Search engine (Serper / DuckDuckGo).

2. **Hybrid RAG Knowledge Ingestion & OCR**:
   - **Unlimited Local OCR**: Processes digital PDFs, text files, Word documents, and scanned images (PNG, JPG, WEBP) using `EasyOCR` and `PyPDF`.
   - **Section & Clause Aware Chunking**: Smart paragraph and clause boundary splitting.
   - **Dense + Sparse Hybrid Retrieval**: Combines Hugging Face embeddings (`sentence-transformers/all-MiniLM-L6-v2`) with BM25 keyword matching via **Reciprocal Rank Fusion (RRF)**.

3. **Intent Routing & Grievance Flow**:
   - Automatically routes legal/scheme queries to the RAG Knowledge Base.
   - Identifies citizen complaints and guides users through structured follow-up (PACS, district) and filing on the **CPGRAMS** portal (`pgportal.gov.in`).

4. **Speech & Multilingual Engine**:
   - **Faster-Whisper (`medium` model)** for speech recognition & automatic language detection.
   - **Silero VAD** for silence & utterance boundary detection.
   - **Edge-TTS** for speech output matching the user's native script or Tanglish (Latin Tamil).

5. **Modern Glassmorphism Web UI & Voice CLI**:
   - **Web UI (`app.py`)**: Glassmorphism dark-mode web application featuring real-time mic streaming, audio player, drag-and-drop document ingestion, and interactive architecture flow visualizer.
   - **Hands-Free Voice CLI (`chatbot.py`)**: Continuous voice interaction with wake-word detection ("alexa").

---

## 🛠️ Architecture Workflow

```
[Citizen Input] -> [Silero VAD + Faster-Whisper (medium)]
                          │
                          ▼
                 [Language & Intent Router]
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
       [RAG Store]   [Web Search]  [Grievance Flow]
     (OCR + Vector)  (Serper/DDG)   (CPGRAMS Guide)
            └─────────────┬─────────────┘
                          ▼
               [Nemotron 3.5 Flash Free]
                          │
                          ▼
              [Edge-TTS Spoken Audio Output]
```

---

## 🚀 Quick Start & Installation

### 1. Prerequisites & Virtual Environment

```bash
# Clone repository and navigate to directory
cd /path/to/mpva

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install all requirements
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and set your OpenCode Zen API key:

```bash
cp .env.example .env
```

`.env` configuration:
```env
OPENCODE_API_KEY="sk-your-opencode-api-key-here"
OPENCODE_MODEL="nemotron-3.5-flash-free"

# Optional: Fast Google Search via Serper (2,500 free queries, fallback to DuckDuckGo if empty)
SERPER_API_KEY="your-serper-api-key-here"
```

---

## 💻 Running the Application

### Option A: Web User Interface (Recommended)

Start the FastAPI Web Server:

```bash
python app.py
```

Then open your browser and navigate to:
👉 **`http://localhost:8000`**

- **Chat & Voice**: Click the microphone icon to record speech or type in Tanglish, Tamil, Hindi, or English.
- **RAG Knowledge Hub**: Drag & drop PDFs, images, or TXT files onto the sidebar to run local OCR & document QA.
- **Architecture Visualizer**: Click **Flow Diagram** in the header to inspect the system routing.

### Option B: Hands-Free Voice CLI

Start the terminal voice chatbot with wake-word support:

```bash
python chatbot.py
```

- Say **"alexa"** to wake the assistant, then speak your query.
- Say **"alexa"** again at any time to interrupt speech output.
- Ask to switch language: *"talk in Tamil"*, *"switch to Hindi"*, *"speak in Tanglish"*.

---

## 📁 Repository Structure

```
├── app.py              # FastAPI Web Application & API endpoints (/chat, /upload, /stt, /tts)
├── chatbot.py          # Hands-free voice assistant CLI (VAD, Whisper medium, wake word)
├── llm_service.py      # Nemotron 3.5 LLM agent service with dual tool execution (RAG + Web)
├── rag_service.py      # Hybrid RAG engine (embeddings + BM25 + RRF ranking)
├── ocr_service.py      # OCR & document extraction (PDF, EasyOCR, images, docx)
├── requirements.txt    # Python package dependencies
├── .env.example        # Environment variable template
├── static/
│   ├── index.html      # Glassmorphism HTML5 UI
│   ├── css/style.css   # Dark theme CSS styling & micro-animations
│   └── js/app.js       # Client JS (mic recording, audio playback, file drag-and-drop)
└── rag_storage/        # Local RAG vector index & chunk metadata storage
```

---

## 📜 License

MIT License. Designed for citizen services and open multilingual assistant research.