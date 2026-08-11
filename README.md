# Multilingual Voice Assistant Chatbot

Voice-to-voice multilingual assistant for government schemes. Speech is
transcribed, the LLM translates intent to English, searches the live web
via DuckDuckGo (as many rounds as needed), and replies in the user's
language (Tanglish, Tamil, Hindi, English, etc.).

## Features

- Speech input via microphone with Silero VAD
- Whisper (faster-whisper) transcription, language auto-detection
- LLM agent loop with `web_search` tool calling (OpenCode Zen, OpenAI-compatible)
- Always searches for the latest data (current year is injected into prompts)
- Language mirroring: Tanglish -> Tanglish, Tamil -> Tamil script, etc.
- edge-tts spoken replies

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set your key:

```bash
cp .env.example .env
# edit .env -> OPENCODE_API_KEY=your-key-here
```

## Run

```bash
python chatbot.py
```

Say "talk in Tamil" / "switch to Hindi" to change the response language.

## Files

- `chatbot.py` — mic capture, VAD, Whisper, language resolution, TTS
- `llm_service.py` — search-agent LLM service (web search + tool calling)
