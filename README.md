# Multilingual Voice Assistant Chatbot

Voice-to-voice multilingual assistant for government schemes. Speech is
transcribed, the LLM translates intent to English, searches the live web
via DuckDuckGo (as many rounds as needed), and replies in the user's
language (Tanglish, Tamil, Hindi, English, etc.), spoken aloud with
edge-tts.

## Features

- Hands-free wake-word control: say **"alexa"** to start a conversation,
  and say **"alexa"** again to interrupt while speaking or searching
- Continuous conversation: after a reply it keeps listening, only
  returning to the wake-word idle state after ~15s of inactivity
- Speech input via microphone with Silero VAD
- Whisper (faster-whisper) transcription, language auto-detection
- LLM agent loop with `web_search` tool calling (OpenCode Zen, OpenAI-compatible)
- Parallel web searches: multiple tool calls per round run concurrently via
  `ThreadPoolExecutor` (up to `MAX_PARALLEL_SEARCHES=4`), usually ~4x faster than
  running them one by one
- Always searches for the latest data (current year is injected into prompts)
- Language mirroring: Tanglish -> Tanglish, Tamil -> Tamil script, etc.
- Soft acceptance chime when the wake word is detected / interrupt is accepted
- edge-tts spoken replies using the newer natural Indian voice
  (`en-IN-NeerjaExpressiveNeural`)

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

Say **"alexa"** to wake the assistant, then ask your question.
Say **"alexa"** at any time (while the reply is playing or the search
is running) to stop it and speak again.

Say "talk in Tamil" / "switch to Hindi" / "speak in tanglish" to change
the response language.

## Files

- `chatbot.py` — mic capture, VAD, wake word, Whisper, language resolution, TTS
- `llm_service.py` — search-agent LLM service (web search + tool calling)