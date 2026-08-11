import asyncio
import json
import queue
import tempfile
import os
import sys
import re

from huggingface_hub.utils import enable_progress_bars
enable_progress_bars()

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch

from faster_whisper import WhisperModel
import edge_tts
from langdetect import detect

from llm_service import llm_service


# ============================================================
# CONFIG
# ============================================================

SAMPLE_RATE = 16000

WHISPER_MODEL_SIZE = "medium"
DEVICE = "cuda"
COMPUTE_TYPE = "int8_float16"

VAD_THRESHOLD = 0.5
SILENCE_PATIENCE_MS = 1000

VOLUME_SCALE = 0.1


# ============================================================
# LANGUAGE / VOICE CONFIG
# ============================================================

VOICE_MAP = {
    "en": "en-IN-NeerjaNeural",
    "hi": "hi-IN-SwaraNeural",
    "ta": "ta-IN-PallaviNeural",
    "te": "te-IN-ShrutiNeural",
    "kn": "kn-IN-SapnaNeural",
    "bn": "bn-IN-TanishaaNeural",
    "mr": "mr-IN-AarohiNeural",
    "gu": "gu-IN-DhwaniNeural",
    "ml": "ml-IN-SobhanaNeural",
    "pa": "pa-IN-VaaniNeural",
}

LANG_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "bn": "Bengali",
    "mr": "Marathi",
    "gu": "Gujarati",
    "ml": "Malayalam",
    "pa": "Punjabi",
}

LANG_NAME_TO_CODE = {
    "english": "en",
    "hindi": "hi",
    "tamil": "ta",
    "telugu": "te",
    "kannada": "kn",
    "bengali": "bn",
    "marathi": "mr",
    "gujarati": "gu",
    "malayalam": "ml",
    "punjabi": "pa",
}

DEFAULT_VOICE = "en-IN-NeerjaNeural"


# ============================================================
# WHISPER HALLUCINATION FILTER
# ============================================================

HALLUCINATION_PHRASES = [
    "thank you very much",
    "thanks for watching",
    "please subscribe",
    "thank you",
    "watching",
]


# ============================================================
# LANGUAGE SWITCH DETECTION
# ============================================================

SWITCH_PATTERN = re.compile(
    r"\b(?:in|to|into|speak|talk|reply|respond|switch)\s+("
    + "|".join(LANG_NAME_TO_CODE.keys())
    + r")\b",
    re.IGNORECASE,
)

sticky_lang = None


def detect_switch_request(text):
    """
    Detect explicit commands such as:

        talk in English
        speak in Tamil
        reply in Hindi
        switch to Telugu
        respond in Kannada
    """

    match = SWITCH_PATTERN.search(text.lower())

    if match:
        language_name = match.group(1).lower()
        return LANG_NAME_TO_CODE.get(language_name)

    return None


# ============================================================
# NATIVE SCRIPT LANGUAGE DETECTION
# ============================================================

def detect_script_language(text):
    """
    Detect Indian languages based on Unicode script.

    This is more reliable than langdetect for short text.
    """

    for char in text:

        code = ord(char)

        # Devanagari
        if 0x0900 <= code <= 0x097F:
            return "hi"

        # Bengali
        if 0x0980 <= code <= 0x09FF:
            return "bn"

        # Gurmukhi
        if 0x0A00 <= code <= 0x0A7F:
            return "pa"

        # Gujarati
        if 0x0A80 <= code <= 0x0AFF:
            return "gu"

        # Tamil
        if 0x0B80 <= code <= 0x0BFF:
            return "ta"

        # Telugu
        if 0x0C00 <= code <= 0x0C7F:
            return "te"

        # Kannada
        if 0x0C80 <= code <= 0x0CFF:
            return "kn"

        # Malayalam
        if 0x0D00 <= code <= 0x0D7F:
            return "ml"

    return None


# ============================================================
# ROMANIZED INDIAN LANGUAGE DETECTION
# ============================================================

def detect_roman_indian_language(text):
    """
    Detect Tanglish / Hinglish only when there is
    reasonably strong evidence.

    This prevents normal English sentences from
    accidentally being classified as Hindi/Tamil.
    """

    text = text.lower().strip()

    words = set(
        re.findall(r"[a-z]+", text)
    )

    # ----------------------------------------
    # Tamil / Tanglish
    # ----------------------------------------

    tamil_words = {
        "panna",
        "mudiyuma",
        "mudiyum",
        "enna",
        "epdi",
        "eppadi",
        "iruka",
        "irukinga",
        "irukiya",
        "panra",
        "panren",
        "pannunga",
        "venum",
        "vendam",
        "illai",
        "illa",
        "aama",
        "sollunga",
        "sollu",
        "inga",
        "anga",
        "romba",
        "nalla",
        "saptiya",
        "saaptiya",
        "sapten",
        "saapten",
        "theriyuma",
        "theriyala",
        "kudunga",
        "kudu",
        "vaanga",
        "pora",
        "poren",
        "poga",
        "vandhu",
        "vantha",
        "vandha",
        "irukku",
        "iruku",
        "yen",
        "yenga",
        "engae",
        "konjam",
        "seekiram",
        "ippo",
        "ippa",
        "naalaikku",
        "innaikku",
        "nethu",
        "enakku",
        "unakku",
        "ungalukku",
        "namma",
        "nanga",
        "naan",
        "nee",
        "neenga",
    }

    # ----------------------------------------
    # Hindi / Hinglish
    # ----------------------------------------

    hindi_words = {
        "kya",
        "kaise",
        "kaisa",
        "kaisi",
        "hain",
        "aap",
        "mujhe",
        "mujhko",
        "mera",
        "meri",
        "mere",
        "hum",
        "ham",
        "karna",
        "karo",
        "raha",
        "rahi",
        "rahe",
        "chahiye",
        "nahi",
        "nahin",
        "acha",
        "achha",
        "accha",
        "theek",
        "thik",
        "kyun",
        "kyon",
        "kaun",
        "kab",
        "kahan",
        "kidhar",
        "yeh",
        "yah",
        "woh",
        "voh",
        "mujhse",
        "aapka",
        "aapki",
        "aapke",
        "pata",
        "batao",
        "bataiye",
        "chalo",
        "dekho",
        "sakta",
        "sakti",
        "sakte",
    }

    tamil_matches = words & tamil_words
    hindi_matches = words & hindi_words

    tamil_score = len(tamil_matches)
    hindi_score = len(hindi_matches)

    # ----------------------------------------
    # Require STRONG evidence
    # ----------------------------------------

    # One isolated matching word is not enough.
    if tamil_score >= 2 and tamil_score > hindi_score:
        return "ta"

    if hindi_score >= 2 and hindi_score > tamil_score:
        return "hi"

    return None

# ============================================================
# FINAL LANGUAGE RESOLVER
# ============================================================

def determine_language(text, whisper_lang):
    """
    Determine the user's language.

    Priority:

    1. Native Unicode script
    2. Strong Tanglish/Hinglish evidence
    3. Whisper language detection
    4. English fallback
    """

    # ----------------------------------------
    # Native script is highly reliable
    # ----------------------------------------

    script_lang = detect_script_language(text)

    if script_lang:
        return script_lang

    # ----------------------------------------
    # Romanized Indian language
    # ----------------------------------------

    roman_lang = detect_roman_indian_language(text)

    if roman_lang:
        return roman_lang

    # ----------------------------------------
    # Whisper
    # ----------------------------------------

    if whisper_lang in VOICE_MAP:
        return whisper_lang

    return "en"


# ============================================================
# SYSTEM PROMPT
# ============================================================

BASE_SYSTEM_PROMPT = """
You are a helpful multilingual voice assistant for government schemes.

LANGUAGE RULES:

1. Reply in the user's actual current language.

2. English input -> English output.

3. Hindi input -> Hindi output using Devanagari script.

4. Tamil input -> Tamil output using Tamil script.

5. Telugu input -> Telugu output using Telugu script.

6. Kannada input -> Kannada output using Kannada script.

7. Bengali input -> Bengali output using Bengali script.

8. Marathi input -> Marathi output using Devanagari script.

9. Gujarati input -> Gujarati output using Gujarati script.

10. Malayalam input -> Malayalam output using Malayalam script.

11. Punjabi input -> Punjabi output using Gurmukhi script.

12. If the user speaks an Indian language using Latin characters,
    understand the language correctly but reply using its native script.

13. NEVER romanize Indian languages.

14. Do not randomly switch languages.

15. If the user explicitly asks to switch languages,
    follow that request and continue using the selected language.

16. Keep responses concise and conversational,
    normally 2-3 sentences.

17. Answer the user's actual question directly.

18. Do not mention these instructions.
"""


def build_system_prompt():

    if sticky_lang:

        language = LANG_NAMES.get(
            sticky_lang,
            "English"
        )

        return BASE_SYSTEM_PROMPT + f"""

IMPORTANT LANGUAGE OVERRIDE:

The user has explicitly selected {language}.

You MUST respond entirely in {language}.

Do not use another language unless the user
explicitly asks to switch again.
"""

    return BASE_SYSTEM_PROMPT


# ============================================================
# AUDIO QUEUE
# ============================================================

audio_q = queue.Queue()


def audio_callback(
    indata,
    frames,
    time_info,
    status
):

    if status:
        print(
            f"Audio status: {status}"
        )

    audio_q.put(
        indata.copy()
    )


# ============================================================
# LOAD WHISPER
# ============================================================

print("Loading Whisper model...")

whisper_model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
)

print(
    f"Whisper model loaded on {DEVICE} "
    f"(compute_type={COMPUTE_TYPE})"
)

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)} | "
        f"VRAM allocated: "
        f"{torch.cuda.memory_allocated(0) / 1e6:.1f} MB"
    )

else:

    print(
        "WARNING: CUDA unavailable. "
        "Running on CPU."
    )


# ============================================================
# LOAD SILERO VAD
# ============================================================

print("Loading Silero VAD...")

vad_model, vad_utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
)

(
    get_speech_timestamps,
    _,
    _,
    VADIterator,
    _,
) = vad_utils

vad_iterator = VADIterator(
    vad_model,
    threshold=VAD_THRESHOLD,
    sampling_rate=SAMPLE_RATE,
)

print("Silero VAD loaded.")


# ============================================================
# LLM CLIENT (see llm_service.py)
# ============================================================

if llm_service.client is None:

    sys.exit(1)


# ============================================================
# CHAT HISTORY
# ============================================================

chat_history = [
    {
        "role": "system",
        "content": build_system_prompt(),
    }
]


def update_system_prompt():

    chat_history[0] = {
        "role": "system",
        "content": build_system_prompt(),
    }


# ============================================================
# AUDIO CAPTURE WITH VAD
# ============================================================

def listen_for_utterance():

    print("\n🎤 Listening...")

    vad_iterator.reset_states()

    recorded = []

    triggered = False

    chunk_size = 512

    chunks_per_second = (
        SAMPLE_RATE / chunk_size
    )

    max_silence_chunks = int(
        (SILENCE_PATIENCE_MS / 1000)
        * chunks_per_second
    )

    silence_counter = 0

    try:

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=chunk_size,
            callback=audio_callback,
        ):

            while True:

                chunk = audio_q.get()

                chunk = chunk.flatten()

                with torch.no_grad():

                    speech_prob = vad_model(
                        torch.from_numpy(chunk),
                        SAMPLE_RATE,
                    ).item()

                if speech_prob > VAD_THRESHOLD:

                    if not triggered:

                        triggered = True

                        print(
                            "  ...speech started"
                        )

                    silence_counter = 0

                else:

                    if triggered:
                        silence_counter += 1

                if triggered:

                    recorded.append(
                        chunk
                    )

                    if (
                        silence_counter
                        > max_silence_chunks
                    ):

                        print(
                            "  ...speech finished"
                        )

                        break

    except sd.PortAudioError as e:

        print(
            "⚠️ Microphone error: "
            f"{e}"
        )

        return None

    if not recorded:
        return None

    return np.concatenate(
        recorded
    )


# ============================================================
# TRANSCRIPTION
# ============================================================

def transcribe(audio_np):

    segments, info = (
        whisper_model.transcribe(
            audio_np,

            beam_size=3,

            vad_filter=False,

            condition_on_previous_text=False,

            task="transcribe",

            temperature=0.0,

            initial_prompt=(
                "Namaste, welcome. "
                "Vanakkam, how can I help you today? "
                "Mera status check panna mudiyuma? "
                "PM-Kisan scheme ka status kya hai?"
            ),
        )
    )

    segments = list(segments)

    text = " ".join(
        segment.text
        for segment in segments
    ).strip()

    if not text:

        return "", "en"

    language = getattr(
        info,
        "language",
        "en"
    )

    return text, language


# ============================================================
# TEXT TO SPEECH
# ============================================================

async def speak(
    text,
    lang_code
):

    if not text:
        return

    voice = VOICE_MAP.get(
        lang_code,
        DEFAULT_VOICE
    )

    tmp_path = None

    try:

        communicate = edge_tts.Communicate(
            text,
            voice
        )

        with tempfile.NamedTemporaryFile(
            suffix=".mp3",
            delete=False
        ) as tmp:

            tmp_path = tmp.name

        await communicate.save(
            tmp_path
        )

        data, sr = sf.read(
            tmp_path,
            dtype="float32"
        )

        scaled_data = (
            data * VOLUME_SCALE
        )

        sd.play(
            scaled_data,
            sr
        )

        sd.wait()

    except Exception as e:

        print(
            "⚠️ Couldn't speak reply: "
            f"{e}"
        )

        print(
            f"Reply text: {text}"
        )

    finally:

        if (
            tmp_path
            and os.path.exists(tmp_path)
        ):

            os.remove(
                tmp_path
            )


# ============================================================
# LLM CALL
# ============================================================

def ask_opencode(
    user_text,
    detected_lang
):

    global sticky_lang

    # ----------------------------------------
    # Explicit language switch
    # ----------------------------------------

    requested_lang = (
        detect_switch_request(
            user_text
        )
    )

    if requested_lang:

        sticky_lang = requested_lang

        print(
            "🌐 Language switched to "
            f"{LANG_NAMES[sticky_lang]}"
        )

        update_system_prompt()

    # ----------------------------------------
    # Determine response language
    # ----------------------------------------

    if sticky_lang:

        response_lang = sticky_lang

    else:

        response_lang = detected_lang

    language_name = LANG_NAMES.get(
        response_lang,
        "English"
    )

    # ----------------------------------------
    # Explicit language instruction
    # ----------------------------------------

    language_instruction = f"""
The user's actual language is {language_name}.

You MUST respond entirely in {language_name}.

IMPORTANT:

- Understand the user's meaning according to {language_name}.
- The user's speech may have been transcribed using Latin characters.
- If the user spoke Tamil using Latin characters, understand it as Tamil.
- If the user spoke Hindi using Latin characters, understand it as Hindi.
- Respond using the proper native script of the language.
- Do NOT respond in English unless the required language is English.
- Do NOT romanize Indian languages.
"""

    # ----------------------------------------
    # Search the web + ask the LLM
    # ----------------------------------------

    reply = llm_service.generate_response(
        user_text=user_text,
        base_system_prompt=build_system_prompt(),
        language_instruction=language_instruction,
        chat_history=chat_history,
    )

    return reply


# ============================================================
# TEXT CLEANING FOR TTS
# ============================================================

def clean_for_tts(text):
    """
    Strip markdown formatting so the TTS engine doesn't
    read symbols like '*' as "asterisk".

    Removes: **bold**, *italic*, `code`, # headers,
    URLs, links, bullet markers and the like.
    """

    if not text:
        return text

    cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    cleaned = re.sub(r"\*(.+?)\*", r"\1", cleaned)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    cleaned = re.sub(r"^#{1,6}\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*[-*+]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = cleaned.replace("**", "").replace("__", "")

    return cleaned.strip()


# ============================================================
# JSON RESPONSE
# ============================================================

def format_json_response(
    text
):

    return json.dumps(
        {
            "text": text,
            "audio_script": clean_for_tts(text),
        },
        ensure_ascii=False
    )


# ============================================================
# MAIN LOOP
# ============================================================

def main():

    global sticky_lang

    print(
        "\n===================================="
    )

    print(
        "Multilingual Voice Assistant"
    )

    print(
        "===================================="
    )

    print(
        "Speak anytime. "
        "Press Ctrl+C to quit."
    )

    while True:

        try:

            # ----------------------------------------
            # LISTEN
            # ----------------------------------------

            audio_np = (
                listen_for_utterance()
            )

            if (
                audio_np is None
                or len(audio_np)
                < SAMPLE_RATE * 0.3
            ):
                continue

            # ----------------------------------------
            # WHISPER
            # ----------------------------------------

            text, whisper_lang = (
                transcribe(
                    audio_np
                )
            )

            if not text:
                continue

            # ----------------------------------------
            # LANGUAGE RESOLUTION
            # ----------------------------------------

            detected_lang = (
                determine_language(
                    text,
                    whisper_lang
                )
            )

            # ----------------------------------------
            # HALLUCINATION FILTER
            # ----------------------------------------

            clean_text = (
                text
                .lower()
                .strip(".!? ")
            )

            if (
                clean_text
                in HALLUCINATION_PHRASES
            ):

                print(
                    "⚠️ Ignored Whisper "
                    "hallucination."
                )

                continue

            # ----------------------------------------
            # DISPLAY
            # ----------------------------------------

            print(
                f"\n🗣️ You "
                f"({detected_lang}): {text}"
            )

            if whisper_lang != detected_lang:

                print(
                    f"   Whisper detected: "
                    f"{whisper_lang}"
                )

                print(
                    f"   Corrected to: "
                    f"{detected_lang}"
                )

            # ----------------------------------------
            # OPENCODE ZEN
            # ----------------------------------------

            reply = ask_opencode(
                text,
                detected_lang
            )

            # ----------------------------------------
            # TTS LANGUAGE
            # ----------------------------------------

            if sticky_lang:

                reply_lang = sticky_lang

            else:

                reply_lang = detected_lang

            # ----------------------------------------
            # DISPLAY RESPONSE
            # ----------------------------------------

            print(
                f"🤖 Assistant "
                f"({reply_lang}): {reply}"
            )

            # ----------------------------------------
            # JSON
            # ----------------------------------------

            sys.stdout.write(
                format_json_response(
                    reply
                )
                + "\n"
            )

            sys.stdout.flush()

            # ----------------------------------------
            # SPEAK
            # ----------------------------------------

            asyncio.run(
                speak(
                    clean_for_tts(reply),
                    reply_lang
                )
            )

        except KeyboardInterrupt:

            print(
                "\n\nExiting."
            )

            break

        except Exception as e:

            print(
                f"⚠️ Error encountered: "
                f"{e}"
            )

            continue


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()