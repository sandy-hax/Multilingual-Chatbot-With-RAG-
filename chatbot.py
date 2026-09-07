import asyncio
import json
import queue
import tempfile
import os
import sys
import re
import threading
import time

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

from openwakeword import Model as WakeWordModel

import openwakeword as _oww_pkg

from llm_service import llm_service

# [Fix 1] Import shared language config from speech_config so that
# importing chatbot.py does not duplicate definitions already in speech_config.
from speech_config import (
    VOICE_MAP,
    LANG_NAMES,
    LANG_NAME_TO_CODE,
    DEFAULT_VOICE,
    BASE_SYSTEM_PROMPT,
    detect_switch_request,
    detect_script_language,
    detect_roman_indian_language,
    determine_language,
)


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

# Wake word / interrupt detection (openWakeWord)
WAKE_WORD_MODEL = "alexa_v0.1"
WAKE_THRESHOLD = 0.5
WAKE_COOLDOWN_S = 1.0

# How long (seconds) of no wake-word activity before the
# mic-stream is considered silent/idle again.
WAKE_IDLE_TIMEOUT_S = 15.0

# Ignore "alexa" interrupts during the first N seconds of LLM
# processing. The tail of the user's own just-finished speech can
# still false-trigger the wake model right after they stop talking,
# and no genuine interrupt is needed before transcription/search
# has had time to start.
INTERRUPT_GRACE_S = 2.0


# ============================================================
# LANGUAGE / VOICE CONFIG  — imported from speech_config.py
# ============================================================
# (VOICE_MAP, LANG_NAMES, LANG_NAME_TO_CODE, DEFAULT_VOICE already
#  imported at the top of this file from speech_config)


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
# LANGUAGE SWITCH / STICKY LANGUAGE STATE
# ============================================================
# detect_switch_request() imported from speech_config

sticky_lang = None


# ============================================================
# LANGUAGE DETECTION / SYSTEM PROMPT  — imported from speech_config.py
# ============================================================
# (detect_script_language, detect_roman_indian_language, determine_language,
#  BASE_SYSTEM_PROMPT already imported at the top of this file)


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

vad_model, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
)

print("Silero VAD loaded.")


# ============================================================
# WAKE WORD MODEL (openWakeWord "alexa")
# ============================================================

print("Loading wake word model...")

WAKE_MODEL_PATH = os.path.join(
    os.path.dirname(_oww_pkg.__file__),
    "resources",
    "models",
    f"{WAKE_WORD_MODEL}.onnx",
)

wake_word_model = WakeWordModel(
    wakeword_model_paths=[WAKE_MODEL_PATH],
)

print(
    "Wake word model loaded: "
    f"{WAKE_WORD_MODEL}"
)


# ============================================================
# THREAD STATE / EVENTS
# ============================================================

mode_lock = threading.Lock()
current_mode = "idle"  # idle | acknowledging | listening | speaking

wake_event = threading.Event()
interrupt_event = threading.Event()
utterance_event = threading.Event()
utterance_audio = None
last_utterance_ts = 0.0
stop_event = threading.Event()


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
# AUDIO CAPTURE WITH VAD + WAKE WORD WORKER
# ============================================================

def play_acceptance_sound():
    """
    Play a soft, barely-audible 'accepted' blip,
    well below normal speech volume.
    """
    sr = 22050

    duration = 0.35

    t = np.linspace(
        0,
        duration,
        int(sr * duration),
        endpoint=False,
    )

    # Single gentle tone with a slow fade in/out
    tone = np.sin(2 * np.pi * 1174 * t)

    envelope = np.minimum(
        t / 0.1,
        (duration - t) / 0.2,
    )
    envelope = np.clip(envelope, 0, 1)

    blip = tone * envelope * 0.4

    # Low volume: a small fraction of normal speech level
    soft_blip = blip * VOLUME_SCALE * 0.5

    sd.play(
        soft_blip.astype("float32"),
        sr,
    )

    sd.wait()


def audio_worker():
    """
    Background thread: owns the single persistent mic stream.

    - idle / speaking  -> runs openWakeWord to detect "alexa"
    - listening        -> runs Silero VAD to capture an utterance
    """

    global current_mode, utterance_audio, last_utterance_ts

    chunk_size = 512

    chunks_per_second = (
        SAMPLE_RATE / chunk_size
    )

    max_silence_chunks = int(
        (SILENCE_PATIENCE_MS / 1000)
        * chunks_per_second
    )

    recorded = []
    triggered = False
    silence_counter = 0

    # openWakeWord expects 16 kHz int16 mono in 1280-sample
    # (80 ms) frames. We accumulate raw float chunks into
    # an int16 buffer and slice frames off the front.
    ww_buffer = np.zeros(0, dtype=np.int16)

    last_wake_time = 0.0

    # When switching to "listening", drain a short period of
    # audio so the echo of the "Yes"/acceptance chime that
    # just played does not get captured as user speech.
    SETTLE_CHUNKS = int(
        0.6 * SAMPLE_RATE / 512
    )
    settle_chunks = 0
    prev_mode = None

    # When leaving "listening", the user's own just-finished
    # speech is still sitting in the audio queue. Drain it and
    # reset the wake model so that tail can't false-trigger
    # "alexa" once we resume wake-word detection.
    WW_SETTLE_CHUNKS = int(
        1.5 * SAMPLE_RATE / 512
    )
    ww_settle_chunks = 0

    try:

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=chunk_size,
            callback=audio_callback,
        ):

            while not stop_event.is_set():

                chunk = audio_q.get()

                chunk = chunk.flatten()

                with mode_lock:
                    mode = current_mode

                if mode != prev_mode:

                    # Mode changed.
                    if mode == "listening":

                        recorded = []
                        triggered = False
                        silence_counter = 0
                        settle_chunks = SETTLE_CHUNKS

                    else:

                        # Leaving listening: the captured utterance
                        # (and its trailing audio still in the queue)
                        # must not feed the wake-word detector.
                        ww_buffer = np.zeros(0, dtype=np.int16)
                        ww_settle_chunks = WW_SETTLE_CHUNKS

                        with mode_lock:
                            wake_word_model.reset()

                    prev_mode = mode

                if mode == "listening":

                    if settle_chunks > 0:

                        # Drain echo of the acceptance sound
                        settle_chunks -= 1
                        continue

                    # ------------------------------------
                    # VAD-based utterance capture
                    # ------------------------------------

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

                            utterance_audio = (
                                np.concatenate(recorded)
                            )

                            recorded = []
                            triggered = False
                            silence_counter = 0

                            with mode_lock:
                                current_mode = "idle"

                            last_utterance_ts = time.time()

                            utterance_event.set()

                            # Stop here THIS iteration: the chunk
                            # containing the tail of the user's speech
                            # must not fall through into wake-word
                            # detection below, or it can false-trigger
                            # "alexa". Next iteration's mode-change
                            # branch handles the ww_settle drain +
                            # model reset.
                            continue

                elif mode in ("idle", "speaking", "processing"):

                    # ------------------------------------
                    # Wake word detection
                    # ------------------------------------

                    if ww_settle_chunks > 0:

                        # Drain the tail of the user's own
                        # just-finished speech.
                        ww_settle_chunks -= 1
                        continue

                    int16 = (
                        np.clip(
                            chunk,
                            -1.0,
                            1.0,
                        )
                        * 32767.0
                    ).astype(np.int16)

                    ww_buffer = (
                        np.concatenate(
                            [ww_buffer, int16]
                        )
                    )

                    while len(ww_buffer) >= 1280:

                        frame = ww_buffer[:1280]

                        ww_buffer = ww_buffer[1280:]

                        predictions = (
                            wake_word_model.predict(
                                frame,
                            )
                        )

                        score = predictions.get(
                            WAKE_WORD_MODEL,
                            0.0,
                        )

                        if score > WAKE_THRESHOLD:

                            now = time.time()

                            if (
                                now - last_wake_time
                                > WAKE_COOLDOWN_S
                            ):

                                last_wake_time = now

                                with mode_lock:
                                    mode_now = current_mode

                                if mode_now == "idle":

                                    print(
                                        "\n🔔 Wake word "
                                        "detected!"
                                    )

                                    wake_event.set()

                                elif (
                                    mode_now
                                    in ("speaking", "processing")
                                ):

                                    # Skip the false-interrupt window
                                    # right after the user spoke: the
                                    # tail of their speech can trick
                                    # the wake model into thinking
                                    # "alexa" was said.
                                    if (
                                        time.time()
                                        - last_utterance_ts
                                        < INTERRUPT_GRACE_S
                                    ):
                                        continue

                                    print(
                                        "\n⏹️ Interrupt "
                                        "detected!"
                                    )

                                    interrupt_event.set()

                # 'acknowledging' -> just drain audio

    except sd.PortAudioError as e:

        print(
            "⚠️ Microphone error: "
            f"{e}"
        )

    except Exception as e:

        print(
            f"⚠️ Audio worker error: {e}"
        )

    finally:

        print("Audio worker stopped.")


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
    lang_code,
    interruptible=False
):
    """
    Speak `text`. If interruptible, monitor the mic
    wake-word and stop playback early when "alexa"
    is detected. Returns True if interrupted.
    """

    if not text:
        return False

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

        if not interruptible:

            sd.wait()

            return False

        # ----------------------------------------
        # Interruptible playback: keep draining
        # the output stream while listening for
        # the wake word to stop us.
        # ----------------------------------------

        interrupted = False

        while True:

            stream = sd.get_stream()

            if (
                stream is None
                or not stream.active
            ):
                break

            if interrupt_event.is_set():

                interrupt_event.clear()

                sd.stop()

                interrupted = True

                break

            await asyncio.sleep(0.05)

        return interrupted

    except Exception as e:

        print(
            "⚠️ Couldn't speak reply: "
            f"{e}"
        )

        print(
            f"Reply text: {text}"
        )

        return False

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
    detected_lang,
    cancel_event=None
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

    if response_lang == "tl":

        language_instruction = """
The user's actual language is Tanglish (Tamil mixed with English,
written using Latin characters).

You MUST respond entirely in Tanglish.

IMPORTANT:

- Reply naturally in Tanglish, exactly how Tamil speakers chat.
- Write Tamil in Latin script mixed with everyday English words.
- Examples: "Apply panna mudiyum", "konjam wait pannungo",
  "ee scheme la eligibility ennana?", "income limit kaala venum".
- Do NOT use Tamil script.
- Keep it conversational and easy to read aloud.
"""

    else:

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
        cancel_event=cancel_event,
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

    global sticky_lang, current_mode

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
        "Say 'alexa' to start a conversation. "
        "Say 'alexa' while I speak to interrupt. "
        "Press Ctrl+C to quit."
    )

    # ----------------------------------------
    # Start the persistent mic / wake word
    # background thread.
    # ----------------------------------------

    worker = threading.Thread(
        target=audio_worker,
        daemon=True,
    )

    worker.start()

    try:

        while True:

            # ================================
            # 1. IDLE - wait for wake word
            # ================================

            print(
                "\n🔇 Waiting for wake word... "
                "(say 'alexa')"
            )

            wake_event.clear()

            wake_event.wait()

            wake_event.clear()

            print(
                "\n✅ Wake word detected!"
            )

            # ----------------------------------------
            # Acknowledge: say "yes" + acceptance
            # sound, then listen for the user.
            # ----------------------------------------

            with mode_lock:
                current_mode = "acknowledging"

            asyncio.run(
                speak(
                    "Yes",
                    "en"
                )
            )

            play_acceptance_sound()

            with mode_lock:
                current_mode = "listening"

            print(
                "\n🎤 Listening..."
            )

            # ================================
            # 2. CONVERSATION LOOP
            # ================================

            while True:

                # ----------------------------------------
                # Wait for a captured utterance
                # ----------------------------------------

                utterance_ready = (
                    utterance_event.wait(
                        timeout=WAKE_IDLE_TIMEOUT_S
                    )
                )

                if not utterance_ready:

                    # Nobody spoke in time; back to idle
                    with mode_lock:
                        current_mode = "idle"

                    print(
                        "\n⏳ No speech detected. "
                        "Back to idle."
                    )

                    break

                utterance_event.clear()

                with mode_lock:
                    current_mode = "processing"

                audio_np = utterance_audio

                if (
                    audio_np is None
                    or len(audio_np)
                    < SAMPLE_RATE * 0.3
                ):
                    continue

                # ----------------------------------------
                # Run transcription + LLM search in a
                # background thread so "alexa" can cancel
                # the search while it is still running.
                # ----------------------------------------

                result_box = {}
                cancel_event = threading.Event()
                done_event = threading.Event()

                def _process():

                    try:

                        # WHISPER
                        text, whisper_lang = (
                            transcribe(
                                audio_np
                            )
                        )

                        if not text:

                            result_box["done"] = True
                            return

                        # LANGUAGE RESOLUTION
                        detected_lang = (
                            determine_language(
                                text,
                                whisper_lang
                            )
                        )

                        # HALLUCINATION FILTER
                        clean_text = (
                            text
                            .lower()
                            .strip(".!? ")
                        )

                        if (
                            clean_text
                            in HALLUCINATION_PHRASES
                        ):

                            result_box["ignored"] = True
                            result_box["done"] = True
                            return

                        # OPENCODE ZEN (cancellable)
                        reply = ask_opencode(
                            text,
                            detected_lang,
                            cancel_event=cancel_event,
                        )

                        result_box["text"] = text
                        result_box["whisper_lang"] = whisper_lang
                        result_box["detected_lang"] = detected_lang
                        result_box["reply"] = reply

                    except Exception as e:

                        print(
                            f"⚠️ Processing error: {e}"
                        )

                    finally:

                        result_box["done"] = True
                        done_event.set()

                threading.Thread(
                    target=_process,
                    daemon=True,
                ).start()

                # ----------------------------------------
                # Wait, allowing "alexa" to cancel
                # ----------------------------------------

                cancelled = False

                # Right after an utterance, the wake model can still
                # false-fire on the tail of the user's own speech.
                # Ignore interrupts for a short grace period so we
                # don't cancel a real answer over that.
                processing_started = time.time()

                while not done_event.is_set():

                    if interrupt_event.is_set():

                        interrupt_event.clear()

                        if (
                            time.time() - processing_started
                            > INTERRUPT_GRACE_S
                        ):

                            cancel_event.set()

                            cancelled = True

                            with mode_lock:
                                current_mode = "acknowledging"

                            play_acceptance_sound()

                            with mode_lock:
                                current_mode = "listening"

                            print(
                                "\n🎤 Listening..."
                            )

                            break

                        # Within the grace window: a spurious
                        # interrupt, ignore it and keep waiting.

                    time.sleep(0.05)

                # Cancelled mid-search: re-listen
                if cancelled or not result_box.get("done"):
                    continue

                if result_box.get("ignored"):

                    print(
                        "⚠️ Ignored Whisper "
                        "hallucination."
                    )

                    continue

                text = result_box["text"]
                whisper_lang = result_box["whisper_lang"]
                detected_lang = result_box["detected_lang"]
                reply = result_box["reply"]

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
                # SPEAK (interruptible)
                # ----------------------------------------

                with mode_lock:
                    current_mode = "speaking"

                interrupted = asyncio.run(
                    speak(
                        clean_for_tts(reply),
                        reply_lang,
                        interruptible=True
                    )
                )

                if interrupted:

                    # User said "alexa" while we spoke:
                    # stop, chime, and listen again.
                    with mode_lock:
                        current_mode = "acknowledging"

                    play_acceptance_sound()

                    with mode_lock:
                        current_mode = "listening"

                    print(
                        "\n🎤 Listening..."
                    )

                    continue

                # Reply finished normally; keep the
                # conversation going. The inner loop only
                # returns to idle when nobody speaks for
                # WAKE_IDLE_TIMEOUT_S, so we just continue
                # listening without needing "alexa" again.
                with mode_lock:
                    current_mode = "listening"

                print(
                    "\n🎤 Listening..."
                )

                continue

    except KeyboardInterrupt:

        print(
            "\n\nExiting."
        )

    finally:

        stop_event.set()

        sd.stop()

        worker.join(
            timeout=2.0
        )

        llm_service.search_engine.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()