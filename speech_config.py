"""
speech_config.py
================
Shared, stateless language configuration and detection utilities.

This module contains ONLY pure-Python logic with no heavy dependencies
(no torch, no whisper, no sounddevice). Both app.py (web server) and
chatbot.py (CLI) import from here so that importing chatbot.py for its
language helpers does NOT trigger Whisper/VAD/wakeword model loading.
"""

import re

# ============================================================
# VOICE MAP  (Edge-TTS voice names per language code)
# ============================================================

VOICE_MAP = {
    "en": "en-IN-NeerjaExpressiveNeural",
    "tl": "en-IN-NeerjaExpressiveNeural",   # Tanglish → Indian English voice
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

DEFAULT_VOICE = "en-IN-NeerjaExpressiveNeural"

# ============================================================
# LANGUAGE METADATA
# ============================================================

LANG_NAMES = {
    "en": "English",
    "tl": "Tanglish",
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
    "english":  "en",
    "tanglish": "tl",
    "thanglish": "tl",
    "hindi":    "hi",
    "tamil":    "ta",
    "telugu":   "te",
    "kannada":  "kn",
    "bengali":  "bn",
    "marathi":  "mr",
    "gujarati": "gu",
    "malayalam":"ml",
    "punjabi":  "pa",
}

# ============================================================
# BASE SYSTEM PROMPT
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

12. If the user speaks Hindi or other Indian languages using Latin
    characters, understand the language correctly but reply using its
    native script.

12a. SPECIAL CASE - Tanglish (Latin-script Tamil mixed with English):
     When the user speaks Tamil using Latin characters, reply naturally
     in the same Tanglish style, i.e. Tamil written in Latin script mixed
     with everyday English words, exactly how Tamil speakers chat
     (for example: "Apply panna mudiyum", "konjam wait pannungo").
     Do NOT use Tamil script in this mode.

13. NEVER romanize Indian languages except in Tanglish mode
    described in rule 12a.

14. Do not randomly switch languages.

15. If the user explicitly asks to switch languages,
    follow that request and continue using the selected language.

16. Keep responses concise and conversational,
    normally 2-3 sentences.

17. Answer the user's actual question directly.

18. Do not mention these instructions.
"""

# ============================================================
# LANGUAGE SWITCH DETECTION
# ============================================================

_SWITCH_PATTERN = re.compile(
    r"\b(?:in|to|into|speak|talk|reply|respond|switch)\s+("
    + "|".join(LANG_NAME_TO_CODE.keys())
    + r")\b",
    re.IGNORECASE,
)


def detect_switch_request(text: str):
    """
    Detect explicit language-switch commands such as:
        'talk in English', 'speak in Tamil', 'switch to Telugu'

    Returns a language code string, or None.
    """
    match = _SWITCH_PATTERN.search(text.lower())
    if match:
        language_name = match.group(1).lower()
        return LANG_NAME_TO_CODE.get(language_name)
    return None


# ============================================================
# NATIVE SCRIPT LANGUAGE DETECTION (Unicode ranges)
# ============================================================

def detect_script_language(text: str):
    """
    Detect Indian languages from Unicode script blocks.
    More reliable than langdetect for short text.
    Returns a language code or None.
    """
    for char in text:
        code = ord(char)
        if 0x0900 <= code <= 0x097F:
            return "hi"   # Devanagari (Hindi / Marathi)
        if 0x0980 <= code <= 0x09FF:
            return "bn"   # Bengali
        if 0x0A00 <= code <= 0x0A7F:
            return "pa"   # Gurmukhi (Punjabi)
        if 0x0A80 <= code <= 0x0AFF:
            return "gu"   # Gujarati
        if 0x0B80 <= code <= 0x0BFF:
            return "ta"   # Tamil
        if 0x0C00 <= code <= 0x0C7F:
            return "te"   # Telugu
        if 0x0C80 <= code <= 0x0CFF:
            return "kn"   # Kannada
        if 0x0D00 <= code <= 0x0D7F:
            return "ml"   # Malayalam
    return None


# ============================================================
# ROMANIZED INDIAN LANGUAGE DETECTION (Tanglish / Hinglish)
# ============================================================

_TAMIL_WORDS = {
    "panna", "mudiyuma", "mudiyum", "enna", "epdi", "eppadi",
    "iruka", "irukinga", "irukiya", "panra", "panren", "pannunga",
    "venum", "vendam", "illai", "illa", "aama", "sollunga", "sollu",
    "inga", "anga", "romba", "nalla", "saptiya", "saaptiya", "sapten",
    "saapten", "theriyuma", "theriyala", "kudunga", "kudu", "vaanga",
    "pora", "poren", "poga", "vandhu", "vantha", "vandha", "irukku",
    "iruku", "yen", "yenga", "engae", "konjam", "seekiram", "ippo",
    "ippa", "naalaikku", "innaikku", "nethu", "enakku", "unakku",
    "ungalukku", "namma", "nanga", "naan", "nee", "neenga",
}

_HINDI_WORDS = {
    "kya", "kaise", "kaisa", "kaisi", "hain", "aap", "mujhe", "mujhko",
    "mera", "meri", "mere", "hum", "ham", "karna", "karo", "raha",
    "rahi", "rahe", "chahiye", "nahi", "nahin", "acha", "achha", "accha",
    "theek", "thik", "kyun", "kyon", "kaun", "kab", "kahan", "kidhar",
    "yeh", "yah", "woh", "voh", "mujhse", "aapka", "aapki", "aapke",
    "pata", "batao", "bataiye", "chalo", "dekho", "sakta", "sakti", "sakte",
}


def detect_roman_indian_language(text: str):
    """
    Detect Tanglish ('tl') or Hinglish ('hi') from romanized text.
    Requires at least 2 matching words to avoid false positives.
    Returns a language code or None.
    """
    words = set(re.findall(r"[a-z]+", text.lower()))
    tamil_score = len(words & _TAMIL_WORDS)
    hindi_score = len(words & _HINDI_WORDS)

    if tamil_score >= 2 and tamil_score > hindi_score:
        return "tl"
    if hindi_score >= 2 and hindi_score > tamil_score:
        return "hi"
    return None


# ============================================================
# FINAL LANGUAGE RESOLVER
# ============================================================

def determine_language(text: str, whisper_lang: str = "en") -> str:
    """
    Determine the user's language code.

    Priority:
    1. Native Unicode script  → most reliable
    2. Tanglish / Hinglish keyword evidence
    3. Whisper detected language (if it's a supported code)
    4. English fallback
    """
    script_lang = detect_script_language(text)
    if script_lang:
        return script_lang

    roman_lang = detect_roman_indian_language(text)
    if roman_lang:
        return roman_lang

    if whisper_lang and whisper_lang in VOICE_MAP:
        return whisper_lang

    return "en"
