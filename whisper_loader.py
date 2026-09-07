"""
whisper_loader.py
=================
Thread-safe lazy singleton for Faster-Whisper model.

The model is loaded on first request (not at server startup) so that
importing app.py does NOT trigger GPU model initialization.  Subsequent
calls return the already-loaded instance instantly.
"""

import threading
import os
from typing import Optional

from faster_whisper import WhisperModel

_whisper_lock = threading.Lock()
_whisper_model: Optional[WhisperModel] = None


def get_whisper_model() -> WhisperModel:
    """
    Return the shared WhisperModel instance, loading it on first call.

    Model size and compute device are read from environment variables:
      WHISPER_MODEL_SIZE  (default: "medium")
      WHISPER_DEVICE      (default: "cpu"  — use "cuda" when GPU is available)
      WHISPER_COMPUTE     (default: "int8")
    """
    global _whisper_model

    if _whisper_model is not None:
        return _whisper_model

    with _whisper_lock:
        # Double-checked locking to avoid redundant initialization
        if _whisper_model is None:
            model_size = os.environ.get("WHISPER_MODEL_SIZE", "medium")
            device = os.environ.get("WHISPER_DEVICE", "cpu")
            compute_type = os.environ.get("WHISPER_COMPUTE", "int8")

            # Gracefully fall back to CPU if cuda is requested but unavailable
            try:
                import torch
                if device == "cuda" and not torch.cuda.is_available():
                    print("⚠️  CUDA requested but unavailable — falling back to CPU for Whisper.")
                    device = "cpu"
                    compute_type = "int8"
            except ImportError:
                device = "cpu"
                compute_type = "int8"

            print(f"Loading Whisper model '{model_size}' on {device} ({compute_type})…")
            _whisper_model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
            )
            print("✅ Whisper model ready.")

    return _whisper_model
