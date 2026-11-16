"""
Atlas - Text-to-Speech & Acoustic Feedback Engine (speaker.py)
Provides 100% offline, thread-safe spoken confirmations using pyttsx3 or piper-tts,
plus subtle acoustic chimes for wake events.
"""

import os
import subprocess
import tempfile
import threading
import wave
from pathlib import Path

import numpy as np

import config

_tts_lock = threading.Lock()


def play_chime(kind: str = "wake") -> None:
    """
    Play a subtle acoustic chime to confirm wake detection or completion.
    Uses synthesized sine tones via sounddevice if available, falling back to winsound on Windows.
    """
    if not config.PLAY_CHIME_ON_WAKE and kind == "wake":
        return

    try:
        import sounddevice as sd

        sr = 24000
        if kind == "wake":
            # Two-note ascending harmonic chime (523Hz C5 -> 784Hz G5)
            t1 = np.linspace(0, 0.07, int(sr * 0.07), endpoint=False)
            t2 = np.linspace(0, 0.11, int(sr * 0.11), endpoint=False)
            tone1 = 0.22 * np.sin(2 * np.pi * 523.25 * t1) * np.exp(-t1 * 18)
            tone2 = 0.25 * np.sin(2 * np.pi * 783.99 * t2) * np.exp(-t2 * 14)
            audio = np.concatenate([tone1, tone2]).astype(np.float32)
        else:
            # Subtle descending confirmation tone
            t1 = np.linspace(0, 0.09, int(sr * 0.09), endpoint=False)
            audio = (0.20 * np.sin(2 * np.pi * 659.25 * t1) * np.exp(-t1 * 16)).astype(np.float32)

        sd.play(audio, samplerate=sr, blocking=True)
        return
    except Exception:
        pass

    if os.name == "nt":
        try:
            import winsound

            if kind == "wake":
                winsound.Beep(587, 70)
                winsound.Beep(880, 90)
            else:
                winsound.Beep(659, 80)
        except Exception:
            pass


def _speak_with_piper(text: str) -> bool:
    """Attempt offline synthesis with piper-tts if configured and present."""
    model_path = Path(config.PIPER_MODEL_PATH)
    if not model_path.exists():
        return False

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            wav_path = tmp_wav.name

        proc = subprocess.run(
            [
                config.PIPER_EXECUTABLE,
                "--model",
                str(model_path),
                "--output_file",
                wav_path,
            ],
            input=text.encode("utf-8"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
        )
        if proc.returncode == 0 and os.path.exists(wav_path):
            import sounddevice as sd

            with wave.open(wav_path, "rb") as wf:
                sr = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                sd.play(audio, samplerate=sr, blocking=True)
            os.remove(wav_path)
            return True
    except Exception:
        pass
    return False


def _speak_with_pyttsx3(text: str) -> bool:
    """Speak text offline using pyttsx3 (Windows SAPI5 / macOS NSSpeech / Linux espeak)."""
    try:
        if os.name == "nt":
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except Exception:
                pass

        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", config.TTS_RATE)
        engine.setProperty("volume", config.TTS_VOLUME)

        voices = engine.getProperty("voices") or []
        selected_voice_id = None

        if getattr(config, "TTS_VOICE_PREFER_ENGLISH", True) and voices:
            # First check if the configured index is already an English voice
            if 0 <= config.TTS_VOICE_INDEX < len(voices):
                candidate = voices[config.TTS_VOICE_INDEX]
                cand_meta = f"{candidate.id} {candidate.name} {getattr(candidate, 'languages', '')}".lower()
                if any(tag in cand_meta for tag in ("en-us", "en_us", "en-gb", "english", "david", "zira")):
                    selected_voice_id = candidate.id

            # Otherwise auto-select the first English voice (e.g. David or Zira)
            if not selected_voice_id:
                for v in voices:
                    v_meta = f"{v.id} {v.name} {getattr(v, 'languages', '')}".lower()
                    if any(tag in v_meta for tag in ("en-us", "en_us", "en-gb", "english", "david", "zira")):
                        selected_voice_id = v.id
                        break

        if not selected_voice_id and voices and 0 <= config.TTS_VOICE_INDEX < len(voices):
            selected_voice_id = voices[config.TTS_VOICE_INDEX].id

        if selected_voice_id:
            engine.setProperty("voice", selected_voice_id)

        engine.say(text)
        engine.runAndWait()
        engine.stop()
        return True
    except Exception as exc:
        print(f"[Speaker] pyttsx3 error: {exc}")
        return False


def speak(text: str, async_mode: bool = False) -> None:
    """
    Speak a short confirmation message locally.
    If async_mode is True, runs in a daemon thread so the caller is not blocked.
    """
    if not text or not text.strip():
        return

    clean_text = " ".join(text.strip().split())
    print(f"\n[Atlas] 🔊 {clean_text}")

    def _worker() -> None:
        with _tts_lock:
            if config.TTS_ENGINE.lower() == "piper":
                if _speak_with_piper(clean_text):
                    return
            _speak_with_pyttsx3(clean_text)

    if async_mode:
        threading.Thread(target=_worker, daemon=True).start()
    else:
        _worker()

