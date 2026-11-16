"""
Atlas - Offline Voice & Acoustic Desktop Agent
Configuration & Tunable Parameters
"""

import os
from pathlib import Path

# Suppress noisy HuggingFace symlink warning on Windows
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ==============================================================================
# 1. BASE DIRECTORIES & PATHS
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
NOTES_DIR = DATA_DIR / "notes"
SNIPPETS_DIR = DATA_DIR / "snippets"
DOWNLOADS_DIR = Path.home() / "Downloads" / "AtlasDownloads"
MODELS_DIR = BASE_DIR / "models"

# Ensure local directories exist
for _directory in (DATA_DIR, NOTES_DIR, SNIPPETS_DIR, DOWNLOADS_DIR, MODELS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. AUDIO CAPTURE SETTINGS
# ==============================================================================
SAMPLE_RATE = 16000          # 16 kHz required for Vosk, WebRTC VAD, and Whisper
CHANNELS = 1                 # Mono audio
BLOCK_DURATION_MS = 30       # 30ms frame size (480 samples @ 16kHz) for WebRTC VAD compatibility
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION_MS / 1000)
INPUT_DEVICE_INDEX = None    # None = default system microphone

# ==============================================================================
# 3. DUAL WAKE ENGINE SETTINGS (listener.py)
# ==============================================================================
# --- A. Double-Clap Detector ---
CLAP_DETECTION_ENABLED = True
CLAP_BANDPASS_LOW_HZ = 1000.0     # Lower cutoff for percussive transient filter (1 kHz)
CLAP_BANDPASS_HIGH_HZ = 4000.0    # Upper cutoff for percussive transient filter (4 kHz)
CLAP_RMS_THRESHOLD = 0.08         # Minimum bandpass RMS amplitude for a clap spike (tune 0.03 - 0.25)
CLAP_PEAK_TO_AVG_RATIO = 4.5      # Spike sharpness ratio vs rolling background energy (ignores speech)
CLAP_MAX_SPIKE_MS = 120           # Max duration (ms) of a single clap impulse (speech syllables last longer)
CLAP_WINDOW_MIN_SEC = 0.20        # Minimum interval between 1st and 2nd clap (seconds)
CLAP_WINDOW_MAX_SEC = 0.80        # Maximum interval between 1st and 2nd clap (seconds)
CLAP_COOLDOWN_SEC = 1.20          # Cooldown after triggering a double-clap

# --- B. Wake Word Detector ---
WAKE_WORD_ENABLED = True
WAKE_WORD_ENGINE = "vosk"         # Options: "vosk" or "openwakeword"
WAKE_PHRASES = ["wake up", "hey atlas", "atlas"]
VOSK_MODEL_PATH = MODELS_DIR / "vosk-model-small-en-us-0.15"
OPENWAKEWORD_MODEL = "hey_jarvis" # Fallback/custom openWakeWord model name or .onnx path
OPENWAKEWORD_THRESHOLD = 0.55

# ==============================================================================
# 4. SPEECH-TO-TEXT & VAD SETTINGS (transcriber.py)
# ==============================================================================
WHISPER_MODEL_SIZE = "base.en"    # Options: "base.en", "small.en"
WHISPER_DEVICE = "cpu"            # "cpu" (int8, fast & no CUDA DLL needed) or "auto" / "cuda"
WHISPER_COMPUTE_TYPE = "int8"     # "int8" for fast CPU inference, "float16"/"float32" for GPU

VAD_ENGINE = "webrtcvad"          # Options: "webrtcvad", "silero", or "energy"
VAD_AGGRESSIVENESS = 2            # WebRTC VAD aggressiveness (0 to 3, 3 = most aggressive filtering)
VAD_ENERGY_THRESHOLD = 0.015      # Fallback RMS threshold when computing voice activity
VAD_SPEECH_START_TIMEOUT_SEC = 6.0  # Max wait time for user to start speaking after wake
VAD_SILENCE_STOP_SEC = 1.2        # Stop recording after this many seconds of silence post-speech
VAD_MAX_RECORD_SEC = 25.0         # Hard ceiling for single command recording length

# ==============================================================================
# 5. LOCAL LLM ROUTER SETTINGS (brain.py)
# ==============================================================================
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("ATLAS_LLM_MODEL", "llama3.2")  # Or "qwen2.5:7b"
OLLAMA_TIMEOUT_SEC = 45

# ==============================================================================
# 6. TEXT-TO-SPEECH & FEEDBACK SETTINGS (speaker.py)
# ==============================================================================
TTS_ENGINE = "pyttsx3"            # Options: "pyttsx3" or "piper"
TTS_RATE = 185                    # Words per minute for pyttsx3
TTS_VOLUME = 1.0                  # 0.0 to 1.0
TTS_VOICE_INDEX = 2               # 2 = Microsoft David (English US), 1 = Microsoft Zira (English US)
TTS_VOICE_PREFER_ENGLISH = True   # Always select an English voice (ignores French/Hortense default)
PIPER_EXECUTABLE = "piper"        # Path to piper binary if TTS_ENGINE == "piper"
PIPER_MODEL_PATH = MODELS_DIR / "en_US-lessac-medium.onnx"

PLAY_CHIME_ON_WAKE = True
SPEAK_LISTENING_ON_WAKE = True    # Speaks "Listening" upon wake trigger

# ==============================================================================
# 7. EXTERNAL BINARIES & TOOL SETTINGS (tools.py)
# ==============================================================================
# Tesseract OCR path on Windows (auto-detected if on PATH)
TESSERACT_CMD = os.environ.get(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe" if os.name == "nt" else "tesseract",
)

# Workspace Profiles for Tool #12: switch_workspace(mode)
WORKSPACE_PROFILES = {
    "coding": {
        "urls": [
            "https://github.com",
            "http://localhost:3000",
        ],
        "apps_launch": [
            "code",  # VS Code
        ],
        "processes_close": [
            "steam.exe" if os.name == "nt" else "steam",
        ],
    },
    "chill": {
        "urls": [
            "https://www.youtube.com",
            "https://open.spotify.com",
        ],
        "apps_launch": [],
        "processes_close": [
            "Code.exe" if os.name == "nt" else "code",
        ],
    },
}

# Default folder organization categories for Tool #6: organize_folder(target_dir)
FOLDER_CATEGORIES = {
    "Images": {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp", ".ico"},
    "Documents": {".pdf", ".doc", ".docx", ".txt", ".md", ".rtf", ".odt", ".xlsx", ".csv", ".pptx"},
    "Archives": {".zip", ".rar", ".7z", ".tar", ".gz", ".bz2"},
    "Audio": {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a"},
    "Video": {".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv"},
    "Code": {".py", ".js", ".ts", ".html", ".css", ".json", ".yaml", ".yml", ".cpp", ".c", ".rs", ".go", ".sh", ".ps1"},
    "Installers": {".exe", ".msi", ".dmg", ".deb", ".rpm", ".appimage"},
}
