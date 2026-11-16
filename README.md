# Atlas — 100% Offline, Local-First Voice & Acoustic Desktop Agent

**Atlas** is a 100% free, completely offline, privacy-first voice and acoustic desktop assistant built in Python. Every component—wake-word spotting, double-clap transient detection, speech-to-text, LLM tool routing, and speech synthesis—runs strictly on your local hardware with **zero paid APIs and zero cloud dependencies**.

---

## 1. Architecture & File Structure

| File | Responsibility |
| :--- | :--- |
| `main.py` | Core state loop (`Idle -> Chime/Listening -> Record until Silence -> Whisper STT -> Ollama Tool Router -> Execute Tool -> Speak Confirmation -> Idle`) + CLI flags |
| `listener.py` | **Dual Wake Engine**: (1) Real-time **1kHz–4kHz Bandpass Double-Clap Detector** (`0.2s–0.8s` window, transient sharpness rejection of speech) and (2) Offline **Wake Word Detector** (`"wake up"` via `Vosk` or `openWakeWord`) |
| `transcriber.py` | Voice Activity Detection (`webrtcvad` / Silero VAD / RMS fallback) + local `faster-whisper` (`base.en` / `small.en`) |
| `brain.py` | Local `Ollama` LLM router (`llama3.2` or `qwen2.5:7b`) using strict JSON structured outputs & tool-calling |
| `tools.py` | Validated registry of the **15 local desktop tools** (zero arbitrary shell execution) |
| `speaker.py` | Thread-safe offline Text-to-Speech (`pyttsx3` / `piper-tts`) and synthesized acoustic wake chimes |
| `config.py` | Centralized paths, audio device selection, clap sensitivity thresholds, VAD timeouts, and workspace profiles |
| `Atlas.lnk` | Taskbar-pinnable Windows shortcut (created via `create_shortcut.py` / `launch_atlas.bat`) |

---

## 2. Prerequisites & System Dependencies

### A. Local LLM via Ollama
1. Install [Ollama](https://ollama.com/download).
2. Pull either `llama3.2` (default, fast & lightweight) or `qwen2.5:7b`:
   ```powershell
   ollama pull llama3.2
   # OR
   ollama pull qwen2.5:7b
   ```
   *(If you use `qwen2.5:7b`, set `OLLAMA_MODEL = "qwen2.5:7b"` in `config.py` or set environment variable `ATLAS_LLM_MODEL=qwen2.5:7b`.)*

### B. FFmpeg & yt-dlp (For `download_media`)
- **Windows (winget / scoop / choco):**
  ```powershell
  winget install Gyan.FFmpeg
  ```
- **Linux (Ubuntu/Debian):**
  ```bash
  sudo apt install ffmpeg
  ```
  *(`yt-dlp` is automatically installed via `requirements.txt`.)*

### C. Tesseract OCR (For `screenshot_ocr`)
- **Windows:**
  ```powershell
  winget install UB-Mannheim.TesseractOCR
  ```
  By default, `config.py` checks `C:\Program Files\Tesseract-OCR\tesseract.exe` or your system `PATH`.
- **Linux:**
  ```bash
  sudo apt install tesseract-ocr
  ```

---

## 3. Installation & Setup

```powershell
cd d:\Atlas

# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Download offline Vosk wake-word model (~40MB) and pre-cache faster-whisper (base.en)
python main.py --download-models
```

---

## 4. Running Atlas & Pinning to the Taskbar

### Start the Voice & Acoustic Agent
```powershell
python main.py
```
Or double-click **`Atlas.lnk`** (or `launch_atlas.bat`).

### Pinning `Atlas.lnk` to the Windows Taskbar
1. In `d:\Atlas` (or on your Desktop), locate **`Atlas.lnk`**.
2. Right-click **`Atlas.lnk`** -> **Show more options** -> **Pin to taskbar** (or drag `Atlas.lnk` directly onto the taskbar).
   - *To regenerate the shortcut at any time, run:* `python create_shortcut.py`.

### Helpful CLI Flags
- **Interactive Text Mode** (test LLM tool routing without a mic):
  ```powershell
  python main.py --text
  ```
- **List Microphone Input Devices**:
  ```powershell
  python main.py --list-devices
  ```

---

## 5. The 15 Built-in Local Tools (`tools.py`)

1. **`append_to_file(filename, content)`** — Appends timestamped notes to `.txt` or `.md` files in `data/notes/`.
2. **`summarize_clipboard(mode="summary")`** — Summarizes copied text or explains copied error logs via local Ollama, speaks the result, and optionally saves it.
3. **`download_media(format_type="mp3")`** — Reads a video URL from the clipboard and downloads it via `yt-dlp` in a background thread (`mp3` or `mp4`) to `~/Downloads/AtlasDownloads/`.
4. **`dictate_to_cursor(text)`** — Cleans grammar/punctuation via local Ollama and pastes at the active cursor (`pyperclip` + `pyautogui`).
5. **`save_clipboard_snippet(name, language="txt")`** — Saves current clipboard text to `data/snippets/<name>.<language>`.
6. **`organize_folder(target_dir)`** — Sorts files in a folder (e.g., `Downloads`) into categorized subfolders, or invokes `custom_file_organizer_hook(target_dir)` in `tools.py`.
7. **`search_files(query, search_dir)`** — Recursively searches filenames and highlights the top match in File Explorer.
8. **`scaffold_project(project_name, stack="python")`** — Creates a project folder, runs `git init`, writes `.gitignore`, and creates starter boilerplate.
9. **`git_quick_push(repo_path, commit_message)`** — Runs `git add .`, `git commit -m`, and `git push`.
10. **`system_control(action, value=None)`** — Cross-platform OS control for `"lock"`, `"mute"`, `"unmute"`, or `"volume"` (`0–100` via `pycaw` on Windows / `pactl` or `amixer` on Linux).
11. **`docker_control(action, container_name="")`** — Runs `docker ps`, `docker start`, or `docker stop`.
12. **`switch_workspace(mode)`** — Opens URLs/apps and closes distractions for `"coding"` or `"chill"` profiles configured in `config.py`.
13. **`screenshot_ocr()`** — Captures the screen (`mss`), extracts text with `pytesseract`, copies it to the clipboard, and reports character count.
14. **`start_timer(minutes, label="Focus")`** — Runs a non-blocking background timer thread that chimes and speaks when finished.
15. **`system_health()`** — Reports CPU %, RAM %, and free disk space in GB via `psutil`.

---

## 6. Tuning Double-Clap Sensitivity (`config.py`)

If your room or microphone requires sensitivity adjustment, edit Section 3A of `config.py`:
- `CLAP_RMS_THRESHOLD = 0.08`: Lower to `0.04` for quieter claps or distant mics; raise to `0.14` in noisy environments.
- `CLAP_PEAK_TO_AVG_RATIO = 4.5`: Controls how sharp the transient spike must be compared to rolling background noise.
- `CLAP_WINDOW_MIN_SEC = 0.20` / `CLAP_WINDOW_MAX_SEC = 0.80`: Required time window between the first and second clap.

