"""
Atlas - 15 Registered Local Desktop Tools (tools.py)
Provides a decorator-based tool registry with JSON schema generation, parameter alias
normalization, spoken-number parsing, smart defaults, and strict argument validation.
Never executes raw arbitrary shell strings from the LLM.
"""

from datetime import datetime
import inspect
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional
import webbrowser

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore

try:
    import pyperclip
except ImportError:
    pyperclip = None  # type: ignore

import config
import speaker

# ==============================================================================
# TOOL REGISTRY, ALIAS NORMALIZER & SPOKEN NUMBER PARSER
# ==============================================================================
TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {}

# Maps common LLM-hallucinated parameter names to the canonical parameter names
PARAMETER_ALIASES: Dict[str, Dict[str, str]] = {
    "append_to_file": {
        "file": "filename",
        "file_name": "filename",
        "name": "filename",
        "path": "filename",
        "note": "content",
        "text": "content",
        "body": "content",
        "message": "content",
    },
    "summarize_clipboard": {
        "type": "mode",
        "action": "mode",
        "style": "mode",
    },
    "download_media": {
        "format": "format_type",
        "fmt": "format_type",
        "type": "format_type",
        "ext": "format_type",
    },
    "dictate_to_cursor": {
        "content": "text",
        "message": "text",
        "dictation": "text",
        "words": "text",
    },
    "save_clipboard_snippet": {
        "filename": "name",
        "file_name": "name",
        "title": "name",
        "snippet_name": "name",
        "lang": "language",
        "ext": "language",
        "extension": "language",
    },
    "organize_folder": {
        "folder": "target_dir",
        "directory": "target_dir",
        "dir": "target_dir",
        "path": "target_dir",
        "target": "target_dir",
    },
    "search_files": {
        "filename": "query",
        "name": "query",
        "keyword": "query",
        "search": "query",
        "file": "query",
        "folder": "search_dir",
        "directory": "search_dir",
        "dir": "search_dir",
        "path": "search_dir",
    },
    "scaffold_project": {
        "name": "project_name",
        "project": "project_name",
        "folder": "project_name",
        "language": "stack",
        "template": "stack",
        "framework": "stack",
        "type": "stack",
    },
    "git_quick_push": {
        "path": "repo_path",
        "repo": "repo_path",
        "directory": "repo_path",
        "folder": "repo_path",
        "message": "commit_message",
        "msg": "commit_message",
        "commit": "commit_message",
    },
    "system_control": {
        "command": "action",
        "type": "action",
        "mode": "action",
        "level": "value",
        "volume": "value",
        "percent": "value",
        "amount": "value",
    },
    "docker_control": {
        "command": "action",
        "subcommand": "action",
        "container": "container_name",
        "name": "container_name",
        "id": "container_name",
    },
    "switch_workspace": {
        "profile": "mode",
        "workspace": "mode",
        "name": "mode",
        "environment": "mode",
    },
    "start_timer": {
        "duration": "minutes",
        "time": "minutes",
        "mins": "minutes",
        "minute": "minutes",
        "name": "label",
        "title": "label",
        "reason": "label",
    },
}

_SPOKEN_NUMBERS: Dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "twenty five": 25, "twenty-five": 25, "thirty": 30, "forty": 40,
    "forty five": 45, "forty-five": 45, "fifty": 50, "sixty": 60,
    "seventy": 70, "seventy five": 75, "seventy-five": 75, "eighty": 80,
    "ninety": 90, "hundred": 100, "one hundred": 100, "a minute": 1, "an hour": 60,
}


def parse_spoken_int(value: Any, default: int = 1) -> int:
    """Convert numeric strings, percentages ('65%'), or spoken English numbers ('fifteen') to int."""
    if isinstance(value, (int, float)):
        return int(round(value))
    if not isinstance(value, str):
        return default

    cleaned = value.lower().strip().replace("%", "").replace("percent", "").replace("minutes", "").replace("minute", "").strip()
    if cleaned in _SPOKEN_NUMBERS:
        return _SPOKEN_NUMBERS[cleaned]

    # Check numeric digits inside string
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if match:
        return int(round(float(match.group(0))))

    # Check compound spoken numbers like "twenty five"
    total = 0
    matched_any = False
    for word in cleaned.replace("-", " ").split():
        if word in _SPOKEN_NUMBERS:
            total += _SPOKEN_NUMBERS[word]
            matched_any = True
    if matched_any:
        return total

    return default


def register_tool(description: str, parameters: Dict[str, Dict[str, Any]], required: Optional[List[str]] = None):
    """Decorator to register one of the 15 allowed desktop tools with JSON schema metadata."""
    def decorator(func: Callable) -> Callable:
        TOOL_REGISTRY[func.__name__] = {
            "name": func.__name__,
            "description": description,
            "func": func,
            "schema": {
                "type": "function",
                "function": {
                    "name": func.__name__,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": parameters,
                        "required": required or [],
                    },
                },
            },
        }
        return func
    return decorator


def get_ollama_tools_schema() -> List[Dict[str, Any]]:
    """Return the OpenAI/Ollama-compatible tool schema list for all 15 registered tools."""
    return [entry["schema"] for entry in TOOL_REGISTRY.values()]


def _normalize_arguments(tool_name: str, raw_args: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize LLM argument keys using PARAMETER_ALIASES for the given tool."""
    if not isinstance(raw_args, dict):
        return {}
    aliases = PARAMETER_ALIASES.get(tool_name, {})
    normalized: Dict[str, Any] = {}
    for k, v in raw_args.items():
        canon_key = aliases.get(k.lower().strip(), k)
        if canon_key not in normalized or normalized[canon_key] in (None, ""):
            normalized[canon_key] = v
    return normalized


def execute_registered_tool(tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
    """
    Validate, normalize, and execute an explicitly registered tool from TOOL_REGISTRY.
    Refuses unknown tool names and safely coerces parameter types.
    """
    if tool_name not in TOOL_REGISTRY:
        return f"Tool '{tool_name}' is not in the approved registry."

    entry = TOOL_REGISTRY[tool_name]
    func = entry["func"]
    sig = inspect.signature(func)
    norm_args = _normalize_arguments(tool_name, arguments or {})

    validated_kwargs: Dict[str, Any] = {}
    for param_name, param in sig.parameters.items():
        if param_name in norm_args and norm_args[param_name] is not None and norm_args[param_name] != "":
            val = norm_args[param_name]
            if param.annotation in (int, Optional[int]):
                val = parse_spoken_int(val, default=50 if param_name == "value" else 1)
            elif param.annotation is str:
                val = str(val)
            validated_kwargs[param_name] = val
        elif param.default is not inspect.Parameter.empty:
            validated_kwargs[param_name] = param.default
        else:
            return f"Missing required parameter '{param_name}' for {tool_name}."

    try:
        return str(func(**validated_kwargs))
    except Exception as exc:
        return f"Error executing {tool_name}: {exc}"


def _sanitize_name(name: str, default: str = "untitled") -> str:
    """Sanitize a filename stem to prevent path traversal."""
    cleaned = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name.strip())
    cleaned = cleaned.strip("._")
    return cleaned or default


def _resolve_common_directory(path_str: str) -> Path:
    """Resolve user shorthand like 'Downloads', 'Desktop', 'Documents', or full paths."""
    raw = (path_str or "").strip().strip("\"'")
    lower = raw.lower()
    home = Path.home()
    shortcuts = {
        "downloads": home / "Downloads",
        "download": home / "Downloads",
        "my downloads": home / "Downloads",
        "downloads folder": home / "Downloads",
        "desktop": home / "Desktop",
        "my desktop": home / "Desktop",
        "documents": home / "Documents",
        "my documents": home / "Documents",
        "docs": home / "Documents",
        "notes": config.NOTES_DIR,
        "snippets": config.SNIPPETS_DIR,
        "atlas": config.BASE_DIR,
        "home": home,
        ".": config.BASE_DIR,
        "": home / "Downloads",
    }
    if lower in shortcuts:
        return shortcuts[lower]

    expanded = Path(os.path.expandvars(os.path.expanduser(raw)))
    if not expanded.is_absolute():
        for base in (home, config.BASE_DIR, config.BASE_DIR.parent):
            candidate = base / expanded
            if candidate.exists():
                return candidate.resolve()
    return expanded.resolve()


# ==============================================================================
# CUSTOM HOOK FOR TOOL #6 (organize_folder)
# ==============================================================================
def custom_file_organizer_hook(target_dir: Path) -> Optional[str]:
    """
    Hook function where you can plug in your own custom file manager script!
    If this function returns a non-None string, `organize_folder` will use it
    instead of the default extension-based categorizer.
    """
    return None


# ==============================================================================
# THE 15 REGISTERED TOOLS
# ==============================================================================

# 1. append_to_file
@register_tool(
    description="Appends timestamped text to a .txt or .md file in the configured notes directory.",
    parameters={
        "filename": {"type": "string", "description": "Name of the note file (e.g. 'notes.md', 'ideas.md', 'todo.txt')"},
        "content": {"type": "string", "description": "The text content to append to the file"},
    },
    required=["content"],
)
def append_to_file(filename: str = "notes.md", content: str = "") -> str:
    if not content.strip() and filename != "notes.md":
        # If LLM put the note text in `filename` and left `content` empty
        content, filename = filename, "notes.md"

    if not content.strip():
        return "No note content provided to append."

    safe_name = _sanitize_name(filename, default="notes.md")
    if not (safe_name.lower().endswith(".txt") or safe_name.lower().endswith(".md")):
        safe_name += ".md"

    target_path = config.NOTES_DIR / safe_name
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"\n[{timestamp}] {content.strip()}\n"

    with open(target_path, "a", encoding="utf-8") as f:
        f.write(entry)

    return f"Appended note to {safe_name}."


# 2. summarize_clipboard
@register_tool(
    description="Reads the current system clipboard, passes it to local Ollama to summarize long text or explain an error log, speaks the result, and optionally saves it to a note.",
    parameters={
        "mode": {
            "type": "string",
            "enum": ["summary", "error", "save_summary"],
            "description": "'summary' to summarize text, 'error' to explain a stack trace/log, or 'save_summary' to also save to a note.",
        }
    },
    required=[],
)
def summarize_clipboard(mode: str = "summary") -> str:
    clip_text = pyperclip.paste() if pyperclip else ""
    if not clip_text or not clip_text.strip():
        return "Your clipboard is currently empty."

    import brain

    mode_lower = (mode or "summary").lower().strip()
    if any(k in mode_lower for k in ("error", "bug", "log", "trace", "debug", "explain")):
        system_prompt = (
            "You are a concise senior debugging assistant. Explain the root cause of this error log "
            "and the exact fix in 2 short spoken sentences."
        )
    else:
        system_prompt = (
            "You are a concise assistant. Summarize the following clipboard text in 2 short, "
            "clear sentences suitable for text-to-speech."
        )

    summary = brain.query_local_llm_text(
        prompt=clip_text[:6000],
        system_prompt=system_prompt,
    )

    if "save" in mode_lower:
        append_to_file("clipboard_summaries.md", summary)
        return f"{summary} Saved to clipboard summaries."

    return summary


# 3. download_media
@register_tool(
    description="Reads a video URL from the clipboard and downloads it via yt-dlp in a background thread as 'mp3' or 'mp4'.",
    parameters={
        "format_type": {
            "type": "string",
            "enum": ["mp3", "mp4"],
            "description": "Output media format: 'mp3' (audio only) or 'mp4' (video).",
        }
    },
    required=[],
)
def download_media(format_type: str = "mp3") -> str:
    raw_clip = (pyperclip.paste() if pyperclip else "").strip()
    # Extract first URL if clipboard contains extra text around the link
    url_match = re.search(r"https?://[^\s\"'<>]+", raw_clip)
    if not url_match:
        return "No valid HTTP URL found in the clipboard. Copy a video link first."

    url = url_match.group(0)
    fmt = "mp4" if "mp4" in (format_type or "mp3").lower() or "video" in (format_type or "").lower() else "mp3"
    out_template = str(config.DOWNLOADS_DIR / "%(title)s.%(ext)s")

    def _download_worker(target_url: str, media_fmt: str) -> None:
        has_ffmpeg = shutil.which("ffmpeg") is not None
        cmd = [sys.executable, "-m", "yt_dlp", "--no-playlist", "-o", out_template]
        if media_fmt == "mp3":
            if has_ffmpeg:
                cmd.extend(["-x", "--audio-format", "mp3"])
            else:
                cmd.extend(["-f", "bestaudio/best"])
        else:
            if has_ffmpeg:
                cmd.extend(["-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"])
            else:
                cmd.extend(["-f", "best[ext=mp4]/best"])

        cmd.append(target_url)
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode == 0:
                speaker.speak(f"{media_fmt.upper()} download finished.", async_mode=True)
            else:
                speaker.speak("Media download failed. Check the URL in clipboard.", async_mode=True)
        except Exception:
            speaker.speak("yt-dlp encountered an error.", async_mode=True)

    threading.Thread(target=_download_worker, args=(url, fmt), daemon=True).start()
    return f"Started {fmt.upper()} download in the background."


# 4. dictate_to_cursor
@register_tool(
    description="Cleans punctuation and grammar of dictated text via the local LLM and pastes/types it at the active cursor.",
    parameters={
        "text": {"type": "string", "description": "Raw spoken text to clean and type at the active cursor"},
    },
    required=["text"],
)
def dictate_to_cursor(text: str) -> str:
    if not text or not text.strip():
        return "No dictation text provided."

    import brain

    cleaned = brain.query_local_llm_text(
        prompt=text,
        system_prompt=(
            "Fix only grammar, capitalization, and punctuation for the user's dictated text. "
            "Output ONLY the corrected text with no quotes or commentary."
        ),
    ).strip()

    if not cleaned or cleaned.startswith("Could not reach") or cleaned.startswith("Local LLM error"):
        cleaned = text.strip()

    if pyperclip:
        pyperclip.copy(cleaned)

    try:
        import pyautogui

        time.sleep(0.15)
        pyautogui.hotkey("ctrl", "v")
    except Exception:
        return "Copied cleaned dictation to clipboard, but could not simulate keystroke."

    return "Dictation pasted at cursor."


# 5. save_clipboard_snippet
@register_tool(
    description="Saves current clipboard content into snippets/<name>.<language>.",
    parameters={
        "name": {"type": "string", "description": "Filename stem for the snippet (e.g. 'auth_helper')"},
        "language": {"type": "string", "description": "File extension or language (e.g. 'py', 'js', 'txt', 'md')"},
    },
    required=["name"],
)
def save_clipboard_snippet(name: str = "snippet", language: str = "txt") -> str:
    content = pyperclip.paste() if pyperclip else ""
    if not content:
        return "Clipboard is empty; nothing to save."

    ext_map = {
        "python": "py",
        "py": "py",
        "javascript": "js",
        "js": "js",
        "typescript": "ts",
        "ts": "ts",
        "markdown": "md",
        "md": "md",
        "text": "txt",
        "txt": "txt",
        "shell": "sh",
        "bash": "sh",
        "powershell": "ps1",
        "html": "html",
        "css": "css",
        "json": "json",
        "rust": "rs",
        "go": "go",
        "cpp": "cpp",
        "c++": "cpp",
        "c": "c",
    }
    lang_clean = (language or "txt").lower().strip().lstrip(".")
    ext = ext_map.get(lang_clean, re.sub(r"[^a-z0-9]", "", lang_clean) or "txt")
    safe_stem = _sanitize_name(name, default="snippet")
    if safe_stem.lower().endswith(f".{ext}"):
        filename = safe_stem
    else:
        filename = f"{safe_stem}.{ext}"

    out_path = config.SNIPPETS_DIR / filename
    out_path.write_text(content, encoding="utf-8")
    return f"Saved clipboard snippet as {filename}."


# 6. organize_folder
@register_tool(
    description="Cleans up a directory (e.g., Downloads) by sorting files into categorized subfolders or calling the custom organizer hook.",
    parameters={
        "target_dir": {"type": "string", "description": "Directory name or path to organize (default: 'Downloads')"},
    },
    required=[],
)
def organize_folder(target_dir: str = "Downloads") -> str:
    folder = _resolve_common_directory(target_dir)
    if not folder.exists() or not folder.is_dir():
        return f"Directory {folder} does not exist."

    custom_result = custom_file_organizer_hook(folder)
    if custom_result is not None:
        return custom_result

    moved_count = 0
    for item in folder.iterdir():
        if not item.is_file() or item.name.startswith("."):
            continue

        ext = item.suffix.lower()
        dest_category = "Other"
        for category, extensions in config.FOLDER_CATEGORIES.items():
            if ext in extensions:
                dest_category = category
                break

        dest_folder = folder / dest_category
        dest_folder.mkdir(exist_ok=True)
        dest_path = dest_folder / item.name

        if dest_path.exists():
            stem, suffix = item.stem, item.suffix
            counter = 1
            while dest_path.exists():
                dest_path = dest_folder / f"{stem}_{counter}{suffix}"
                counter += 1

        shutil.move(str(item), str(dest_path))
        moved_count += 1

    return f"Organized {moved_count} files in {folder.name}."


# 7. search_files
@register_tool(
    description="Recursively searches filenames for a query in search_dir and opens the folder of the top match.",
    parameters={
        "query": {"type": "string", "description": "Filename keyword to search for"},
        "search_dir": {"type": "string", "description": "Directory to search in (e.g. 'Documents', 'Downloads', 'Desktop')"},
    },
    required=["query"],
)
def search_files(query: str, search_dir: str = "Documents") -> str:
    q_lower = query.lower().strip()
    if not q_lower:
        return "Please provide a filename keyword to search for."

    primary_root = _resolve_common_directory(search_dir)
    search_roots = [primary_root]
    # If user didn't specify a custom path and file isn't in Documents, also check Downloads & Desktop
    if search_dir.lower().strip() in ("documents", "downloads", "desktop", ""):
        for fallback_name in ("Documents", "Downloads", "Desktop", "Atlas"):
            fb_path = _resolve_common_directory(fallback_name)
            if fb_path not in search_roots and fb_path.exists():
                search_roots.append(fb_path)

    top_match: Optional[Path] = None
    for root in search_roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d for d in dirnames
                if not d.startswith(".") and d not in ("node_modules", "__pycache__", "venv", ".venv", ".git")
            ]
            for fname in filenames:
                if q_lower in fname.lower():
                    top_match = Path(dirpath) / fname
                    break
            if top_match is not None:
                break
        if top_match is not None:
            break

    if top_match is None:
        return f"No files matching '{query}' found in {primary_root.name}."

    parent_dir = top_match.parent
    try:
        if os.name == "nt":
            subprocess.Popen(["explorer", f"/select,{str(top_match)}"])
        elif shutil.which("xdg-open"):
            subprocess.Popen(["xdg-open", str(parent_dir)])
        elif shutil.which("open"):
            subprocess.Popen(["open", "-R", str(top_match)])
    except Exception:
        pass

    return f"Found {top_match.name} and opened {parent_dir.name}."


# 8. scaffold_project
@register_tool(
    description="Creates a new project folder, runs git init, creates a .gitignore, and generates starter boilerplate.",
    parameters={
        "project_name": {"type": "string", "description": "Name of the new project folder"},
        "stack": {
            "type": "string",
            "enum": ["python", "node", "web", "rust"],
            "description": "Tech stack boilerplate to generate (default: 'python')",
        },
    },
    required=["project_name"],
)
def scaffold_project(project_name: str, stack: str = "python") -> str:
    safe_name = _sanitize_name(project_name, default="new_project")
    project_dir = config.BASE_DIR.parent / safe_name
    project_dir.mkdir(parents=True, exist_ok=True)

    stack_lower = (stack or "python").lower().strip()
    if stack_lower in ("node", "javascript", "js", "ts", "typescript"):
        stack_lower = "node"
        (project_dir / ".gitignore").write_text("node_modules/\n.env\ndist/\n", encoding="utf-8")
        (project_dir / "index.js").write_text(f"console.log('Hello from {safe_name}');\n", encoding="utf-8")
    elif stack_lower in ("web", "html", "frontend"):
        stack_lower = "web"
        (project_dir / ".gitignore").write_text(".DS_Store\ndist/\n", encoding="utf-8")
        (project_dir / "index.html").write_text(
            f"<!DOCTYPE html>\n<html><head><title>{safe_name}</title></head><body><h1>{safe_name}</h1></body></html>\n",
            encoding="utf-8",
        )
    else:
        stack_lower = "python"
        (project_dir / ".gitignore").write_text("__pycache__/\n.venv/\nvenv/\n.env\n*.pyc\n", encoding="utf-8")
        (project_dir / "main.py").write_text(
            f'def main():\n    print("Hello from {safe_name}!")\n\n\nif __name__ == "__main__":\n    main()\n',
            encoding="utf-8",
        )
        (project_dir / "README.md").write_text(f"# {safe_name}\n", encoding="utf-8")

    subprocess.run(["git", "init"], cwd=str(project_dir), capture_output=True, check=False)
    return f"Scaffolded {stack_lower} project '{safe_name}' with Git initialized."


# 9. git_quick_push
@register_tool(
    description="Runs git add ., git commit -m <commit_message>, and git push in the specified repository path.",
    parameters={
        "repo_path": {"type": "string", "description": "Path to the git repository (default: current directory '.')"},
        "commit_message": {"type": "string", "description": "Git commit message"},
    },
    required=[],
)
def git_quick_push(repo_path: str = ".", commit_message: str = "Quick update via Atlas") -> str:
    repo = _resolve_common_directory(repo_path)
    if not (repo / ".git").exists():
        return f"{repo.name} is not a Git repository."

    msg = (commit_message or "").strip() or "Quick update via Atlas"
    add_res = subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True, text=True, check=False)
    if add_res.returncode != 0:
        return f"git add failed: {add_res.stderr.strip()}"

    commit_res = subprocess.run(
        ["git", "commit", "-m", msg], cwd=str(repo), capture_output=True, text=True, check=False
    )
    if commit_res.returncode != 0 and "nothing to commit" in (commit_res.stdout + commit_res.stderr).lower():
        return "Working tree is clean; nothing to commit."

    push_res = subprocess.run(["git", "push"], cwd=str(repo), capture_output=True, text=True, check=False)
    if push_res.returncode != 0:
        return f"Committed changes, but git push failed: {push_res.stderr.strip()[:120]}"

    return f"Committed and pushed changes to {repo.name}."


# 10. system_control
@register_tool(
    description="Cross-platform OS control for 'lock' (locks screen), 'mute', 'unmute', or 'volume' (sets volume 0-100).",
    parameters={
        "action": {
            "type": "string",
            "enum": ["lock", "mute", "unmute", "volume"],
            "description": "OS control action to execute",
        },
        "value": {
            "type": "integer",
            "description": "Volume percentage (0-100) when action is 'volume'",
        },
    },
    required=["action"],
)
def system_control(action: str, value: Optional[int] = None) -> str:
    act = (action or "").lower().strip()

    # Normalize action synonyms
    if any(k in act for k in ("lock", "screensaver", "screen")):
        act = "lock"
    elif "unmute" in act:
        act = "unmute"
    elif "mute" in act:
        act = "mute"
    elif any(k in act for k in ("vol", "sound", "audio", "louder", "quieter", "set")):
        act = "volume"

    if act == "lock":
        if os.name == "nt":
            import ctypes

            ctypes.windll.user32.LockWorkStation()
            return "Workstation locked."
        elif shutil.which("loginctl"):
            subprocess.run(["loginctl", "lock-session"], check=False)
            return "Session locked."
        elif shutil.which("xdg-screensaver"):
            subprocess.run(["xdg-screensaver", "lock"], check=False)
            return "Screen locked."
        return "Lock screen command not supported on this OS."

    # Audio control on Windows via pycaw
    if os.name == "nt":
        try:
            import pythoncom
            from ctypes import POINTER, cast
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

            pythoncom.CoInitialize()
            devices = AudioUtilities.GetSpeakers()
            if hasattr(devices, "EndpointVolume"):
                volume_iface = devices.EndpointVolume
            else:
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                volume_iface = cast(interface, POINTER(IAudioEndpointVolume))

            if act == "mute":
                volume_iface.SetMute(1, None)
                return "System audio muted."
            elif act == "unmute":
                volume_iface.SetMute(0, None)
                return "System audio unmuted."
            elif act == "volume":
                lvl = max(0, min(100, parse_spoken_int(value, default=50)))
                volume_iface.SetMute(0, None)
                volume_iface.SetMasterVolumeLevelScalar(lvl / 100.0, None)
                return f"Volume set to {lvl} percent."
        except Exception as exc:
            return f"Windows audio control error: {exc}"

    # Linux audio control via pactl or amixer
    else:
        lvl = max(0, min(100, parse_spoken_int(value, default=50)))
        if shutil.which("pactl"):
            if act == "mute":
                subprocess.run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "1"], check=False)
                return "System audio muted."
            elif act == "unmute":
                subprocess.run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "0"], check=False)
                return "System audio unmuted."
            elif act == "volume":
                subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{lvl}%"], check=False)
                return f"Volume set to {lvl} percent."
        elif shutil.which("amixer"):
            if act == "mute":
                subprocess.run(["amixer", "set", "Master", "mute"], check=False)
                return "System audio muted."
            elif act == "unmute":
                subprocess.run(["amixer", "set", "Master", "unmute"], check=False)
                return "System audio unmuted."
            elif act == "volume":
                subprocess.run(["amixer", "set", "Master", f"{lvl}%"], check=False)
                return f"Volume set to {lvl} percent."

    return f"Unsupported system_control action '{action}'."


# 11. docker_control
@register_tool(
    description="Controls local Docker containers via subprocess ('ps', 'start', or 'stop').",
    parameters={
        "action": {
            "type": "string",
            "enum": ["ps", "start", "stop"],
            "description": "Docker subcommand: 'ps' to list running containers, 'start' or 'stop' for a specific container.",
        },
        "container_name": {
            "type": "string",
            "description": "Container name or ID (required for 'start' and 'stop')",
        },
    },
    required=[],
)
def docker_control(action: str = "ps", container_name: str = "") -> str:
    act = (action or "ps").lower().strip()
    if act in ("list", "status", "show", "running"):
        act = "ps"
    if act not in ("ps", "start", "stop"):
        return "Invalid Docker action. Use 'ps', 'start', or 'stop'."

    if act == "ps":
        try:
            res = subprocess.run(
                ["docker", "ps", "--format", "{{.Names}} ({{.Status}})"],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return "Docker is not installed or not on PATH."
        if res.returncode != 0:
            return "Docker daemon is not running or not installed."
        containers = [line.strip() for line in res.stdout.splitlines() if line.strip()]
        if not containers:
            return "No Docker containers are currently running."
        return f"Running containers: {', '.join(containers[:5])}."

    safe_container = re.sub(r"[^a-zA-Z0-9_\-\.]", "", (container_name or "").strip())
    if not safe_container:
        return f"Please specify a container name to {act}."

    try:
        res = subprocess.run(
            ["docker", act, safe_container],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return "Docker is not installed or not on PATH."

    if res.returncode == 0:
        return f"Docker container {safe_container} {act}ed successfully."
    return f"Failed to {act} container {safe_container}: {res.stderr.strip()[:100]}"


# 12. switch_workspace
@register_tool(
    description="Launches or closes predefined apps and URLs based on 'coding' or 'chill' workspace profiles in config.py.",
    parameters={
        "mode": {
            "type": "string",
            "enum": ["coding", "chill"],
            "description": "Workspace profile name ('coding' or 'chill')",
        },
    },
    required=["mode"],
)
def switch_workspace(mode: str = "coding") -> str:
    raw_mode = (mode or "coding").lower().strip()
    if any(k in raw_mode for k in ("code", "coding", "work", "dev", "development")):
        profile_key = "coding"
    elif any(k in raw_mode for k in ("chill", "relax", "music", "break", "gaming")):
        profile_key = "chill"
    else:
        profile_key = raw_mode

    profile = config.WORKSPACE_PROFILES.get(profile_key)
    if not profile:
        return f"Workspace profile '{mode}' not found. Available: {', '.join(config.WORKSPACE_PROFILES.keys())}."

    for url in profile.get("urls", []):
        webbrowser.open(url)

    for app in profile.get("apps_launch", []):
        if shutil.which(app):
            subprocess.Popen([app])

    target_procs = {p.lower() for p in profile.get("processes_close", [])}
    if target_procs and psutil:
        for proc in psutil.process_iter(["name"]):
            try:
                pname = (proc.info.get("name") or "").lower()
                if pname in target_procs:
                    proc.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    return f"Switched to {profile_key} workspace."


def _windows_native_ocr(img) -> str:
    """100% offline Windows 10/11 built-in OCR fallback when Tesseract binary is not installed."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        img.save(tmp_path)
        ps_script = (
            "Add-Type -AssemblyName System.Runtime.WindowsRuntime\n"
            "[void][Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]\n"
            "[void][Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime]\n"
            "[void][Windows.Graphics.Imaging.BitmapDecoder, Windows.Foundation, ContentType = WindowsRuntime]\n"
            "function Await($WinRtTask, $ResultType) {\n"
            "    $asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {\n"
            "        $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'\n"
            "    })[0]\n"
            "    $asTask = $asTaskGeneric.MakeGenericMethod($ResultType)\n"
            "    $netTask = $asTask.Invoke($null, @($WinRtTask))\n"
            "    $netTask.Wait(-1) | Out-Null\n"
            "    return $netTask.Result\n"
            "}\n"
            f"$file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync('{tmp_path}')) ([Windows.Storage.StorageFile])\n"
            "$stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])\n"
            "$decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])\n"
            "$bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])\n"
            "$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()\n"
            "$result = Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])\n"
            "Write-Output $result.Text\n"
            "$stream.Dispose()\n"
        )
        res = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True,
            text=True,
            check=False,
        )
        return res.stdout.strip()
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# 13. screenshot_ocr
@register_tool(
    description="Captures the screen using mss, extracts text via pytesseract OCR, copies it to the clipboard, and reports character count.",
    parameters={},
    required=[],
)
def screenshot_ocr() -> str:
    import mss
    from PIL import Image, ImageGrab
    import pytesseract

    if os.name == "nt" and os.path.exists(config.TESSERACT_CMD):
        pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD

    img = None
    try:
        mss_cls = getattr(mss, "MSS", mss.mss)
        with mss_cls() as sct:
            monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
            shot = sct.grab(monitor)
            img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
    except Exception:
        try:
            img = ImageGrab.grab()
        except Exception:
            img = Image.new("RGB", (320, 80), color=(255, 255, 255))

    try:
        extracted = pytesseract.image_to_string(img).strip()
    except Exception:
        if os.name == "nt":
            extracted = _windows_native_ocr(img)
        else:
            return "Tesseract OCR is not installed."

    if pyperclip:
        pyperclip.copy(extracted)
    char_count = len(extracted)
    return f"Extracted {char_count} characters from screen and copied to clipboard."


# 14. start_timer
@register_tool(
    description="Starts a non-blocking background timer for the given minutes and speaks an alert when finished.",
    parameters={
        "minutes": {"type": "integer", "description": "Duration of the timer in minutes"},
        "label": {"type": "string", "description": "Optional label for the timer (default: 'Focus')"},
    },
    required=["minutes"],
)
def start_timer(minutes: int = 5, label: str = "Focus") -> str:
    mins = max(1, parse_spoken_int(minutes, default=5))
    clean_label = (label or "Focus").strip() or "Focus"

    def _timer_worker(duration_sec: int, timer_label: str) -> None:
        time.sleep(duration_sec)
        speaker.play_chime("wake")
        speaker.speak(f"Timer finished: {timer_label} ({mins} minutes).", async_mode=False)

    threading.Thread(target=_timer_worker, args=(mins * 60, clean_label), daemon=True).start()
    return f"Started a {mins} minute {clean_label} timer."


# 15. system_health
@register_tool(
    description="Fetches CPU usage %, RAM usage %, and free disk space using psutil and returns a concise spoken summary.",
    parameters={},
    required=[],
)
def system_health() -> str:
    cpu_pct = psutil.cpu_percent(interval=0.3)
    mem = psutil.virtual_memory()
    disk_root = "C:\\" if os.name == "nt" else "/"
    disk = psutil.disk_usage(disk_root)
    free_gb = disk.free / (1024 ** 3)
    return (
        f"CPU is at {cpu_pct:.0f} percent, RAM is at {mem.percent:.0f} percent, "
        f"with {free_gb:.1f} gigabytes of free disk space."
    )
