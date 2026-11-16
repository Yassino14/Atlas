"""
Atlas - Next-Gen Hybrid Local LLM Router & Tool Dispatcher (brain.py)
Combines:
1. Instant Deterministic Pattern Router (<1ms latency, 100% precision for direct commands)
2. Local Ollama Structured JSON Router (with 15 few-shot examples + auto-daemon startup)
3. Parameter Alias & Spoken-Number Normalizer via tools.py
"""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any, Dict, Optional
import urllib.error
import urllib.request

import config
import tools

_ollama_checked = False


def ensure_ollama_running() -> bool:
    """
    Verify local Ollama server is reachable; if not, automatically launch `ollama serve`
    in the background so the user never gets connection errors.
    """
    global _ollama_checked
    url = f"{config.OLLAMA_BASE_URL.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=2.0) as resp:
            if resp.status == 200:
                _ollama_checked = True
                return True
    except Exception:
        pass

    # Locate ollama binary
    ollama_bin = shutil.which("ollama")
    if not ollama_bin and os.name == "nt":
        candidate = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
        if candidate.exists():
            ollama_bin = str(candidate)

    if not ollama_bin:
        return False

    try:
        print("[Brain] Starting local Ollama background service...")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        subprocess.Popen(
            [ollama_bin, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        for _ in range(10):
            time.sleep(0.5)
            try:
                with urllib.request.urlopen(url, timeout=1.5) as resp:
                    if resp.status == 200:
                        _ollama_checked = True
                        return True
            except Exception:
                continue
    except Exception:
        pass

    return False


def _fast_route_intent(user_text: str) -> Optional[Dict[str, Any]]:
    """
    Stage 1: Zero-latency deterministic intent parser for unambiguous voice commands.
    Returns {"tool_name": ..., "arguments": ...} if matched with high confidence, else None.
    """
    raw = user_text.strip()
    # Strip leading wake phrase if the user spoke wake word + command in one breath
    for prefix in ("wake up", "hey atlas", "ok atlas", "atlas"):
        if raw.lower().startswith(prefix):
            trimmed = raw[len(prefix):].lstrip(" ,.:;!?-")
            if trimmed:
                raw = trimmed
            break

    text = raw.strip().rstrip(".!?")
    lower = text.lower()

    # 15. system_health
    if any(
        phrase in lower
        for phrase in (
            "system health",
            "cpu usage",
            "ram usage",
            "cpu and ram",
            "memory usage",
            "free disk",
            "disk space",
            "how is my system",
            "system status",
            "pc health",
        )
    ):
        return {"tool_name": "system_health", "arguments": {}}

    # 13. screenshot_ocr
    if any(
        phrase in lower
        for phrase in (
            "screenshot",
            "screen ocr",
            "ocr",
            "extract text from screen",
            "extract the text",
            "read my screen",
            "capture screen",
        )
    ):
        return {"tool_name": "screenshot_ocr", "arguments": {}}

    # 4. dictate_to_cursor
    m_dictate = re.match(
        r"^(?:please\s+)?(?:dictate(?:\s+to\s+cursor)?|type\s+at\s+cursor|paste\s+at\s+cursor|write\s+at\s+cursor)[\s,:]+(.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if m_dictate:
        return {"tool_name": "dictate_to_cursor", "arguments": {"text": m_dictate.group(1).strip()}}

    # 1. append_to_file
    m_append_explicit = re.match(
        r"^(?:append|add|write)(?:\s+a\s+note)?\s+(?:to|in)\s+(?:file\s+)?([a-zA-Z0-9_\-]+(?:\.(?:md|txt))?)[\s,:]+(.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if m_append_explicit:
        return {
            "tool_name": "append_to_file",
            "arguments": {
                "filename": m_append_explicit.group(1).strip(),
                "content": m_append_explicit.group(2).strip(),
            },
        }
    m_take_note = re.match(
        r"^(?:take\s+a\s+note|write\s+a\s+note|save\s+a\s+note|add\s+a\s+note|note\s+down)[\s,:]+(.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if m_take_note:
        return {
            "tool_name": "append_to_file",
            "arguments": {"filename": "notes.md", "content": m_take_note.group(1).strip()},
        }

    # 2. summarize_clipboard
    if "clipboard" in lower and any(
        k in lower for k in ("summarize", "summary", "explain", "error", "debug", "what is in", "read")
    ):
        if "save" in lower:
            mode = "save_summary"
        elif any(k in lower for k in ("error", "bug", "log", "trace", "debug", "explain")):
            mode = "error"
        else:
            mode = "summary"
        return {"tool_name": "summarize_clipboard", "arguments": {"mode": mode}}

    # 3. download_media
    if "download" in lower and any(k in lower for k in ("mp3", "mp4", "video", "audio", "media", "youtube", "clipboard", "song")):
        fmt = "mp4" if ("mp4" in lower or "video" in lower) and "mp3" not in lower else "mp3"
        return {"tool_name": "download_media", "arguments": {"format_type": fmt}}

    # 5. save_clipboard_snippet
    if "snippet" in lower and any(k in lower for k in ("save", "store", "create", "clipboard")):
        m_snip = re.search(
            r"(?:named|called|as|name)\s+([a-zA-Z0-9_\-\.]+)(?:\s+(?:in|as|language)\s+([a-zA-Z0-9_\+#]+))?",
            text,
            flags=re.IGNORECASE,
        )
        if m_snip:
            name = m_snip.group(1).strip()
            lang = (m_snip.group(2) or "txt").strip()
            return {"tool_name": "save_clipboard_snippet", "arguments": {"name": name, "language": lang}}

    # 6. organize_folder
    if any(k in lower for k in ("organize", "clean up", "sort files", "tidy up")) and any(
        d in lower for d in ("folder", "directory", "downloads", "desktop", "documents")
    ):
        target = "Downloads"
        for cand in ("downloads", "desktop", "documents"):
            if cand in lower:
                target = cand.capitalize()
                break
        return {"tool_name": "organize_folder", "arguments": {"target_dir": target}}

    # 7. search_files
    m_search = re.match(
        r"^(?:search\s+(?:for\s+)?(?:files?\s+)?(?:for\s+|named\s+|called\s+)?|find\s+(?:file\s+|a\s+file\s+)?(?:named\s+|called\s+)?)(.+?)(?:\s+in\s+(?:my\s+)?([a-zA-Z0-9_:\\/\.\-]+))?(?:\s+folder)?$",
        text,
        flags=re.IGNORECASE,
    )
    if m_search and ("search" in lower or "find file" in lower or "find my" in lower):
        query = m_search.group(1).strip().strip("\"'")
        sdir = (m_search.group(2) or "Documents").strip()
        return {"tool_name": "search_files", "arguments": {"query": query, "search_dir": sdir}}

    # 8. scaffold_project
    if any(k in lower for k in ("scaffold", "create a new project", "create project", "new python project", "new web project", "new node project")):
        stack = "python"
        for s in ("python", "node", "javascript", "web", "rust"):
            if s in lower:
                stack = "node" if s == "javascript" else s
                break
        m_proj = re.search(r"(?:called|named|project)\s+([a-zA-Z0-9_\-]+)", text, flags=re.IGNORECASE)
        if m_proj:
            return {
                "tool_name": "scaffold_project",
                "arguments": {"project_name": m_proj.group(1).strip(), "stack": stack},
            }

    # 9. git_quick_push
    if "git" in lower and "push" in lower:
        m_msg = re.search(r"(?:message|commit)\s+(.+)$", text, flags=re.IGNORECASE)
        msg = m_msg.group(1).strip() if m_msg else "Quick update via Atlas"
        m_repo = re.search(r"\bin\s+([a-zA-Z0-9_:\\/\.\-]+?)(?:\s+with\b|\s+message\b|$)", text, flags=re.IGNORECASE)
        repo = m_repo.group(1).strip() if m_repo else "."
        return {"tool_name": "git_quick_push", "arguments": {"repo_path": repo, "commit_message": msg}}

    # 10. system_control
    if any(
        phrase in lower
        for phrase in ("lock my screen", "lock screen", "lock workstation", "lock the screen", "lock my computer", "lock pc")
    ):
        return {"tool_name": "system_control", "arguments": {"action": "lock"}}
    if "unmute" in lower:
        return {"tool_name": "system_control", "arguments": {"action": "unmute"}}
    if re.search(r"\bmute\b", lower):
        return {"tool_name": "system_control", "arguments": {"action": "mute"}}
    if "volume" in lower:
        val = tools.parse_spoken_int(lower, default=50)
        return {"tool_name": "system_control", "arguments": {"action": "volume", "value": val}}

    # 11. docker_control
    if "docker" in lower or "containers" in lower:
        if any(k in lower for k in ("show", "list", "running", "ps", "status", "what")):
            return {"tool_name": "docker_control", "arguments": {"action": "ps"}}
        m_dock = re.search(r"\b(start|stop)\b.*?(?:container\s+)?([a-zA-Z0-9_\-\.]+)$", text, flags=re.IGNORECASE)
        if m_dock:
            return {
                "tool_name": "docker_control",
                "arguments": {"action": m_dock.group(1).lower(), "container_name": m_dock.group(2).strip()},
            }

    # 12. switch_workspace
    if "workspace" in lower or "switch to coding" in lower or "switch to chill" in lower or "coding mode" in lower or "chill mode" in lower:
        mode = "chill" if any(k in lower for k in ("chill", "relax", "break", "music")) else "coding"
        return {"tool_name": "switch_workspace", "arguments": {"mode": mode}}

    # 14. start_timer
    if "timer" in lower:
        mins = tools.parse_spoken_int(lower, default=5)
        m_label = re.search(r"(?:for|called|named|labeled)\s+([a-zA-Z0-9_ ]+)$", text, flags=re.IGNORECASE)
        label = "Focus"
        if m_label:
            cand_label = m_label.group(1).strip()
            if not re.search(r"\b(?:minute|minutes|min|mins)\b", cand_label.lower()):
                label = cand_label
        return {"tool_name": "start_timer", "arguments": {"minutes": mins, "label": label}}

    return None


def _build_system_prompt() -> str:
    """Build the system prompt listing all 15 registered tools, their schemas, and 15 few-shot examples."""
    catalog_lines = []
    for name, meta in tools.TOOL_REGISTRY.items():
        params = meta["schema"]["function"]["parameters"]
        props = params.get("properties", {})
        req = params.get("required", [])
        param_desc = ", ".join(
            f"{k}: {v.get('type', 'string')}"
            + (f" (enum={v['enum']})" if "enum" in v else "")
            + (" [REQUIRED]" if k in req else " [optional]")
            for k, v in props.items()
        )
        catalog_lines.append(f"- {name}({param_desc}): {meta['description']}")

    catalog_text = "\n".join(catalog_lines)
    return (
        "You are Atlas, a 100% offline local desktop voice assistant.\n"
        "Map the user's voice command to EXACTLY ONE of the 15 registered tools below, "
        "or 'none' only if the command is unrelated to all 15 tools.\n\n"
        "AVAILABLE TOOLS:\n"
        f"{catalog_text}\n\n"
        "FEW-SHOT EXAMPLES:\n"
        'User: "Write a note in ideas.md to build a local agent"\n'
        'JSON: {"tool_name": "append_to_file", "arguments": {"filename": "ideas.md", "content": "build a local agent"}, "spoken_fallback": ""}\n'
        'User: "Summarize my clipboard"\n'
        'JSON: {"tool_name": "summarize_clipboard", "arguments": {"mode": "summary"}, "spoken_fallback": ""}\n'
        'User: "Download this video from clipboard as mp3"\n'
        'JSON: {"tool_name": "download_media", "arguments": {"format_type": "mp3"}, "spoken_fallback": ""}\n'
        'User: "Dictate to cursor hello everyone how are you"\n'
        'JSON: {"tool_name": "dictate_to_cursor", "arguments": {"text": "hello everyone how are you"}, "spoken_fallback": ""}\n'
        'User: "Save clipboard snippet called auth_helper in python"\n'
        'JSON: {"tool_name": "save_clipboard_snippet", "arguments": {"name": "auth_helper", "language": "python"}, "spoken_fallback": ""}\n'
        'User: "Clean up my Downloads folder"\n'
        'JSON: {"tool_name": "organize_folder", "arguments": {"target_dir": "Downloads"}, "spoken_fallback": ""}\n'
        'User: "Find file invoice in Documents"\n'
        'JSON: {"tool_name": "search_files", "arguments": {"query": "invoice", "search_dir": "Documents"}, "spoken_fallback": ""}\n'
        'User: "Scaffold a python project called weather_app"\n'
        'JSON: {"tool_name": "scaffold_project", "arguments": {"project_name": "weather_app", "stack": "python"}, "spoken_fallback": ""}\n'
        'User: "Push git changes in D:/Atlas with message initial commit"\n'
        'JSON: {"tool_name": "git_quick_push", "arguments": {"repo_path": "D:/Atlas", "commit_message": "initial commit"}, "spoken_fallback": ""}\n'
        'User: "Set system volume to 65 percent"\n'
        'JSON: {"tool_name": "system_control", "arguments": {"action": "volume", "value": 65}, "spoken_fallback": ""}\n'
        'User: "Show running docker containers"\n'
        'JSON: {"tool_name": "docker_control", "arguments": {"action": "ps", "container_name": ""}, "spoken_fallback": ""}\n'
        'User: "Switch workspace to coding"\n'
        'JSON: {"tool_name": "switch_workspace", "arguments": {"mode": "coding"}, "spoken_fallback": ""}\n'
        'User: "Take a screenshot and extract text"\n'
        'JSON: {"tool_name": "screenshot_ocr", "arguments": {}, "spoken_fallback": ""}\n'
        'User: "Set a 10 minute timer for workout"\n'
        'JSON: {"tool_name": "start_timer", "arguments": {"minutes": 10, "label": "workout"}, "spoken_fallback": ""}\n'
        'User: "Check my system health"\n'
        'JSON: {"tool_name": "system_health", "arguments": {}, "spoken_fallback": ""}\n\n'
        "Respond strictly in JSON matching the schema."
    )


ROUTING_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "tool_name": {
            "type": "string",
            "enum": list(tools.TOOL_REGISTRY.keys()) + ["none"],
        },
        "arguments": {
            "type": "object",
        },
        "spoken_fallback": {
            "type": "string",
        },
    },
    "required": ["tool_name", "arguments"],
}


def query_local_llm_text(prompt: str, system_prompt: str = "You are a concise assistant.") -> str:
    """
    Send a plain-text generation request to local Ollama (used by summarize_clipboard and dictate_to_cursor).
    """
    if not _ollama_checked:
        ensure_ollama_running()

    url = f"{config.OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    payload = {
        "model": config.OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": 0.2},
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=config.OLLAMA_TIMEOUT_SEC) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("message", {}).get("content", "").strip()
    except urllib.error.URLError as exc:
        return f"Could not reach local Ollama server at {config.OLLAMA_BASE_URL}: {exc}"
    except Exception as exc:
        return f"Local LLM error: {exc}"


class LocalBrain:
    """
    Hybrid Two-Stage Router:
      Stage 1: Instant deterministic intent matching (<1ms)
      Stage 2: Local Ollama structured JSON tool-calling with few-shot examples
    """

    def __init__(self, model: str = config.OLLAMA_MODEL):
        self.model = model
        self.base_url = config.OLLAMA_BASE_URL.rstrip("/")
        ensure_ollama_running()

    def route_command(self, user_text: str) -> Dict[str, Any]:
        """
        Route `user_text` to one of the 15 registered tools and extract arguments.
        """
        # Stage 1: Instant Deterministic Router
        fast_match = _fast_route_intent(user_text)
        if fast_match is not None:
            return {
                "tool_name": fast_match["tool_name"],
                "arguments": fast_match["arguments"],
                "spoken_fallback": "",
                "router": "fast_path",
            }

        # Stage 2: Local Ollama Structured JSON Router
        if not _ollama_checked:
            ensure_ollama_running()

        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "stream": False,
            "format": ROUTING_JSON_SCHEMA,
            "messages": [
                {"role": "system", "content": _build_system_prompt()},
                {"role": "user", "content": user_text},
            ],
            "options": {
                "temperature": 0.0,
            },
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=config.OLLAMA_TIMEOUT_SEC) as resp:
                raw_resp = json.loads(resp.read().decode("utf-8"))

            message = raw_resp.get("message", {})

            tool_calls = message.get("tool_calls")
            if tool_calls and isinstance(tool_calls, list):
                first_call = tool_calls[0].get("function", {})
                return {
                    "tool_name": first_call.get("name", "none"),
                    "arguments": first_call.get("arguments", {}) or {},
                    "spoken_fallback": "",
                    "router": "ollama_tool_call",
                }

            content = message.get("content", "{}").strip()
            parsed = json.loads(content)
            return {
                "tool_name": str(parsed.get("tool_name", "none")),
                "arguments": parsed.get("arguments") or {},
                "spoken_fallback": str(parsed.get("spoken_fallback", "")),
                "router": "ollama_json",
            }

        except urllib.error.URLError:
            return {
                "tool_name": "none",
                "arguments": {},
                "spoken_fallback": "Local Ollama is not running. Start Ollama to process commands.",
            }
        except Exception as exc:
            print(f"[Brain] Routing error: {exc}")
            return {
                "tool_name": "none",
                "arguments": {},
                "spoken_fallback": "Sorry, I could not parse that command.",
            }

    def process_and_execute(self, user_text: str) -> str:
        """
        Full pipeline: Route user command -> Validate & execute registered tool -> Return spoken confirmation.
        """
        if not user_text or not user_text.strip():
            return "I didn't catch that."

        decision = self.route_command(user_text)
        tool_name: Optional[str] = decision.get("tool_name")
        arguments: Dict[str, Any] = decision.get("arguments") or {}
        router_type: str = decision.get("router", "llm")

        print(f"[Brain] 🧠 Selected Tool: {tool_name} ({router_type}) | Args: {json.dumps(arguments)}")

        if not tool_name or tool_name == "none":
            fallback = decision.get("spoken_fallback") or "I'm not sure which tool to run for that."
            return fallback

        result = tools.execute_registered_tool(tool_name, arguments)
        return result
