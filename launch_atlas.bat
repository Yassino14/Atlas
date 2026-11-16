@echo off
title Atlas - Offline Voice & Acoustic Desktop Agent
cd /d "%~dp0"
set "PATH=%PATH%;%LOCALAPPDATA%\Programs\Ollama"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py %*
) else (
    python main.py %*
)
pause

