"""
Atlas - 100% Free, Offline, Local-First Voice & Acoustic Desktop Agent (main.py)

Execution State Loop:
  Idle Listening (Wake Word "wake up" / Double-Clap 1kHz-4kHz transient)
  -> Play subtle chime / TTS "Listening"
  -> Record until silence (WebRTC / Silero / Energy VAD)
  -> Transcribe (faster-whisper local)
  -> LLM Tool Selection (Ollama llama3.2 / qwen2.5:7b structured JSON)
  -> Execute Validated Tool (15 registered tools in tools.py)
  -> Speak short confirmation (pyttsx3 / piper-tts)
  -> Return to Idle
"""

import argparse
import signal
import sys
import threading

import brain
import config
import listener
import speaker
import tools
import transcriber


def print_banner() -> None:
    print("=" * 68)
    print("  ATLAS - 100% Offline Voice & Acoustic Desktop Agent")
    print("=" * 68)
    print(f"  LLM Model      : {config.OLLAMA_MODEL} ({config.OLLAMA_BASE_URL})")
    print(f"  Whisper Model  : {config.WHISPER_MODEL_SIZE} ({config.WHISPER_COMPUTE_TYPE})")
    print(f"  Wake Triggers  : Double-Clap (1-4kHz, {config.CLAP_WINDOW_MIN_SEC}-{config.CLAP_WINDOW_MAX_SEC}s) + Wake Word {config.WAKE_PHRASES}")
    print(f"  Loaded Tools   : {len(tools.TOOL_REGISTRY)} registered functions")
    print("=" * 68)


def list_audio_devices() -> None:
    """List available microphone input devices."""
    try:
        import sounddevice as sd

        print("\nAvailable Audio Devices (sounddevice):")
        print(sd.query_devices())
    except ImportError:
        try:
            import pyaudio

            pa = pyaudio.PyAudio()
            print("\nAvailable Audio Input Devices (PyAudio):")
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info.get("maxInputChannels", 0) > 0:
                    print(f"  [{i}] {info.get('name')} (rate={int(info.get('defaultSampleRate', 16000))})")
            pa.terminate()
        except ImportError:
            print("Neither sounddevice nor pyaudio is installed yet.")


def run_text_mode(router: brain.LocalBrain) -> None:
    """Interactive CLI mode for testing LLM tool routing without a microphone."""
    print("\n[Atlas] Interactive Text Mode active. Type a command (or 'exit' to quit).")
    while True:
        try:
            user_input = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Atlas.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            break

        confirmation = router.process_and_execute(user_input)
        speaker.speak(confirmation)


def run_voice_loop() -> None:
    """Main continuous offline voice & acoustic state loop."""
    print_banner()

    stop_event = threading.Event()

    def _sigint_handler(sig, frame):
        print("\n[Atlas] Shutting down gracefully...")
        stop_event.set()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint_handler)

    wake_listener = listener.DualWakeListener()
    stt = transcriber.Transcriber()
    stt._ensure_model_loaded()
    router = brain.LocalBrain()

    speaker.speak("Atlas online and listening.")

    while not stop_event.is_set():
        print("\n[Atlas] 💤 Idle — Say 'wake up', Double-Clap, or press Space/Enter...")
        triggered, source = wake_listener.wait_for_wake(stop_event)
        if not triggered or stop_event.is_set():
            break

        print(f"[Atlas] ⚡ Wake triggered via: {source}")

        # 1. Subtle chime + "Listening" confirmation
        speaker.play_chime("wake")
        if config.SPEAK_LISTENING_ON_WAKE:
            speaker.speak("Listening")

        # 2. Record until silence & transcribe locally via faster-whisper
        user_text = stt.listen_and_transcribe()
        if not user_text:
            speaker.play_chime("done")
            speaker.speak("No command heard.")
            continue

        # Allow voice shutdown
        if user_text.lower().strip(".!? ") in ("stop listening", "shut down atlas", "exit atlas", "goodbye atlas"):
            speaker.speak("Shutting down Atlas. Goodbye.")
            break

        # 3. Route via local Ollama & execute registered tool
        confirmation = router.process_and_execute(user_text)

        # 4. Speak concise confirmation & return to Idle
        speaker.play_chime("done")
        speaker.speak(confirmation)


def main() -> None:
    parser = argparse.ArgumentParser(description="Atlas - Offline Voice & Acoustic Desktop Agent")
    parser.add_argument(
        "--text",
        action="store_true",
        help="Run in interactive text input mode instead of microphone listening",
    )
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Download offline Vosk small model and pre-cache faster-whisper model",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit",
    )
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    if args.download_models:
        listener.download_vosk_small_model()
        stt = transcriber.Transcriber()
        stt._ensure_model_loaded()
        print("[Setup] All offline speech models are ready!")
        return

    if args.text:
        print_banner()
        router = brain.LocalBrain()
        run_text_mode(router)
        return

    run_voice_loop()


if __name__ == "__main__":
    main()

