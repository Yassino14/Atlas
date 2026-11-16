"""
Atlas - Speech-to-Text & Voice Activity Detection Engine (transcriber.py)
Records microphone audio with automatic end-of-speech detection (WebRTC VAD / Silero VAD / Energy fallback)
and transcribes locally using faster-whisper (base.en or small.en).
"""

from collections import deque
import queue
import time
from typing import List, Optional

import numpy as np

import config


class VoiceActivityDetector:
    """
    Hybrid Voice Activity Detector supporting:
    1. webrtcvad (fast C-based 16kHz 30ms frame classifier)
    2. silero-vad (if selected)
    3. RMS energy + zero-crossing fallback
    """

    def __init__(self, sample_rate: int = config.SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.engine = config.VAD_ENGINE.lower()
        self._webrtc_vad = None
        self._silero_model = None

        if self.engine == "webrtcvad":
            try:
                import webrtcvad

                self._webrtc_vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)
            except ImportError:
                print("[Transcriber] `webrtcvad` not installed; using energy VAD fallback.")
                self.engine = "energy"
        elif self.engine == "silero":
            try:
                import torch

                model, _ = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    force_reload=False,
                    trust_repo=True,
                )
                self._silero_model = model
            except Exception as exc:
                print(f"[Transcriber] Silero VAD unavailable ({exc}); falling back to WebRTC/energy VAD.")
                try:
                    import webrtcvad

                    self._webrtc_vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)
                    self.engine = "webrtcvad"
                except ImportError:
                    self.engine = "energy"

    def is_speech(self, pcm_int16: np.ndarray) -> bool:
        """Return True if the 30ms 16kHz mono frame contains human speech."""
        if pcm_int16.size == 0:
            return False

        if self._webrtc_vad is not None:
            try:
                return bool(self._webrtc_vad.is_speech(pcm_int16.tobytes(), self.sample_rate))
            except Exception:
                pass

        if self._silero_model is not None:
            try:
                import torch

                audio_f32 = pcm_int16.astype(np.float32) / 32768.0
                # Silero expects 512 samples @ 16kHz
                chunk = audio_f32[:512] if audio_f32.size >= 512 else np.pad(audio_f32, (0, 512 - audio_f32.size))
                prob = float(self._silero_model(torch.from_numpy(chunk), self.sample_rate).item())
                return prob >= 0.5
            except Exception:
                pass

        # Fallback RMS energy detector
        audio_f32 = pcm_int16.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(np.square(audio_f32)) + 1e-12))
        return rms >= config.VAD_ENERGY_THRESHOLD


class Transcriber:
    """
    Captures spoken commands until silence is detected and transcribes them offline
    using faster-whisper (`base.en` or `small.en`).
    """

    def __init__(self):
        self.vad = VoiceActivityDetector(config.SAMPLE_RATE)
        self._whisper_model = None

    def _ensure_model_loaded(self) -> None:
        if self._whisper_model is not None:
            return

        from faster_whisper import WhisperModel

        device = config.WHISPER_DEVICE
        compute_type = config.WHISPER_COMPUTE_TYPE

        if device == "auto":
            try:
                import ctranslate2

                if ctranslate2.get_cuda_device_count() > 0:
                    device = "cuda"
                    # Older NVIDIA GPUs (GTX 1650/1060, etc.) do not support fast native float16 in ctranslate2.
                    # We try float16 first; if unsupported, gracefully fall back to float32 or int8.
                    compute_type = "float16"
                else:
                    device = "cpu"
                    compute_type = "int8"
            except Exception:
                device = "cpu"
                compute_type = "int8"

        print(
            f"[Transcriber] Loading faster-whisper model '{config.WHISPER_MODEL_SIZE}' "
            f"on {device} ({compute_type})..."
        )
        try:
            self._whisper_model = WhisperModel(
                config.WHISPER_MODEL_SIZE,
                device=device,
                compute_type=compute_type,
                download_root=str(config.MODELS_DIR),
            )
            if device == "cuda":
                # Verify CUDA libraries (e.g. cublas64_12.dll) actually load during encode
                dummy = np.zeros(8000, dtype=np.float32)
                segs, _ = self._whisper_model.transcribe(dummy, language="en")
                list(segs)
        except (ValueError, RuntimeError) as err:
            err_str = str(err).lower()
            if "float16" in err_str:
                try:
                    self._whisper_model = WhisperModel(
                        config.WHISPER_MODEL_SIZE,
                        device=device,
                        compute_type="float32",
                        download_root=str(config.MODELS_DIR),
                    )
                    dummy = np.zeros(8000, dtype=np.float32)
                    segs, _ = self._whisper_model.transcribe(dummy, language="en")
                    list(segs)
                except Exception:
                    print("[Transcriber] CUDA DLLs unavailable; switching faster-whisper to CPU (int8)...")
                    self._whisper_model = WhisperModel(
                        config.WHISPER_MODEL_SIZE,
                        device="cpu",
                        compute_type="int8",
                        download_root=str(config.MODELS_DIR),
                    )
            else:
                print(f"[Transcriber] CUDA runtime unavailable ({err}); switching faster-whisper to CPU (int8)...")
                self._whisper_model = WhisperModel(
                    config.WHISPER_MODEL_SIZE,
                    device="cpu",
                    compute_type="int8",
                    download_root=str(config.MODELS_DIR),
                )
        print("[Transcriber] faster-whisper ready.")

    def record_until_silence(self) -> Optional[np.ndarray]:
        """
        Record 16kHz mono audio from the microphone.
        Automatically stops when the user finishes speaking (VAD silence threshold reached).
        Returns float32 audio normalized to [-1.0, 1.0], or None if no speech was detected.
        """
        frame_duration_sec = config.BLOCK_DURATION_MS / 1000.0
        pre_roll_frames = max(1, int(0.30 / frame_duration_sec))
        silence_frames_limit = max(1, int(config.VAD_SILENCE_STOP_SEC / frame_duration_sec))
        min_speech_frames = max(2, int(0.18 / frame_duration_sec))

        pre_roll: deque = deque(maxlen=pre_roll_frames)
        recorded_frames: List[np.ndarray] = []

        speech_started = False
        speech_frame_count = 0
        trailing_silence_frames = 0
        start_time = time.monotonic()

        audio_q: "queue.Queue[np.ndarray]" = queue.Queue()

        try:
            import sounddevice as sd

            def _callback(indata, frames, time_info, status):
                pcm = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
                audio_q.put(pcm)

            with sd.InputStream(
                samplerate=config.SAMPLE_RATE,
                blocksize=config.BLOCK_SIZE,
                device=config.INPUT_DEVICE_INDEX,
                channels=config.CHANNELS,
                dtype="int16",
                callback=_callback,
            ):
                print("[Transcriber] 🎙️ Listening for command...")
                while True:
                    elapsed = time.monotonic() - start_time
                    if elapsed > config.VAD_MAX_RECORD_SEC:
                        print("[Transcriber] Max recording duration reached.")
                        break

                    if not speech_started and elapsed > config.VAD_SPEECH_START_TIMEOUT_SEC:
                        print("[Transcriber] No speech detected within timeout.")
                        return None

                    try:
                        frame = audio_q.get(timeout=0.25)
                    except queue.Empty:
                        continue

                    active = self.vad.is_speech(frame)

                    if not speech_started:
                        pre_roll.append(frame)
                        if active:
                            speech_frame_count += 1
                            if speech_frame_count >= min_speech_frames:
                                speech_started = True
                                recorded_frames.extend(pre_roll)
                                trailing_silence_frames = 0
                        else:
                            speech_frame_count = 0
                    else:
                        recorded_frames.append(frame)
                        if active:
                            trailing_silence_frames = 0
                        else:
                            trailing_silence_frames += 1
                            if trailing_silence_frames >= silence_frames_limit:
                                break

        except ImportError:
            return self._record_with_pyaudio(
                pre_roll_frames, silence_frames_limit, min_speech_frames
            )

        if not recorded_frames:
            return None

        full_pcm = np.concatenate(recorded_frames)
        return full_pcm.astype(np.float32) / 32768.0

    def _record_with_pyaudio(
        self, pre_roll_frames: int, silence_frames_limit: int, min_speech_frames: int
    ) -> Optional[np.ndarray]:
        """Fallback recorder using PyAudio."""
        import pyaudio

        pre_roll: deque = deque(maxlen=pre_roll_frames)
        recorded_frames: List[np.ndarray] = []
        speech_started = False
        speech_frame_count = 0
        trailing_silence_frames = 0
        start_time = time.monotonic()

        pa = pyaudio.PyAudio()
        stream = pa.open(
            rate=config.SAMPLE_RATE,
            channels=config.CHANNELS,
            format=pyaudio.paInt16,
            input=True,
            input_device_index=config.INPUT_DEVICE_INDEX,
            frames_per_buffer=config.BLOCK_SIZE,
        )
        try:
            print("[Transcriber] 🎙️ Listening for command (PyAudio)...")
            while True:
                elapsed = time.monotonic() - start_time
                if elapsed > config.VAD_MAX_RECORD_SEC:
                    break
                if not speech_started and elapsed > config.VAD_SPEECH_START_TIMEOUT_SEC:
                    return None

                raw = stream.read(config.BLOCK_SIZE, exception_on_overflow=False)
                frame = np.frombuffer(raw, dtype=np.int16)
                active = self.vad.is_speech(frame)

                if not speech_started:
                    pre_roll.append(frame)
                    if active:
                        speech_frame_count += 1
                        if speech_frame_count >= min_speech_frames:
                            speech_started = True
                            recorded_frames.extend(pre_roll)
                            trailing_silence_frames = 0
                    else:
                        speech_frame_count = 0
                else:
                    recorded_frames.append(frame)
                    if active:
                        trailing_silence_frames = 0
                    else:
                        trailing_silence_frames += 1
                        if trailing_silence_frames >= silence_frames_limit:
                            break
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

        if not recorded_frames:
            return None

        full_pcm = np.concatenate(recorded_frames)
        return full_pcm.astype(np.float32) / 32768.0

    # Domain-biased prompt so Whisper accurately recognizes technical tool terms (OCR, Docker, git, mp3, etc.)
    WHISPER_DOMAIN_PROMPT = (
        "Atlas desktop voice commands: append to file notes.md todo.txt, summarize clipboard, "
        "explain error in clipboard, download media mp3 mp4, dictate to cursor, "
        "save clipboard snippet in python javascript, organize folder Downloads Desktop Documents, "
        "search files for, scaffold project python node web, git quick push with commit message, "
        "system control set volume percent mute unmute lock screen, docker containers ps start stop, "
        "switch workspace to coding or chill, screenshot OCR extract text, start timer minutes Focus, "
        "check system health CPU and RAM usage."
    )

    # Common Whisper hallucinations produced on near-silent background noise
    _HALLUCINATION_PHRASES = {
        "thank you",
        "thank you.",
        "thanks for watching",
        "thanks for watching.",
        "you",
        "bye",
        "bye.",
        "subtitles by",
    }

    def transcribe_audio(self, audio_f32: np.ndarray) -> str:
        """Transcribe a float32 16kHz audio array into English text via faster-whisper."""
        if audio_f32 is None or audio_f32.size == 0:
            return ""

        self._ensure_model_loaded()
        try:
            segments, _info = self._whisper_model.transcribe(
                audio_f32,
                language="en",
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False,
                initial_prompt=self.WHISPER_DOMAIN_PROMPT,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
        except RuntimeError as err:
            print(f"[Transcriber] Runtime error ({err}); switching to CPU int8 and retrying...")
            from faster_whisper import WhisperModel

            self._whisper_model = WhisperModel(
                config.WHISPER_MODEL_SIZE,
                device="cpu",
                compute_type="int8",
                download_root=str(config.MODELS_DIR),
            )
            segments, _info = self._whisper_model.transcribe(
                audio_f32,
                language="en",
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False,
                initial_prompt=self.WHISPER_DOMAIN_PROMPT,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()

        if text.lower().strip(".!? ") in self._HALLUCINATION_PHRASES:
            rms = float(np.sqrt(np.mean(np.square(audio_f32)) + 1e-12))
            if rms < 0.03:
                return ""

        return text

    def listen_and_transcribe(self) -> str:
        """Record speech until silence and return the transcribed text."""
        audio = self.record_until_silence()
        if audio is None:
            return ""
        text = self.transcribe_audio(audio)
        if text:
            print(f"[Transcriber] 📝 Heard: \"{text}\"")
        return text

