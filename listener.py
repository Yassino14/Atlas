"""
Atlas - Dual Wake Engine (listener.py)
Monitors microphone stream continuously with ultra-low CPU overhead for:
1. Double-Clap Detector: 1kHz-4kHz bandpass transient detector with a 0.2s-0.8s inter-clap window,
   rejecting continuous speech and background noise.
2. Offline Wake Word Detector: Keyword spotting for "wake up" via Vosk or openWakeWord.
"""

from collections import deque
import json
import queue
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import config

try:
    from scipy.signal import butter, sosfilt
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


class DoubleClapDetector:
    """
    Real-time 1kHz-4kHz bandpass transient detector that isolates sharp percussive claps
    while ignoring continuous speech or ambient noise.
    """

    def __init__(self, sample_rate: int = config.SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.low_hz = config.CLAP_BANDPASS_LOW_HZ
        self.high_hz = config.CLAP_BANDPASS_HIGH_HZ
        self.rms_threshold = config.CLAP_RMS_THRESHOLD
        self.peak_to_avg_ratio = config.CLAP_PEAK_TO_AVG_RATIO
        self.max_spike_sec = config.CLAP_MAX_SPIKE_MS / 1000.0
        self.window_min = config.CLAP_WINDOW_MIN_SEC
        self.window_max = config.CLAP_WINDOW_MAX_SEC
        self.cooldown_sec = config.CLAP_COOLDOWN_SEC

        if _HAS_SCIPY:
            nyq = 0.5 * self.sample_rate
            low = max(0.01, min(self.low_hz / nyq, 0.98))
            high = max(low + 0.05, min(self.high_hz / nyq, 0.99))
            self._sos = butter(4, [low, high], btype="bandpass", output="sos")
        else:
            self._sos = None

        # Rolling history of bandpass RMS values (~1 second of 30ms frames)
        self._rms_history: deque = deque(maxlen=35)
        self._in_spike: bool = False
        self._spike_start_time: float = 0.0
        self._spike_peak_rms: float = 0.0
        self._clap_timestamps: deque = deque(maxlen=4)
        self._last_trigger_time: float = -10.0

    def bandpass_filter(self, audio_float32: np.ndarray) -> np.ndarray:
        """Apply 1kHz-4kHz bandpass filter using Butterworth SOS or FFT fallback."""
        if audio_float32.size == 0:
            return audio_float32

        if self._sos is not None:
            return sosfilt(self._sos, audio_float32)

        # Pure NumPy FFT bandpass fallback (1kHz - 4kHz)
        freqs = np.fft.rfftfreq(audio_float32.size, d=1.0 / self.sample_rate)
        spectrum = np.fft.rfft(audio_float32)
        mask = (freqs >= self.low_hz) & (freqs <= self.high_hz)
        spectrum[~mask] = 0
        return np.fft.irfft(spectrum, n=audio_float32.size).astype(np.float32)

    def process_frame(self, pcm_int16: np.ndarray, now: Optional[float] = None) -> bool:
        """
        Process a single audio frame (int16) and return True if a valid double-clap occurred.
        """
        if not config.CLAP_DETECTION_ENABLED:
            return False

        if now is None:
            now = time.monotonic()

        if now - self._last_trigger_time < self.cooldown_sec:
            return False

        audio_f32 = pcm_int16.astype(np.float32) / 32768.0
        filtered = self.bandpass_filter(audio_f32)
        rms = float(np.sqrt(np.mean(np.square(filtered)) + 1e-12))

        baseline_rms = (
            float(np.median(self._rms_history)) if len(self._rms_history) >= 5 else 0.005
        )
        baseline_rms = max(baseline_rms, 0.002)

        is_above_threshold = (
            rms >= self.rms_threshold
            and (rms / baseline_rms) >= self.peak_to_avg_ratio
        )

        triggered = False

        if is_above_threshold:
            if not self._in_spike:
                self._in_spike = True
                self._spike_start_time = now
                self._spike_peak_rms = rms
            else:
                self._spike_peak_rms = max(self._spike_peak_rms, rms)
        else:
            if self._in_spike:
                spike_duration = now - self._spike_start_time
                self._in_spike = False

                # Only register sharp percussive transients; reject sustained speech syllables
                if spike_duration <= self.max_spike_sec:
                    triggered = self._register_clap(self._spike_start_time)

            # Update ambient baseline only during non-spike frames
            self._rms_history.append(rms)

        return triggered

    def _register_clap(self, clap_time: float) -> bool:
        """Record a valid clap transient and check if two claps fall within [0.2s, 0.8s]."""
        # Prune stale claps older than window_max
        while self._clap_timestamps and (clap_time - self._clap_timestamps[0]) > self.window_max:
            self._clap_timestamps.popleft()

        if self._clap_timestamps:
            delta = clap_time - self._clap_timestamps[-1]
            if self.window_min <= delta <= self.window_max:
                self._clap_timestamps.clear()
                self._last_trigger_time = clap_time
                return True
            elif delta < self.window_min:
                # Echo/reverb of the same clap; ignore
                return False

        self._clap_timestamps.append(clap_time)
        return False


class WakeWordDetector:
    """
    Offline wake-word spotter supporting Vosk (default) or openWakeWord.
    Detects 'wake up' (and optional configured aliases).
    """

    def __init__(self, sample_rate: int = config.SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.engine_type = config.WAKE_WORD_ENGINE.lower()
        self.wake_phrases = [p.lower().strip() for p in config.WAKE_PHRASES]
        self._vosk_recognizer = None
        self._oww_model = None
        self.ready = False

        if not config.WAKE_WORD_ENABLED:
            return

        if self.engine_type == "vosk":
            self._init_vosk()
        elif self.engine_type == "openwakeword":
            self._init_openwakeword()
        else:
            self._init_vosk()

    def _init_vosk(self) -> None:
        try:
            from vosk import KaldiRecognizer, Model, SetLogLevel

            SetLogLevel(-1)
            model_path = Path(config.VOSK_MODEL_PATH)
            if not model_path.exists():
                print(
                    f"[Listener] Vosk model not found at {model_path}. "
                    "Run `python main.py --download-models` or use Double-Clap wake."
                )
                return

            model = Model(str(model_path))
            grammar = json.dumps(self.wake_phrases + ["[unk]"])
            self._vosk_recognizer = KaldiRecognizer(model, self.sample_rate, grammar)
            self.ready = True
            print(f"[Listener] Vosk wake-word engine active (phrases: {self.wake_phrases})")
        except ImportError:
            print("[Listener] `vosk` package not installed; wake-word disabled (Double-Clap active).")
        except Exception as exc:
            print(f"[Listener] Failed to initialize Vosk wake word: {exc}")

    def _init_openwakeword(self) -> None:
        try:
            from openwakeword.model import Model as OWWModel

            self._oww_model = OWWModel(
                wakeword_models=[config.OPENWAKEWORD_MODEL],
                inference_framework="onnx",
            )
            self.ready = True
            print(f"[Listener] openWakeWord engine active ({config.OPENWAKEWORD_MODEL})")
        except ImportError:
            print("[Listener] `openwakeword` not installed; trying Vosk fallback...")
            self._init_vosk()
        except Exception as exc:
            print(f"[Listener] Failed to initialize openWakeWord: {exc}")

    def process_frame(self, pcm_int16: np.ndarray) -> bool:
        """Process a frame of 16kHz 16-bit mono PCM audio and return True on wake word match."""
        if not self.ready:
            return False

        if self._vosk_recognizer is not None:
            raw_bytes = pcm_int16.tobytes()
            if self._vosk_recognizer.AcceptWaveform(raw_bytes):
                res = json.loads(self._vosk_recognizer.Result())
                text = res.get("text", "").lower().strip()
            else:
                partial = json.loads(self._vosk_recognizer.PartialResult())
                text = partial.get("partial", "").lower().strip()

            if text and any(phrase in text for phrase in self.wake_phrases):
                self._vosk_recognizer.Reset()
                return True

        elif self._oww_model is not None:
            prediction = self._oww_model.predict(pcm_int16)
            for _model_name, score in prediction.items():
                if float(score) >= config.OPENWAKEWORD_THRESHOLD:
                    self._oww_model.reset()
                    return True

        return False


def download_vosk_small_model() -> Path:
    """
    Helper to download the official lightweight Vosk English model (~40MB)
    into `models/vosk-model-small-en-us-0.15` for 100% offline wake word detection.
    """
    target_dir = Path(config.VOSK_MODEL_PATH)
    if target_dir.exists():
        print(f"[Setup] Vosk model already exists at: {target_dir}")
        return target_dir

    url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    zip_path = config.MODELS_DIR / "vosk-model-small-en-us-0.15.zip"
    print(f"[Setup] Downloading lightweight Vosk wake-word model from {url} ...")
    urllib.request.urlretrieve(url, zip_path)
    print("[Setup] Extracting Vosk model...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(config.MODELS_DIR)
    zip_path.unlink(missing_ok=True)
    print(f"[Setup] Vosk model installed at: {target_dir}")
    return target_dir


class DualWakeListener:
    """
    Unified microphone stream manager combining:
    - DoubleClapDetector (1kHz-4kHz bandpass transient detector)
    - WakeWordDetector ("wake up" offline spotter)
    Supports sounddevice (primary) with PyAudio fallback.
    """

    def __init__(self):
        self.clap_detector = DoubleClapDetector(config.SAMPLE_RATE)
        self.wake_word_detector = WakeWordDetector(config.SAMPLE_RATE)
        self._audio_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=150)

    def wait_for_wake(self, stop_event=None) -> Tuple[bool, str]:
        """
        Block in low-CPU streaming mode until either a Double-Clap or Wake Word is detected.
        Returns (True, trigger_source) where trigger_source is "double_clap" or "wake_word".
        """
        # Drain any stale frames
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        try:
            import sounddevice as sd

            def _sd_callback(indata, frames, time_info, status):
                pcm = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
                try:
                    self._audio_queue.put_nowait(pcm)
                except queue.Full:
                    pass

            with sd.InputStream(
                samplerate=config.SAMPLE_RATE,
                blocksize=config.BLOCK_SIZE,
                device=config.INPUT_DEVICE_INDEX,
                channels=config.CHANNELS,
                dtype="int16",
                callback=_sd_callback,
            ):
                return self._consume_loop(stop_event)

        except ImportError:
            return self._wait_with_pyaudio(stop_event)

    def _wait_with_pyaudio(self, stop_event=None) -> Tuple[bool, str]:
        """Fallback stream reader using PyAudio if sounddevice is unavailable."""
        import pyaudio

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
            while stop_event is None or not stop_event.is_set():
                raw = stream.read(config.BLOCK_SIZE, exception_on_overflow=False)
                pcm = np.frombuffer(raw, dtype=np.int16)
                if self.clap_detector.process_frame(pcm):
                    return True, "double_clap"
                if self.wake_word_detector.process_frame(pcm):
                    return True, "wake_word"
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()
        return False, "stopped"

    def _consume_loop(self, stop_event=None) -> Tuple[bool, str]:
        """Process incoming 30ms audio frames from the microphone queue (plus optional Space/Enter key trigger)."""
        has_msvcrt = False
        try:
            import msvcrt
            has_msvcrt = True
        except ImportError:
            msvcrt = None  # type: ignore

        while stop_event is None or not stop_event.is_set():
            if has_msvcrt and msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch in (b" ", b"\r", b"\n"):
                    return True, "keyboard_push_to_talk"

            try:
                pcm = self._audio_queue.get(timeout=0.15)
            except queue.Empty:
                continue

            if self.clap_detector.process_frame(pcm):
                return True, "double_clap"

            if self.wake_word_detector.process_frame(pcm):
                return True, "wake_word"

        return False, "stopped"
