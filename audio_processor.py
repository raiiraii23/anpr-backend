import os
import numpy as np
import time
import threading
import logging
from scipy.signal import butter, sosfiltfilt

logger = logging.getLogger(__name__)

MUFFLER_FREQ_LOW  = 50
MUFFLER_FREQ_HIGH = 1000

# ── Calibration ────────────────────────────────────────────────────────────────
# DB_CALIBRATION_OFFSET: maps 0 dBFS → dB SPL.
#   94  = IEC standard (calibrated measurement mic; most physically accurate).
#   105 = typical laptop/USB built-in mic (~11 dB less sensitive than SLM).
#   115 = distant or very insensitive mic.
#   Raise if readings look consistently too low for your microphone.
DB_CALIBRATION_OFFSET = float(os.getenv("DB_CALIBRATION_OFFSET", "94.0"))

# MIC_GAIN_DB: additive dB gain applied AFTER the RMS→dB conversion.
#   0   = no boost (default, readings match the calibration offset)
#   +6  = double apparent loudness (effectively shifts the whole scale up 6 dB)
#   Use this to fine-tune without touching the offset.
MIC_GAIN_DB = float(os.getenv("MIC_GAIN_DB", "0.0"))


def butter_bandpass_sos(lowcut: float, highcut: float, fs: float, order: int = 4):
    nyq  = 0.5 * fs
    low  = max(1e-6, lowcut / nyq)
    high = min(0.9999, highcut / nyq)
    return butter(order, [low, high], btype="band", output="sos")


def _rms_to_db_spl(audio: np.ndarray) -> float:
    """float32 [-1,1] array → dB SPL."""
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if rms <= 1e-9:
        return 0.0
    dbfs   = 20.0 * np.log10(rms)
    db_spl = dbfs + DB_CALIBRATION_OFFSET + MIC_GAIN_DB
    return float(np.clip(db_spl, 0.0, 140.0))


def _to_float(audio_chunk: np.ndarray) -> np.ndarray:
    if audio_chunk.dtype == np.int16:
        return audio_chunk.astype(np.float32) / 32768.0
    if audio_chunk.dtype != np.float32:
        return audio_chunk.astype(np.float32)
    return audio_chunk


def compute_db(audio_chunk: np.ndarray, sample_rate: int = 44100) -> float:
    """Full-spectrum RMS → dB SPL (for the live meter display)."""
    if audio_chunk is None or len(audio_chunk) == 0:
        return 0.0
    return _rms_to_db_spl(_to_float(audio_chunk))


def compute_db_filtered(audio_chunk: np.ndarray, sample_rate: int = 44100) -> float:
    """Bandpass 50–1000 Hz RMS → dB SPL (muffler-band trigger)."""
    if audio_chunk is None or len(audio_chunk) == 0:
        return 0.0
    audio = _to_float(audio_chunk)
    sos   = butter_bandpass_sos(MUFFLER_FREQ_LOW, MUFFLER_FREQ_HIGH, sample_rate)
    audio = sosfiltfilt(sos, audio)
    return _rms_to_db_spl(audio)


compute_db_fft = compute_db  # back-compat alias


class AcousticTrigger:
    """
    Fires a callback when sound exceeds the threshold for the required duration.

    use_bandpass=True  → compare muffler-band (50–1000 Hz) dB to threshold.
                         Best for RTSP camera audio (motorcycle enforcement).
    use_bandpass=False → compare full-spectrum dB to threshold.
                         Best for microphone input (any loud sound triggers).

    current_db is always the FULL-SPECTRUM reading so the meter responds to
    all sounds regardless of mode.
    """

    def __init__(
        self,
        threshold_db: float = 99.0,
        trigger_duration_ms: int = 200,
        sample_rate: int = 44100,
        on_trigger=None,
        use_bandpass: bool = True,
    ):
        self.threshold_db        = threshold_db
        self.trigger_duration_ms = trigger_duration_ms
        self.sample_rate         = sample_rate
        self.on_trigger          = on_trigger
        self.use_bandpass        = use_bandpass

        self._above_threshold_since: float | None = None
        self._lock              = threading.Lock()
        self._last_trigger_time: float = 0
        self._cooldown_seconds:  float = 3.0
        self._current_db:        float = 0.0
        self._current_waveform:  list[float] = []
        self._last_chunk_time:   float = 0.0

    @property
    def current_db(self) -> float:
        # Decay to 0 if no audio chunk received in the last 300 ms (source disconnected)
        if self._last_chunk_time and time.monotonic() - self._last_chunk_time > 0.3:
            return 0.0
        return self._current_db

    @property
    def current_waveform(self) -> list[float]:
        # Return empty list (flat line) when source is silent / disconnected
        if self._last_chunk_time and time.monotonic() - self._last_chunk_time > 0.3:
            return []
        return self._current_waveform

    def process_chunk(self, audio_chunk: np.ndarray) -> float:
        db_display = compute_db(audio_chunk, self.sample_rate)
        db_trigger = (
            compute_db_filtered(audio_chunk, self.sample_rate)
            if self.use_bandpass
            else db_display
        )
        self._current_db       = db_display
        self._last_chunk_time  = time.monotonic()

        # Downsample to 256 points for WebSocket waveform streaming
        audio_f = _to_float(audio_chunk)
        step = max(1, len(audio_f) // 256)
        self._current_waveform = [round(float(v), 4) for v in audio_f[::step][:256]]

        now = time.monotonic()
        with self._lock:
            if db_trigger >= self.threshold_db:
                if self._above_threshold_since is None:
                    self._above_threshold_since = now
                elapsed_ms = (now - self._above_threshold_since) * 1000
                if elapsed_ms >= self.trigger_duration_ms:
                    if now - self._last_trigger_time >= self._cooldown_seconds:
                        self._last_trigger_time     = now
                        self._above_threshold_since = None
                        if self.on_trigger:
                            threading.Thread(
                                target=self.on_trigger, args=(db_display,), daemon=True
                            ).start()
            else:
                self._above_threshold_since = None

        return db_display
