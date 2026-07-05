import re
import socket
import subprocess
import threading
import logging
import time
from typing import Callable, Optional

import numpy as np
import cv2

logger = logging.getLogger(__name__)


def _get_ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _probe_dimensions(ffmpeg_exe: str, rtsp_url: str, timeout: float = 12.0):
    """Return (width, height) of the RTSP stream, or (None, None) on failure."""
    try:
        result = subprocess.run(
            [ffmpeg_exe, "-rtsp_transport", "tcp",
             "-i", rtsp_url, "-t", "0", "-f", "null", "-"],
            capture_output=True, text=True, timeout=timeout,
        )
        # ffmpeg prints stream info to stderr even on "error" exit
        match = re.search(r"(\d{2,4})x(\d{2,4})", result.stderr)
        if match:
            return int(match.group(1)), int(match.group(2))
        logger.warning("Could not parse dimensions from ffmpeg output:\n%s",
                       result.stderr[-600:])
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg probe timed out after %.0fs", timeout)
    except Exception as e:
        logger.error("ffmpeg probe error: %s", e)
    return None, None


class RTSPStream:
    """
    Reads an RTSP stream via an ffmpeg subprocess (TCP transport, rawvideo pipe).
    Provides the latest decoded frame on demand.
    """

    def __init__(self, rtsp_url: str, reconnect_delay: float = 5.0):
        self.rtsp_url        = rtsp_url
        self.reconnect_delay = reconnect_delay

        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock  = threading.Lock()
        self._running     = False
        self._thread: Optional[threading.Thread] = None
        self._ffmpeg_exe  = _get_ffmpeg_exe()

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()
        logger.info("RTSP stream thread started (ffmpeg subprocess).")

    def stop(self):
        self._running = False

    def get_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    @property
    def connected(self) -> bool:
        return self.get_frame() is not None

    def _stream_loop(self):
        while self._running:
            logger.info("Probing RTSP stream dimensions…")
            width, height = _probe_dimensions(self._ffmpeg_exe, self.rtsp_url)

            if width is None:
                logger.warning("Probe failed — retrying in %.0fs", self.reconnect_delay)
                time.sleep(self.reconnect_delay)
                continue

            logger.info("Stream is %dx%d — starting capture.", width, height)
            frame_bytes = width * height * 3  # BGR24

            cmd = [
                self._ffmpeg_exe,
                "-loglevel",        "error",
                # Low-latency input flags
                "-fflags",          "nobuffer",
                "-flags",           "low_delay",
                "-rtsp_transport",  "tcp",
                "-i",               self.rtsp_url,
                "-f",               "rawvideo",
                "-pix_fmt",         "bgr24",
                "-vf",              "fps=25",
                "-vsync",           "0",
                "pipe:1",
            ]

            proc: Optional[subprocess.Popen] = None
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=frame_bytes,  # 1-frame buffer keeps latency minimal
                )
                logger.info("RTSP stream connected.")

                while self._running:
                    raw = proc.stdout.read(frame_bytes)
                    if len(raw) < frame_bytes:
                        logger.warning("Short read (%d/%d bytes) — reconnecting…",
                                       len(raw), frame_bytes)
                        break
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
                    with self._frame_lock:
                        self._latest_frame = frame

            except Exception as e:
                logger.error("RTSP capture error: %s", e)
            finally:
                if proc:
                    try:
                        err = proc.stderr.read(2000) if proc.stderr else b""
                        proc.kill()
                        proc.wait()
                        if err:
                            logger.warning("ffmpeg stderr: %s",
                                           err.decode("utf-8", errors="ignore").strip())
                    except Exception:
                        pass
                with self._frame_lock:
                    self._latest_frame = None

            if self._running:
                logger.info("Reconnecting in %.0fs…", self.reconnect_delay)
                time.sleep(self.reconnect_delay)

        logger.info("RTSP stream thread stopped.")


class CombinedRTSPStream:
    """
    Single ffmpeg process that captures BOTH video and audio from one RTSP connection.

    Video  → stdout  (rawvideo BGR24, 15 fps)
    Audio  → local TCP socket  (s16le mono 44100 Hz)

    Using one connection avoids cameras that reject a second RTSP session and
    eliminates the DTS timestamp conflicts caused by two independent ffmpeg
    processes racing for the same stream.
    """

    def __init__(
        self,
        rtsp_url: str,
        on_audio_chunk: Optional[Callable[[np.ndarray], None]] = None,
        sample_rate: int = 44100,
        chunk_samples: int = 2205,       # 50 ms @ 44100 Hz
        reconnect_delay: float = 5.0,
    ):
        self.rtsp_url        = rtsp_url
        self.on_audio_chunk  = on_audio_chunk
        self.sample_rate     = sample_rate
        self.chunk_samples   = chunk_samples
        self.reconnect_delay = reconnect_delay

        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock  = threading.Lock()
        self._running     = False
        self._ffmpeg_exe  = _get_ffmpeg_exe()

    # ── public API ──────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        threading.Thread(target=self._stream_loop, daemon=True).start()
        logger.info("CombinedRTSPStream started.")

    def stop(self):
        self._running = False

    def get_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    @property
    def connected(self) -> bool:
        return self._latest_frame is not None

    # ── internals ────────────────────────────────────────────────────────────

    @staticmethod
    def _free_port() -> int:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def _log_stderr(self, proc: subprocess.Popen):
        try:
            for raw_line in proc.stderr:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if line:
                    logger.warning("ffmpeg: %s", line)
        except Exception:
            pass

    def _audio_receiver(
        self,
        server_sock: socket.socket,
        proc: subprocess.Popen,
    ):
        """Accept the TCP connection from ffmpeg and feed PCM chunks to callback."""
        bytes_per_chunk = self.chunk_samples * 2  # int16 mono
        try:
            server_sock.settimeout(15.0)
            conn, addr = server_sock.accept()
            logger.info("Audio TCP connected from %s.", addr)
            conn.settimeout(2.0)
            buf = b""
            with conn:
                while self._running and proc.poll() is None:
                    try:
                        data = conn.recv(bytes_per_chunk * 4)
                    except socket.timeout:
                        continue
                    if not data:
                        break
                    buf += data
                    while len(buf) >= bytes_per_chunk:
                        chunk = np.frombuffer(buf[:bytes_per_chunk], dtype=np.int16)
                        buf   = buf[bytes_per_chunk:]
                        if self.on_audio_chunk:
                            try:
                                self.on_audio_chunk(chunk)
                            except Exception as e:
                                logger.error("Audio callback error: %s", e)
        except socket.timeout:
            logger.warning("Audio TCP: no connection from ffmpeg within 15 s — stream may have no audio track.")
        except Exception as e:
            logger.warning("Audio TCP receiver error: %s", e)

    def _stream_loop(self):
        while self._running:
            logger.info("Probing RTSP stream dimensions…")
            width, height = _probe_dimensions(self._ffmpeg_exe, self.rtsp_url)
            if width is None:
                logger.warning("Probe failed — retrying in %.0fs", self.reconnect_delay)
                time.sleep(self.reconnect_delay)
                continue

            logger.info("Stream is %dx%d — starting combined capture.", width, height)
            frame_bytes = width * height * 3  # BGR24

            audio_port  = self._free_port()
            server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind(("127.0.0.1", audio_port))
            server_sock.listen(1)

            cmd = [
                self._ffmpeg_exe,
                "-loglevel",                   "error",
                # Low-latency + DTS fix flags on input
                "-fflags",                     "nobuffer+discardcorrupt+genpts",
                "-flags",                      "low_delay",
                "-use_wallclock_as_timestamps","1",
                "-rtsp_transport",             "tcp",
                "-i",                          self.rtsp_url,
                # ── video output → stdout ──
                "-map",    "0:v:0",
                "-f",      "rawvideo",
                "-pix_fmt","bgr24",
                "-vf",     "fps=25",
                "-vsync",  "0",
                "pipe:1",
                # ── audio output → local TCP socket ──
                # aresample=async=1 inserts/drops samples to maintain monotonic timestamps,
                # eliminating the "non monotonically increasing dts" warnings from the camera.
                "-map",    "0:a:0",
                "-af",     "aresample=async=1:min_hard_comp=0.1:first_pts=0",
                "-f",      "s16le",
                "-acodec", "pcm_s16le",
                "-ac",     "1",
                "-ar",     str(self.sample_rate),
                f"tcp://127.0.0.1:{audio_port}",
            ]

            proc: Optional[subprocess.Popen] = None
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=frame_bytes,  # 1-frame buffer keeps latency minimal
                )

                threading.Thread(target=self._log_stderr, args=(proc,), daemon=True).start()

                if self.on_audio_chunk:
                    threading.Thread(
                        target=self._audio_receiver,
                        args=(server_sock, proc),
                        daemon=True,
                    ).start()
                else:
                    server_sock.close()

                logger.info("RTSP stream connected (combined video+audio).")

                while self._running:
                    raw = proc.stdout.read(frame_bytes)
                    if len(raw) < frame_bytes:
                        logger.warning(
                            "Short read (%d/%d bytes) — reconnecting…",
                            len(raw), frame_bytes,
                        )
                        break
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
                    with self._frame_lock:
                        self._latest_frame = frame

            except Exception as e:
                logger.error("RTSP capture error: %s", e)
            finally:
                if proc:
                    try:
                        proc.kill()
                        proc.wait()
                    except Exception:
                        pass
                try:
                    server_sock.close()
                except Exception:
                    pass
                with self._frame_lock:
                    self._latest_frame = None

            if self._running:
                logger.info("Reconnecting in %.0fs…", self.reconnect_delay)
                time.sleep(self.reconnect_delay)

        logger.info("CombinedRTSPStream stopped.")
