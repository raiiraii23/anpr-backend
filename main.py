import os
import asyncio
import logging
import threading
import time
import secrets
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.responses import StreamingResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func
from dotenv import load_dotenv
from pydantic import BaseModel

from database import init_db, get_db, AsyncSessionLocal
from models import Violation, Detection
from rtsp_handler import RTSPStream, CombinedRTSPStream
from audio_processor import AcousticTrigger
from audio_source import MicrophoneSource
from ai_engine import AIEngine

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RTSP_URL            = os.getenv("RTSP_URL", "rtsp://admin1:admin123@192.168.8.46:554/stream1")
NOISE_THRESHOLD_DB  = float(os.getenv("NOISE_THRESHOLD_DB", "99"))
TRIGGER_DURATION_MS = int(os.getenv("TRIGGER_DURATION_MS", "500"))
CAPTURE_DIR         = os.getenv("CAPTURE_DIR", "./captures")
AUDIO_SOURCE        = os.getenv("AUDIO_SOURCE", "microphone")
AUDIO_DEVICE        = os.getenv("AUDIO_DEVICE", None)

ADMIN_USERNAME      = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD      = os.getenv("ADMIN_PASSWORD", "admin123")

# Active tokens in memory
ACTIVE_TOKENS: set[str] = set()

class LoginRequest(BaseModel):
    username: str
    password: str

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
security = HTTPBearer(auto_error=False)

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    token: Optional[str] = None
):
    req_token = None
    if credentials:
        req_token = credentials.credentials
    elif token:
        req_token = token
        
    if not req_token or req_token not in ACTIVE_TOKENS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing access token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return "admin"

# rtsp_stream is set in lifespan — either CombinedRTSPStream (rtsp mode) or RTSPStream (mic mode)
rtsp_stream: RTSPStream | CombinedRTSPStream | None = None
ai_engine   = AIEngine()
acoustic_trigger: Optional[AcousticTrigger] = None
audio_source = None
_ws_clients: list[WebSocket] = []

# Shared state for the continuous detection worker
_latest_annotated: Optional[np.ndarray] = None
_annotated_lock = threading.Lock()
_seen_track_ids: set[int] = set()
_detection_stop = threading.Event()
_main_loop: Optional[asyncio.AbstractEventLoop] = None


async def broadcast_ws(message: dict):
    disconnected = []
    for ws in _ws_clients:
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        if ws in _ws_clients:
            _ws_clients.remove(ws)


async def save_violation(record: dict):
    async with AsyncSessionLocal() as session:
        violation = Violation(**record)
        session.add(violation)
        await session.commit()
        await session.refresh(violation)
        logger.info("Violation saved: plate=%s dB=%.1f", violation.plate_number, violation.decibel_level)
        await broadcast_ws({
            "event": "new_violation",
            "data": {
                "id": violation.id,
                "plate_number": violation.plate_number,
                "decibel_level": violation.decibel_level,
                "timestamp": violation.timestamp.isoformat(),
                "image_path": violation.image_path,
                "status": violation.status,
                "location": violation.location,
            },
        })


async def save_detection(record: dict):
    async with AsyncSessionLocal() as session:
        detection = Detection(**record)
        session.add(detection)
        await session.commit()
        await session.refresh(detection)
        logger.info(
            "Detection saved: #%d %s track=%d conf=%.2f",
            detection.id, detection.class_name, detection.track_id, detection.confidence,
        )
        await broadcast_ws({
            "event": "new_detection",
            "data": {
                "id":           detection.id,
                "track_id":     detection.track_id,
                "class_name":   detection.class_name,
                "confidence":   detection.confidence,
                "plate_number": detection.plate_number,
                "timestamp":    detection.timestamp.isoformat(),
                "image_path":   detection.image_path,
            },
        })


def _grab_frame() -> np.ndarray | None:
    """Get the best available frame: RTSP first, local webcam as fallback."""
    if rtsp_stream is not None:
        frame = rtsp_stream.get_frame()
        if frame is not None:
            return frame
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                logger.info("Snapshot taken from local webcam (RTSP unavailable).")
                return frame
    except Exception:
        pass
    return None


def on_noise_trigger(db_level: float):
    """Called from audio thread when noise threshold is sustained."""
    logger.info("ACOUSTIC TRIGGER: %.1f dB", db_level)
    frame = _grab_frame()
    if frame is None:
        logger.warning("No frame available at trigger time.")
        return
    record = ai_engine.process_frame(frame, db_level)
    if _main_loop is not None:
        asyncio.run_coroutine_threadsafe(save_violation(record), _main_loop)


def _detection_worker():
    """Continuously pulls frames, runs YOLO tracker, dedupes, persists new vehicles."""
    global _latest_annotated
    logger.info("Detection worker started.")
    while not _detection_stop.is_set():
        if rtsp_stream is None:
            time.sleep(0.1)
            continue
        frame = rtsp_stream.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue

        try:
            annotated, detections = ai_engine.detect_vehicles(frame)
        except Exception as e:
            logger.error("detect_vehicles failed: %s", e)
            time.sleep(0.1)
            continue

        with _annotated_lock:
            _latest_annotated = annotated

        for det in detections:
            tid = det["track_id"]
            if tid in _seen_track_ids:
                continue
            _seen_track_ids.add(tid)
            image_path   = ai_engine.save_detection_crop(annotated, det["box"], tid)
            plate_number = ai_engine.read_plate_from_frame(frame, det["box"])
            record = {
                "track_id":     tid,
                "class_name":   det["class_name"],
                "confidence":   det["confidence"],
                "image_path":   image_path,
                "plate_number": plate_number,
            }
            if _main_loop is not None:
                asyncio.run_coroutine_threadsafe(save_detection(record), _main_loop)

        time.sleep(0.025)
    logger.info("Detection worker stopped.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _main_loop, rtsp_stream, audio_source
    _main_loop = asyncio.get_running_loop()
    await init_db()

    global acoustic_trigger
    acoustic_trigger = AcousticTrigger(
        threshold_db=NOISE_THRESHOLD_DB,
        trigger_duration_ms=TRIGGER_DURATION_MS,
        on_trigger=on_noise_trigger,
        use_bandpass=(AUDIO_SOURCE == "rtsp"),
    )

    if AUDIO_SOURCE == "rtsp":
        # One ffmpeg process handles both video frames and audio — avoids the
        # "second connection rejected" problem on single-session cameras.
        rtsp_stream = CombinedRTSPStream(
            rtsp_url=RTSP_URL,
            on_audio_chunk=acoustic_trigger.process_chunk,
        )
        audio_source = None
        logger.info("Audio source: RTSP via CombinedRTSPStream (%s)", RTSP_URL)
    else:
        rtsp_stream = RTSPStream(RTSP_URL)
        device = int(AUDIO_DEVICE) if AUDIO_DEVICE is not None else None
        audio_source = MicrophoneSource(
            on_chunk=acoustic_trigger.process_chunk,
            device=device,
        )
        audio_source.start()
        logger.info("Audio source: microphone (device=%s)", device)

    rtsp_stream.start()
    worker = threading.Thread(target=_detection_worker, daemon=True)
    worker.start()
    logger.info(
        "System ready. Source: %s | Threshold: %.0f dB | Duration: %dms",
        AUDIO_SOURCE, NOISE_THRESHOLD_DB, TRIGGER_DURATION_MS,
    )
    yield
    _detection_stop.set()
    if audio_source:
        audio_source.stop()
    rtsp_stream.stop()


app = FastAPI(title="Motorcycle Noise Enforcement API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(CAPTURE_DIR, exist_ok=True)

@app.post("/api/auth/login")
async def auth_login(data: LoginRequest):
    if data.username == ADMIN_USERNAME and data.password == ADMIN_PASSWORD:
        token = secrets.token_hex(32)
        ACTIVE_TOKENS.add(token)
        return {"token": token}
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid username or password"
    )

@app.get("/api/auth/verify")
async def auth_verify(user: str = Depends(get_current_user)):
    return {"status": "ok", "user": user}

@app.post("/api/auth/logout")
async def auth_logout(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    token: Optional[str] = None
):
    req_token = None
    if credentials:
        req_token = credentials.credentials
    elif token:
        req_token = token
        
    if req_token in ACTIVE_TOKENS:
        ACTIVE_TOKENS.remove(req_token)
    return {"status": "logged out"}

@app.get("/captures/{filename}")
async def get_capture(filename: str, user: str = Depends(get_current_user)):
    file_path = os.path.join(CAPTURE_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


@app.get("/api/violations")
async def list_violations(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    user: str = Depends(get_current_user),
):
    result = await db.execute(
        select(Violation).order_by(desc(Violation.timestamp)).limit(limit).offset(offset)
    )
    violations = result.scalars().all()
    return [
        {
            "id": v.id,
            "plate_number": v.plate_number,
            "decibel_level": v.decibel_level,
            "timestamp": v.timestamp.isoformat() if v.timestamp else None,
            "image_path": v.image_path,
            "confidence": v.confidence,
            "location": v.location,
            "status": v.status,
            "notes": v.notes,
        }
        for v in violations
    ]


@app.get("/api/violations/{violation_id}")
async def get_violation(
    violation_id: int,
    db: AsyncSession = Depends(get_db),
    user: str = Depends(get_current_user),
):
    result = await db.execute(select(Violation).where(Violation.id == violation_id))
    v = result.scalar_one_or_none()
    if not v:
        raise HTTPException(status_code=404, detail="Violation not found")
    return {
        "id": v.id,
        "plate_number": v.plate_number,
        "decibel_level": v.decibel_level,
        "timestamp": v.timestamp.isoformat() if v.timestamp else None,
        "image_path": v.image_path,
        "confidence": v.confidence,
        "location": v.location,
        "status": v.status,
        "notes": v.notes,
    }


@app.patch("/api/violations/{violation_id}")
async def update_violation(
    violation_id: int,
    status: str,
    notes: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    user: str = Depends(get_current_user),
):
    result = await db.execute(select(Violation).where(Violation.id == violation_id))
    v = result.scalar_one_or_none()
    if not v:
        raise HTTPException(status_code=404, detail="Violation not found")
    v.status = status
    if notes is not None:
        v.notes = notes
    await db.commit()
    return {"status": "updated"}


@app.get("/api/detections")
async def list_detections(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    user: str = Depends(get_current_user),
):
    result = await db.execute(
        select(Detection).order_by(desc(Detection.timestamp)).limit(limit).offset(offset)
    )
    rows = result.scalars().all()
    return [
        {
            "id":           d.id,
            "track_id":     d.track_id,
            "class_name":   d.class_name,
            "confidence":   d.confidence,
            "plate_number": d.plate_number,
            "timestamp":    d.timestamp.isoformat() if d.timestamp else None,
            "image_path":   d.image_path,
        }
        for d in rows
    ]


@app.post("/api/detections/reset")
async def reset_detections(db: AsyncSession = Depends(get_db), user: str = Depends(get_current_user)):
    _seen_track_ids.clear()
    await db.execute(Detection.__table__.delete())
    await db.commit()
    return {"status": "cleared"}


@app.post("/api/violations/clear")
async def clear_violations(db: AsyncSession = Depends(get_db), user: str = Depends(get_current_user)):
    """Delete all violation records and their snapshot files."""
    result = await db.execute(select(Violation.image_path))
    paths  = [r[0] for r in result.fetchall() if r[0]]

    deleted_files = 0
    for p in paths:
        try:
            if os.path.exists(p):
                os.remove(p)
                deleted_files += 1
        except Exception as e:
            logger.warning("Could not delete file %s: %s", p, e)

    count_r = await db.execute(select(func.count()).select_from(Violation))
    total   = count_r.scalar() or 0
    await db.execute(Violation.__table__.delete())
    await db.commit()
    logger.info("Cleared %d violations, deleted %d files.", total, deleted_files)
    return {"deleted_records": total, "deleted_files": deleted_files}


@app.post("/api/trigger/test")
async def test_trigger(db_level: float = 99.5, user: str = Depends(get_current_user)):
    """Manually fire the noise trigger — useful for testing the snapshot pipeline."""
    frame = _grab_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="No camera frame available")
    record = ai_engine.process_frame(frame, db_level)
    if _main_loop is not None:
        asyncio.run_coroutine_threadsafe(save_violation(record), _main_loop)
    return {"status": "triggered", "db_level": db_level}


@app.get("/api/stats")
async def get_stats(db: AsyncSession = Depends(get_db), user: str = Depends(get_current_user)):
    total_r = await db.execute(select(func.count()).select_from(Violation))
    pending_r = await db.execute(select(func.count()).select_from(Violation).where(Violation.status == "pending"))
    cited_r = await db.execute(select(func.count()).select_from(Violation).where(Violation.status == "cited"))
    avg_db_r = await db.execute(select(func.avg(Violation.decibel_level)).select_from(Violation))
    det_count_r = await db.execute(select(func.count()).select_from(Detection))
    total = total_r.scalar() or 0
    pending = pending_r.scalar() or 0
    cited = cited_r.scalar() or 0
    return {
        "total": total,
        "pending": pending,
        "cited": cited,
        "dismissed": total - pending - cited,
        "avg_decibel": round(float(avg_db_r.scalar() or 0), 1),
        "detections": det_count_r.scalar() or 0,
    }


@app.get("/api/status")
async def system_status(user: str = Depends(get_current_user)):
    rtsp_ok = rtsp_stream is not None and rtsp_stream.get_frame() is not None
    return {
        "rtsp_connected":     rtsp_ok,
        "current_db":         acoustic_trigger.current_db if acoustic_trigger else 0,
        "threshold_db":       NOISE_THRESHOLD_DB,
        "trigger_duration_ms": TRIGGER_DURATION_MS,
        "unique_vehicles":    len(_seen_track_ids),
    }


def _make_no_signal_frame() -> bytes:
    """Return a JPEG-encoded 'NO SIGNAL' placeholder frame."""
    h, w = 360, 640
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (10, 20, 10)
    cv2.putText(img, "NO SIGNAL", (w // 2 - 130, h // 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 200, 0), 3, cv2.LINE_AA)
    cv2.putText(img, "Waiting for camera...", (w // 2 - 140, h // 2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 0), 2, cv2.LINE_AA)
    _, jpeg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return jpeg.tobytes()


_no_signal_jpeg: bytes = b""


@app.get("/api/video_feed")
async def video_feed(user: str = Depends(get_current_user)):
    boundary = b"--frame"

    async def gen():
        global _no_signal_jpeg
        while True:
            with _annotated_lock:
                frame = None if _latest_annotated is None else _latest_annotated.copy()

            if frame is None:
                # Fall back to a raw RTSP frame (detection worker may not have started yet)
                frame = rtsp_stream.get_frame()

            if frame is not None:
                ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    data = jpeg.tobytes()
                else:
                    data = None
            else:
                # Nothing available — send the cached placeholder
                if not _no_signal_jpeg:
                    _no_signal_jpeg = await asyncio.get_event_loop().run_in_executor(
                        None, _make_no_signal_frame
                    )
                data = _no_signal_jpeg

            if data:
                yield (
                    boundary + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n"
                    + data + b"\r\n"
                )

            await asyncio.sleep(0.05)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/snapshot")
async def snapshot(user: str = Depends(get_current_user)):
    """Return a single JPEG frame — used by the frontend for polling-based display."""
    with _annotated_lock:
        frame = None if _latest_annotated is None else _latest_annotated.copy()

    if frame is None:
        frame = rtsp_stream.get_frame()

    if frame is None:
        jpeg_bytes = _make_no_signal_frame()
    else:
        ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        jpeg_bytes = jpeg.tobytes() if ok else _make_no_signal_frame()

    from fastapi.responses import Response
    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket, token: Optional[str] = None):
    if not token or token not in ACTIVE_TOKENS:
        await websocket.accept()
        await websocket.close(code=1008)
        return
    await websocket.accept()
    _ws_clients.append(websocket)
    try:
        while True:
            db_val   = acoustic_trigger.current_db       if acoustic_trigger else 0.0
            wave     = acoustic_trigger.current_waveform  if acoustic_trigger else []
            rtsp_ok  = rtsp_stream is not None and rtsp_stream.get_frame() is not None
            await websocket.send_json({
                "event":           "db_update",
                "value":           round(db_val, 1),
                "waveform":        wave,
                "rtsp_connected":  rtsp_ok,
                "unique_vehicles": len(_seen_track_ids),
                "threshold_db":    NOISE_THRESHOLD_DB,
            })
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)
