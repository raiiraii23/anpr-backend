# Backend — Python FastAPI AI Engine

The core processing engine. Connects to the Hikvision RTSP stream, monitors audio in real-time using FFT, triggers the AI pipeline when motorcycle noise exceeds 99 dB, and serves violation data to the dashboard via REST and WebSocket.

---

## Requirements

- Python 3.10+
- PostgreSQL 16 running (see root `docker-compose.yml`)
- (Optional) CUDA GPU for faster YOLOv8 inference

---

## Setup

```bash
# 1. Copy and configure environment
cp .env.example .env

# 2. Edit .env — at minimum set RTSP_URL and DATABASE_URL
#    RTSP_URL=rtsp://admin:yourpassword@192.168.1.100:554/Streaming/Channels/101

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The server auto-creates the `violations` database table on first startup.

---

## Environment Variables (`.env`)

```env
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/thesis_db
RTSP_URL=rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101
NOISE_THRESHOLD_DB=99
TRIGGER_DURATION_MS=500
CAPTURE_DIR=./captures
```

| Variable | Description |
|---|---|
| `DATABASE_URL` | PostgreSQL async connection string (must use `asyncpg` driver) |
| `RTSP_URL` | Full RTSP URL to the Hikvision camera stream |
| `NOISE_THRESHOLD_DB` | dB level that triggers a capture (default 99, per LTO AO 2006-003) |
| `TRIGGER_DURATION_MS` | Noise must exceed threshold for this many milliseconds continuously |
| `CAPTURE_DIR` | Directory where annotated violation JPEGs are saved |

---

## File Reference

### `main.py` — FastAPI Application
Entry point. Manages app lifecycle (DB init, RTSP start), defines all routes and WebSocket, and wires the acoustic trigger callback to the AI pipeline.

**Routes:**
| Method | Path | Description |
|---|---|---|
| GET | `/api/violations` | Paginated list, newest first (`limit`, `offset`) |
| GET | `/api/violations/{id}` | Single record, 404 if missing |
| PATCH | `/api/violations/{id}` | Update `status` (pending/cited/dismissed) and `notes` |
| GET | `/api/stats` | Aggregate counts + average dB |
| GET | `/api/status` | RTSP live status + current dB reading |
| WS | `/ws/live` | Pushes `db_update` every 100 ms; broadcasts `new_violation` on trigger |
| GET | `/captures/*` | Serves saved JPEG files as static assets |

---

### `models.py` — Database Model
SQLAlchemy ORM for the `violations` table.

| Column | Type | Notes |
|---|---|---|
| `id` | Integer PK | Auto-increment |
| `plate_number` | String(20) | Nullable — OCR may fail on obscured plates |
| `decibel_level` | Float | Measured at the moment of trigger |
| `timestamp` | DateTime | UTC, set automatically |
| `image_path` | String(255) | Relative path to annotated JPEG |
| `confidence` | Float | YOLOv8 detection confidence score |
| `location` | String(100) | Defaults to `"Checkpoint A"` |
| `status` | String(20) | `pending` / `cited` / `dismissed` |
| `notes` | Text | Enforcer remarks |

---

### `database.py` — Async Database Layer
- Creates an async SQLAlchemy engine using `asyncpg`
- Exposes `AsyncSessionLocal` for direct session use in trigger callbacks
- `get_db()` — FastAPI dependency that yields an `AsyncSession`
- `init_db()` — runs `CREATE TABLE IF NOT EXISTS` on startup

---

### `audio_processor.py` — Noise Detection

#### `compute_db_fft(chunk, sample_rate=44100)`
1. Normalizes raw PCM int16 to float32
2. Applies a 5th-order Butterworth bandpass filter (**50–1000 Hz**) — this is the mechanical harmonic range of motorcycle engines, filtering out wind, voices, and sharp transients like horns
3. Computes FFT, sums energy in the 50–1000 Hz band
4. Converts to dB: `10 * log10(energy)` + calibration offset (+60)
5. Returns a clamped value in range 0–140 dB

> **Field calibration note:** The +60 offset is a placeholder. During field testing, compare readings against a Class 2 reference meter and adjust this constant accordingly.

#### `AcousticTrigger`
Stateful monitor that calls `on_trigger(db_level)` when:
- dB ≥ `threshold_db` (default 99)
- Sustained continuously for ≥ `trigger_duration_ms` (default 500 ms)
- At least `_cooldown_seconds` (3.0 s) since the last trigger

The callback runs in a new daemon thread to avoid blocking the audio loop.

---

### `rtsp_handler.py` — Video Stream

`RTSPStream` opens `cv2.VideoCapture` with `CAP_FFMPEG` and reads frames in a background daemon thread. Features:
- **Auto-reconnect** — on stream failure, waits 5 seconds and reopens
- **Thread-safe frame access** — `get_frame()` returns a copy under a lock
- Returns `None` if the stream has not yet connected

> **Note:** The camera's built-in microphone audio must be extracted separately (via PyAudio or an ffmpeg subprocess) and fed to `AcousticTrigger.process_chunk()`. This integration is the next implementation step.

---

### `ai_engine.py` — Detection + OCR Pipeline

`AIEngine` uses lazy initialization — models load on the first trigger, not at server startup.

**`process_frame(frame, decibel_level)` steps:**
1. YOLOv8 inference → find highest-confidence bounding box
2. Crop detected region with +5 px padding on each side
3. PaddleOCR on the crop (falls back to full frame if no box found)
4. Filter OCR results by confidence > 0.5, join, uppercase
5. Annotate frame copy: dB in red (top-left), plate text in green
6. Save JPEG to `captures/violation_YYYYMMDD_HHMMSS_ffffff.jpg`
7. Return `{plate_number, decibel_level, timestamp, image_path, confidence}`

> **TODO:** Replace `yolov8n.pt` (generic COCO model) with a fine-tuned model trained on Philippine motorcycle plate images for accurate small-object detection.

---

## Dependencies

```
fastapi          0.115.0   — web framework
uvicorn          0.30.6    — ASGI server
sqlalchemy       2.0.35    — async ORM
asyncpg          0.29.0    — async PostgreSQL driver
opencv-python    4.10.0    — RTSP video capture + image I/O
numpy            1.26.4    — array processing
scipy            1.14.0    — FFT + Butterworth filter
pyaudio          0.2.14    — microphone / audio capture
ultralytics      8.2.87    — YOLOv8
paddlepaddle     2.6.1     — PaddleOCR backend
paddleocr        2.7.3     — OCR engine
python-dotenv    1.0.1     — .env loading
```

---

## Captures Directory

Violation evidence JPEGs are saved to `./captures/` and served at `http://localhost:8000/captures/<filename>`.

The frontend constructs image URLs as:
```
${API_URL}/${image_path.replace('./', '')}
```
