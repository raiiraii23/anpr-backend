import cv2
import numpy as np
import os
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

CAPTURE_DIR     = Path(os.getenv("CAPTURE_DIR", "./captures"))
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Minimum crop dimension for OCR — smaller crops are upscaled
OCR_MIN_DIM = 64


def _preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    """
    Sharpen and normalise a vehicle crop to maximise OCR accuracy:
      1. Upscale if too small (interpolation=CUBIC)
      2. Convert to grayscale
      3. CLAHE contrast enhancement
      4. Bilateral denoise (keeps edges sharp)
      5. Adaptive threshold → clean black-on-white image
    """
    if img is None or img.size == 0:
        return img

    # Upscale small crops so characters are readable
    h, w = img.shape[:2]
    scale = max(1, OCR_MIN_DIM // min(h, w, OCR_MIN_DIM + 1))
    if scale > 1:
        img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

    # CLAHE — local contrast boost
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)

    # Bilateral filter — remove noise while keeping plate edges
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Adaptive threshold → binary image (better for OCR engines)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2,
    )
    return binary


class AIEngine:
    """YOLOv8 vehicle detection/tracking + EasyOCR plate recognition."""

    def __init__(self):
        self._yolo        = None
        self._ocr         = None
        self._ocr_type    = None   # "easy" | "paddle" | None
        self._initialized = False

    # ── Lazy init ─────────────────────────────────────────────────────────────

    def _lazy_init(self):
        if self._initialized:
            return
        self._initialized = True

        # YOLO
        try:
            from ultralytics import YOLO
            self._yolo = YOLO("yolov8n.pt")
            logger.info("YOLOv8n loaded.")
        except Exception as e:
            logger.error("YOLOv8 load failed: %s", e)

        # OCR — try EasyOCR first (reliable on Windows), fall back to PaddleOCR
        try:
            import easyocr
            self._ocr      = easyocr.Reader(["en"], gpu=False, verbose=False)
            self._ocr_type = "easy"
            logger.info("EasyOCR loaded (CPU).")
        except Exception as e1:
            logger.warning("EasyOCR unavailable (%s), trying PaddleOCR…", e1)
            try:
                from paddleocr import PaddleOCR
                self._ocr      = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
                self._ocr_type = "paddle"
                logger.info("PaddleOCR loaded.")
            except Exception as e2:
                logger.warning("No OCR engine available — plate text will be skipped. (%s)", e2)

    # ── Vehicle detection / tracking ──────────────────────────────────────────

    def detect_vehicles(self, frame: np.ndarray):
        """
        YOLO tracker on frame. Returns (annotated_frame, detections).
        Each detection: {track_id, class_name, confidence, box}.
        """
        self._lazy_init()
        annotated  = frame.copy()
        detections: list[dict] = []

        if self._yolo is None:
            return annotated, detections

        try:
            results = self._yolo.track(
                frame,
                persist=True,
                verbose=False,
                classes=list(VEHICLE_CLASSES.keys()),
                tracker="bytetrack.yaml",
                conf=0.35,   # minimum confidence
            )
        except Exception as e:
            logger.error("YOLO track error: %s", e)
            return annotated, detections

        if not results:
            return annotated, detections

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return annotated, detections

        for box in r.boxes:
            if box.id is None:
                continue
            cls_id   = int(box.cls[0]) if box.cls is not None else -1
            cls_name = VEHICLE_CLASSES.get(cls_id, "vehicle")
            conf     = float(box.conf[0]) if box.conf is not None else 0.0
            track_id = int(box.id[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 65), 2)
            label = f"{cls_name} #{track_id} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 6, y1), (0, 255, 65), -1)
            cv2.putText(
                annotated, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
            )
            detections.append({
                "track_id":  track_id,
                "class_name": cls_name,
                "confidence": conf,
                "box":        (x1, y1, x2, y2),
            })

        return annotated, detections

    # ── Crop helpers ──────────────────────────────────────────────────────────

    def _safe_crop(
        self, frame: np.ndarray, box: tuple, pad: int = 0
    ) -> np.ndarray | None:
        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = frame.shape[:2]
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
        if x2 - x1 < 5 or y2 - y1 < 5:
            return None
        return frame[y1:y2, x1:x2].copy()

    def save_detection_crop(
        self, frame: np.ndarray, box: tuple, track_id: int
    ) -> str | None:
        crop = self._safe_crop(frame, box)
        if crop is None:
            return None
        try:
            ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
            path = CAPTURE_DIR / f"detection_{ts}_id{track_id}.jpg"
            cv2.imwrite(str(path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            return str(path)
        except Exception as e:
            logger.error("Detection crop save error: %s", e)
            return None

    # ── OCR ───────────────────────────────────────────────────────────────────

    def _run_ocr(self, img: np.ndarray) -> str | None:
        """Run the loaded OCR engine on img; try both original and preprocessed, keep best."""
        if self._ocr is None or img is None or img.size == 0:
            return None

        processed = _preprocess_for_ocr(img)
        PLATE_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

        def _easy_read(src) -> tuple[str, float]:
            results = self._ocr.readtext(
                src, detail=1, paragraph=False,
                allowlist=PLATE_CHARS,
                text_threshold=0.5, low_text=0.3, link_threshold=0.4,
            )
            texts  = [(r[1], r[2]) for r in results if r[2] >= 0.25]
            if not texts:
                return "", 0.0
            raw   = "".join(t for t, _ in texts).strip().upper()
            plate = "".join(c for c in raw if c.isalnum() or c in " -").strip()
            score = sum(c for _, c in texts) / len(texts)
            return plate, score

        def _paddle_read(src) -> tuple[str, float]:
            results = self._ocr.ocr(src, cls=True)
            if not results or not results[0]:
                return "", 0.0
            lines  = [(line[1][0], line[1][1]) for line in results[0] if line[1][1] >= 0.25]
            if not lines:
                return "", 0.0
            raw   = "".join(t for t, _ in lines).strip().upper()
            plate = "".join(c for c in raw if c.isalnum() or c in " -").strip()
            score = sum(c for _, c in lines) / len(lines)
            return plate, score

        try:
            if self._ocr_type == "easy":
                plate_orig, score_orig = _easy_read(img)
                plate_proc, score_proc = _easy_read(processed)
                # Prefer the read with higher average confidence and sufficient length
                if len(plate_proc) >= 3 and score_proc > score_orig:
                    plate, score = plate_proc, score_proc
                else:
                    plate, score = plate_orig, score_orig
            else:
                plate, score = _paddle_read(processed)

            return plate if len(plate) >= 3 else None

        except Exception as e:
            logger.error("OCR error: %s", e)
            return None

    def read_plate_from_frame(
        self, frame: np.ndarray, box: tuple
    ) -> str | None:
        """OCR the vehicle region of the clean (un-annotated) frame."""
        self._lazy_init()
        crop = self._safe_crop(frame, box, pad=12)
        return self._run_ocr(crop)

    # ── Violation snapshot ────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, decibel_level: float) -> dict:
        """
        Full violation pipeline:
          1. Detect highest-confidence vehicle with YOLO.
          2. OCR the crop with preprocessing.
          3. Draw professional annotation overlay.
          4. Save at JPEG quality 95.
        """
        self._lazy_init()
        timestamp  = datetime.utcnow()
        filename   = f"violation_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        image_path = str(CAPTURE_DIR / filename)

        plate_text = None
        confidence = None
        best_box   = None

        # ── Detect vehicles ───────────────────────────────────────────────────
        if self._yolo is not None:
            try:
                results   = self._yolo(
                    frame, verbose=False,
                    classes=list(VEHICLE_CLASSES.keys()),
                    conf=0.30,
                )
                best_conf = 0.0
                for result in results:
                    for box in result.boxes:
                        c = float(box.conf[0])
                        if c > best_conf:
                            best_conf  = c
                            best_box   = box.xyxy[0].cpu().numpy().astype(int)
                            confidence = best_conf
            except Exception as e:
                logger.error("YOLO inference error: %s", e)

        # ── OCR on vehicle crop ───────────────────────────────────────────────
        if best_box is not None:
            crop       = self._safe_crop(frame, tuple(best_box), pad=12)
            plate_text = self._run_ocr(crop)

        # ── Annotate ──────────────────────────────────────────────────────────
        annotated = frame.copy()
        h, w      = annotated.shape[:2]

        # Green bounding box + corner brackets
        if best_box is not None:
            x1, y1, x2, y2 = best_box
            x1 = max(0, x1 - 4); y1 = max(0, y1 - 4)
            x2 = min(w, x2 + 4); y2 = min(h, y2 + 4)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 65), 3)
            al = min(32, (x2 - x1) // 4, (y2 - y1) // 4)
            for px, py, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(annotated, (px, py), (px + dx * al, py), (0, 255, 65), 4)
                cv2.line(annotated, (px, py), (px, py + dy * al), (0, 255, 65), 4)

        # Semi-transparent bottom banner
        banner_h = max(76, h // 6)
        overlay  = annotated.copy()
        cv2.rectangle(overlay, (0, h - banner_h), (w, h), (4, 8, 4), -1)
        cv2.addWeighted(overlay, 0.72, annotated, 0.28, 0, annotated)
        cv2.line(annotated, (0, h - banner_h), (w, h - banner_h), (0, 255, 65), 2)

        # Timestamp (top-right)
        ts_str      = timestamp.strftime("%Y-%m-%d  %H:%M:%S UTC")
        (tsw, _), _ = cv2.getTextSize(ts_str, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        cv2.putText(annotated, ts_str, (w - tsw - 10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 200, 50), 1, cv2.LINE_AA)

        # dB (large, colour-coded red/green)
        db_col = (40, 40, 255) if decibel_level >= 99 else (0, 200, 80)
        cv2.putText(annotated, f"{decibel_level:.1f} dB",
                    (14, h - banner_h + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, db_col, 3, cv2.LINE_AA)

        # Plate number
        if plate_text:
            cv2.putText(annotated, plate_text,
                        (14, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.05, (0, 255, 65), 2, cv2.LINE_AA)
        else:
            cv2.putText(annotated, "PLATE UNREAD",
                        (14, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 110, 60), 1, cv2.LINE_AA)

        # VIOLATION badge
        bw = 152
        cv2.rectangle(annotated,
                      (w - bw - 10, h - banner_h + 8),
                      (w - 10, h - banner_h + 50),
                      (0, 0, 180), -1)
        cv2.rectangle(annotated,
                      (w - bw - 10, h - banner_h + 8),
                      (w - 10, h - banner_h + 50),
                      (0, 60, 255), 2)
        cv2.putText(annotated, "VIOLATION",
                    (w - bw - 4, h - banner_h + 39),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        if confidence:
            cv2.putText(annotated, f"conf {confidence:.0%}",
                        (w - bw - 10, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 180, 80), 1, cv2.LINE_AA)

        # Save at max quality
        try:
            cv2.imwrite(image_path, annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        except Exception as e:
            logger.error("Image save error: %s", e)
            image_path = None

        return {
            "plate_number":  plate_text,
            "decibel_level": decibel_level,
            "timestamp":     timestamp,
            "image_path":    image_path,
            "confidence":    confidence,
        }
