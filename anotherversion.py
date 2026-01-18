import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from fer import FER
from ultralytics import YOLO


class RealTimeEmotionEngine:
    """
    YOLO Face Detection + FER Emotion + IoU Tracking

    Output per student:
    {
      "student_id": "student_01",
      "expression": "happy",
      "confidence": 0.85,
      "state": "Engaged (0.85)",
      "attention": 70.9,
      "confusion": 12.9,
      "distraction": 16.2
    }
    """

    def __init__(
        self,
        yolo_model_path: str = "yolov8n-face.pt",
        detect_every_n_frames: int = 3,        # smoothness knob (3 or 5)
        emotion_every_n_frames: int = 8,       # smoothness knob (8 or 10)
        emotion_cache_seconds: float = 1.0,    # smoothness knob (1.0 to 2.0)
        track_timeout_seconds: float = 2.0,
        iou_threshold: float = 0.30,
        crop_padding_ratio: float = 0.20,
        min_face_size: int = 45,               # skip tiny face emotion
        resize_width: int = 640,               # speed
        resize_height: int = 360,              # speed
        yolo_conf: float = 0.40,
    ):
        self.detector = YOLO(yolo_model_path)
        self.fer = FER(mtcnn=False)

        self.DETECT_EVERY_N_FRAMES = int(detect_every_n_frames)
        self.EMOTION_EVERY_N_FRAMES = int(emotion_every_n_frames)
        self.EMOTION_CACHE_SECONDS = float(emotion_cache_seconds)

        self.TRACK_TIMEOUT = float(track_timeout_seconds)
        self.IOU_THRESHOLD = float(iou_threshold)

        self.CROP_PADDING_RATIO = float(crop_padding_ratio)
        self.MIN_FACE_SIZE = int(min_face_size)

        self.RW = int(resize_width)
        self.RH = int(resize_height)
        self.YOLO_CONF = float(yolo_conf)

        self._frame_idx = 0
        self._next_track_id = 1
        self._next_student_num = 1

        # track_id -> dict
        self._tracks: Dict[int, Dict[str, object]] = {}

        # detection cache (avoid YOLO every frame)
        self._last_detections: List[List[int]] = []

    # -------------------------
    # MAIN PROCESS
    # -------------------------
    def process_frame(self, frame_bgr: np.ndarray) -> List[dict]:
        self._frame_idx += 1
        now = time.time()

        # Resize for speed (works fine for demo)
        frame_small = cv2.resize(frame_bgr, (self.RW, self.RH))

        # Run YOLO detection only every N frames
        if self._frame_idx % self.DETECT_EVERY_N_FRAMES == 0 or not self._tracks:
            detections = self._detect_faces(frame_small)
            self._last_detections = detections
        else:
            detections = self._last_detections

        # Update tracking
        self._update_tracks(detections, now)

        # Output JSON for all active tracks
        outputs: List[dict] = []
        for tid in sorted(self._tracks.keys()):
            tr = self._tracks[tid]
            bbox = tr["bbox"]
            student_id = tr["student_id"]

            # Get emotion (cached + throttled)
            expression, confidence = self._get_emotion_for_track(frame_small, tid, now)

            # Map to state + metrics
            state, att, confu, dist = self._expression_to_state_and_metrics(expression, confidence)

            outputs.append(
                {
                    "student_id": student_id,
                    "expression": expression,
                    "confidence": float(np.clip(confidence, 0.0, 1.0)),
                    "state": state,
                    "attention": att,
                    "confusion": confu,
                    "distraction": dist,
                }
            )

        return outputs

    def draw_debug(self, frame_bgr: np.ndarray) -> np.ndarray:
        out = cv2.resize(frame_bgr.copy(), (self.RW, self.RH))

        for tid in sorted(self._tracks.keys()):
            tr = self._tracks[tid]
            x, y, w, h = tr["bbox"]
            student_id = tr["student_id"]
            expr = tr.get("expression", "neutral")
            conf = float(tr.get("confidence", 0.5))

            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{student_id} {expr} {conf:.2f}"
            cv2.putText(
                out,
                label,
                (x, max(0, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return out

    # -------------------------
    # FACE DETECTION (YOLO)
    # -------------------------
    def _detect_faces(self, frame_bgr_small: np.ndarray) -> List[List[int]]:
        """
        Returns detections list of [x, y, w, h] in resized frame coordinates.
        """
        results = self.detector(frame_bgr_small, conf=self.YOLO_CONF, verbose=False)[0]

        detections: List[List[int]] = []
        if results.boxes is None:
            return detections

        boxes = results.boxes.xyxy.cpu().numpy()
        for x1, y1, x2, y2 in boxes:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue
            if w < 30 or h < 30:
                continue
            detections.append([x1, y1, w, h])

        return detections

    # -------------------------
    # TRACKING (IoU)
    # -------------------------
    def _update_tracks(self, detections: List[List[int]], now: float) -> None:
        # Remove expired tracks
        expired = [
            tid for tid, tr in self._tracks.items()
            if (now - float(tr["last_seen"])) > self.TRACK_TIMEOUT
        ]
        for tid in expired:
            del self._tracks[tid]

        # If no tracks exist, create for all detections
        if not self._tracks and detections:
            for det in detections:
                self._create_track(det, now)
            return

        if not detections:
            return

        # Match detections to existing tracks (greedy)
        unmatched_dets = set(range(len(detections)))
        used_tracks = set()

        for det_idx, det in enumerate(detections):
            best_tid = None
            best_iou = 0.0

            for tid, tr in self._tracks.items():
                if tid in used_tracks:
                    continue
                iou = self._iou(det, tr["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None and best_iou >= self.IOU_THRESHOLD:
                # Update existing track
                self._tracks[best_tid]["bbox"] = det
                self._tracks[best_tid]["last_seen"] = now
                used_tracks.add(best_tid)
                unmatched_dets.discard(det_idx)

        # New tracks for unmatched detections
        for det_idx in unmatched_dets:
            self._create_track(detections[det_idx], now)

    def _create_track(self, bbox: List[int], now: float) -> None:
        track_id = self._next_track_id
        self._next_track_id += 1

        student_id = f"student_{self._next_student_num:02d}"
        self._next_student_num += 1

        self._tracks[track_id] = {
            "bbox": bbox,
            "last_seen": now,
            "student_id": student_id,
            "expression": "neutral",
            "confidence": 0.5,
            "emotion_ts": 0.0,
            "last_emotion_frame_idx": -10_000,
        }

    # -------------------------
    # EMOTION (FER) with caching
    # -------------------------
    def _get_emotion_for_track(
        self,
        frame_bgr_small: np.ndarray,
        track_id: int,
        now: float,
    ) -> Tuple[str, float]:
        tr = self._tracks.get(track_id)
        if tr is None:
            return "neutral", 0.5

        last_ts = float(tr.get("emotion_ts", 0.0))
        last_expr = str(tr.get("expression", "neutral"))
        last_conf = float(tr.get("confidence", 0.5))
        last_frame = int(tr.get("last_emotion_frame_idx", -10_000))

        # Time-based cache
        if (now - last_ts) <= self.EMOTION_CACHE_SECONDS:
            return last_expr, last_conf

        # Frame-based throttle
        should_run = (self._frame_idx % self.EMOTION_EVERY_N_FRAMES == 0)
        has_never_run = (last_frame < 0)

        if not should_run and not has_never_run:
            return last_expr, last_conf

        face_crop = self._crop_with_padding(frame_bgr_small, tr["bbox"])
        expr, conf = self._infer_emotion(face_crop)

        # store
        tr["expression"] = expr
        tr["confidence"] = conf
        tr["emotion_ts"] = now
        tr["last_emotion_frame_idx"] = self._frame_idx

        return expr, conf

    def _infer_emotion(self, face_bgr: Optional[np.ndarray]) -> Tuple[str, float]:
        # Safe defaults
        if face_bgr is None or face_bgr.size == 0:
            return "neutral", 0.5

        # Skip tiny faces (FER fails often)
        if face_bgr.shape[0] < self.MIN_FACE_SIZE or face_bgr.shape[1] < self.MIN_FACE_SIZE:
            return "neutral", 0.5

        try:
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            result = self.fer.top_emotion(face_rgb)

            # result can be: None, (None,None), ("happy", None)
            if not result or result[0] is None or result[1] is None:
                return "neutral", 0.5

            expr = str(result[0]).lower()
            conf = float(result[1])
            conf = float(np.clip(conf, 0.0, 1.0))
            return expr, conf

        except Exception:
            return "neutral", 0.5

    # -------------------------
    # METRICS MAPPING
    # -------------------------
    def _expression_to_state_and_metrics(self, expression: str, confidence: float) -> Tuple[str, float, float, float]:
        engaged = {"happy", "neutral"}
        confused = {"fear", "sad", "surprise"}
        distracted = {"angry", "disgust"}

        expression = str(expression).lower()
        confidence = float(np.clip(confidence, 0.0, 1.0))

        if expression in engaged:
            state = "Engaged"
            base = {"attention": 75.0, "confusion": 10.0, "distraction": 15.0}
            main_key = "attention"
        elif expression in confused:
            state = "Confused"
            base = {"attention": 45.0, "confusion": 40.0, "distraction": 15.0}
            main_key = "confusion"
        else:
            state = "Distracted"
            base = {"attention": 35.0, "confusion": 15.0, "distraction": 50.0}
            main_key = "distraction"

        boost = (confidence - 0.5) * 20.0

        a = base["attention"]
        c = base["confusion"]
        d = base["distraction"]

        if main_key == "attention":
            a += boost
            c -= boost / 2.0
            d -= boost / 2.0
        elif main_key == "confusion":
            c += boost
            a -= boost / 2.0
            d -= boost / 2.0
        else:
            d += boost
            a -= boost / 2.0
            c -= boost / 2.0

        a = float(np.clip(a, 0.0, 100.0))
        c = float(np.clip(c, 0.0, 100.0))
        d = float(np.clip(d, 0.0, 100.0))

        total = a + c + d
        if total <= 0:
            a, c, d = base["attention"], base["confusion"], base["distraction"]
            total = a + c + d

        a = a / total * 100.0
        c = c / total * 100.0
        d = d / total * 100.0

        a = round(a, 1)
        c = round(c, 1)
        d = round(d, 1)

        # rounding drift fix
        drift = round(100.0 - (a + c + d), 1)
        if main_key == "attention":
            a = round(a + drift, 1)
        elif main_key == "confusion":
            c = round(c + drift, 1)
        else:
            d = round(d + drift, 1)

        state_str = f"{state} ({confidence:.2f})"
        return state_str, float(a), float(c), float(d)

    # -------------------------
    # UTILS
    # -------------------------
    def _crop_with_padding(self, frame_bgr: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        h_img, w_img = frame_bgr.shape[:2]
        x, y, w, h = bbox

        pad_x = int(w * self.CROP_PADDING_RATIO)
        pad_y = int(h * self.CROP_PADDING_RATIO)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w_img, x + w + pad_x)
        y2 = min(h_img, y + h + pad_y)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        return crop

    def _iou(self, a: List[int], b: List[int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b

        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh

        ix1 = max(ax, bx)
        iy1 = max(ay, by)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)

        inter = iw * ih
        if inter <= 0:
            return 0.0

        area_a = aw * ah
        area_b = bw * bh
        denom = float(area_a + area_b - inter)
        return float(inter / denom) if denom > 0 else 0.0


if __name__ == "__main__":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Camera not opened")
        raise SystemExit(1)

    engine = RealTimeEmotionEngine(
        yolo_model_path="yolov8n-face.pt",  # keep in same folder
        detect_every_n_frames=3,
        emotion_every_n_frames=8,
        emotion_cache_seconds=1.0,
        resize_width=640,
        resize_height=360,
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break

        data = engine.process_frame(frame)
        print(data)

        debug = engine.draw_debug(frame)
        cv2.imshow("YOLO + FER Emotion Tracker (Smooth)", debug)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
