import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
from fer import FER
from ultralytics import YOLO


class RealTimeEmotionEngine:
    """
    YOLO (face detection) + FER (emotion)
    Outputs per student:
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
        iou_threshold: float = 0.3,
        track_timeout: float = 2.0,
        crop_padding: float = 0.2,
    ):
        # YOLO face detector
        self.detector = YOLO(yolo_model_path)

        # FER emotion model
        self.fer = FER(mtcnn=False)

        self.IOU_THRESHOLD = iou_threshold
        self.TRACK_TIMEOUT = track_timeout
        self.CROP_PADDING = crop_padding

        self._tracks: Dict[int, Dict] = {}
        self._next_track_id = 1
        self._next_student_num = 1

    # ---------------- MAIN ----------------
    def process_frame(self, frame: np.ndarray) -> List[dict]:
        now = time.time()

        detections = self._detect_faces(frame)
        self._update_tracks(detections, now)

        outputs = []
        for tid, tr in self._tracks.items():
            x, y, w, h = tr["bbox"]
            face = self._crop_face(frame, tr["bbox"])

            expr, conf = self._infer_emotion(face)
            state, att, confu, dist = self._map_metrics(expr, conf)

            tr["expression"] = expr
            tr["confidence"] = conf

            outputs.append(
                {
                    "student_id": tr["student_id"],
                    "expression": expr,
                    "confidence": conf,
                    "state": state,
                    "attention": att,
                    "confusion": confu,
                    "distraction": dist,
                }
            )

        return outputs

    def draw(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        for tr in self._tracks.values():
            x, y, w, h = tr["bbox"]
            label = f'{tr["student_id"]} {tr.get("expression","")} {tr.get("confidence",0):.2f}'
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(out, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return out

    # ---------------- FACE DETECTION ----------------
    def _detect_faces(self, frame: np.ndarray) -> List[List[int]]:
        results = self.detector(frame, conf=0.4, verbose=False)[0]

        detections = []
        if results.boxes is None:
            return detections

        for box in results.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box[:4])
            w, h = x2 - x1, y2 - y1
            if w > 30 and h > 30:
                detections.append([x1, y1, w, h])

        return detections

    # ---------------- TRACKING ----------------
    def _update_tracks(self, detections: List[List[int]], now: float):
        expired = [tid for tid, tr in self._tracks.items()
                   if now - tr["last_seen"] > self.TRACK_TIMEOUT]
        for tid in expired:
            del self._tracks[tid]

        if not self._tracks:
            for d in detections:
                self._create_track(d, now)
            return

        for det in detections:
            matched = False
            for tr in self._tracks.values():
                if self._iou(det, tr["bbox"]) > self.IOU_THRESHOLD:
                    tr["bbox"] = det
                    tr["last_seen"] = now
                    matched = True
                    break
            if not matched:
                self._create_track(det, now)

    def _create_track(self, bbox, now):
        tid = self._next_track_id
        self._next_track_id += 1

        sid = f"student_{self._next_student_num:02d}"
        self._next_student_num += 1

        self._tracks[tid] = {
            "bbox": bbox,
            "last_seen": now,
            "student_id": sid,
            "expression": "neutral",
            "confidence": 0.5,
        }

    # ---------------- EMOTION ----------------
    def _infer_emotion(self, face):
    # invalid crop
     if face is None or face.size == 0:
        return "neutral", 0.5

    # too small face -> FER often fails
     if face.shape[0] < 40 or face.shape[1] < 40:
        return "neutral", 0.5

     face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

     result = self.fer.top_emotion(face_rgb)

    # FER can return:
    # None
    # (None, None)
    # ("happy", None)
     if not result or result[0] is None or result[1] is None:
        return "neutral", 0.5

     expr = str(result[0]).lower()
     conf = float(result[1])

    # keep it safe
     conf = max(0.0, min(conf, 1.0))
     return expr, conf


    # ---------------- METRICS ----------------
    def _map_metrics(self, expr, conf):
        engaged = {"happy", "neutral"}
        confused = {"sad", "fear", "surprise"}

        if expr in engaged:
            state = "Engaged"
            a, c, d = 75, 10, 15
            main = "a"
        elif expr in confused:
            state = "Confused"
            a, c, d = 45, 40, 15
            main = "c"
        else:
            state = "Distracted"
            a, c, d = 35, 15, 50
            main = "d"

        boost = (conf - 0.5) * 20
        if main == "a":
            a += boost; c -= boost / 2; d -= boost / 2
        elif main == "c":
            c += boost; a -= boost / 2; d -= boost / 2
        else:
            d += boost; a -= boost / 2; c -= boost / 2

        a, c, d = map(lambda x: max(0, x), (a, c, d))
        s = a + c + d
        a, c, d = [round(x / s * 100, 1) for x in (a, c, d)]

        return f"{state} ({conf:.2f})", a, c, d

    # ---------------- UTILS ----------------
    def _crop_face(self, frame, bbox):
        x, y, w, h = bbox
        pad = int(w * self.CROP_PADDING)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)
        return frame[y1:y2, x1:x2]

    def _iou(self, a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh

        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)

        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0


# ---------------- RUN ----------------
if __name__ == "__main__":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    engine = RealTimeEmotionEngine()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        data = engine.process_frame(frame)
        print(data)

        cv2.imshow("YOLO + FER Emotion Tracker", engine.draw(frame))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
