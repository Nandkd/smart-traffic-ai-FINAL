"""
ml_models/yolo/detect.py
========================
Real-time YOLOv8 inference pipeline with multi-object tracking.

Usage:
    python detect.py --source 0              # webcam
    python detect.py --source video.mp4      # video file
    python detect.py --source image.jpg      # single image
    python detect.py --source rtsp://...     # IP camera stream
"""

import argparse
import time
import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict, deque

import torch
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent.parent
WEIGHTS = ROOT / "ml_models" / "weights" / "yolov8_traffic.pt"

COLORS = {
    "car": (0, 200, 0),
    "motorcycle": (255, 165, 0),
    "bus": (0, 0, 255),
    "truck": (128, 0, 128),
    "ambulance": (0, 0, 255),
}
EMERGENCY_COLOR = (0, 0, 255)
CLASS_NAMES = ["car", "motorcycle", "bus", "truck", "ambulance"]


class TrafficDetector:
    """
    Real-time traffic detector with ByteTrack multi-object tracking.
    """

    def __init__(self, weights: str = None, conf: float = 0.4, iou: float = 0.5):
        w = weights or str(WEIGHTS)
        if not Path(w).exists():
            print(f"⚠️  Custom weights not found. Using pretrained yolov8n.pt")
            w = "yolov8n.pt"
        self.model = YOLO(w)
        self.conf = conf
        self.iou = iou
        self.device = "0" if torch.cuda.is_available() else "cpu"

        # Per-class vehicle counts per lane zone
        self.frame_counts = defaultdict(int)
        self.track_history = defaultdict(lambda: deque(maxlen=30))

        # FPS
        self._fps_times = deque(maxlen=30)

    def _estimate_lane(self, cx: int, frame_w: int) -> str:
        """Simple vertical-split lane estimation."""
        quarter = frame_w // 4
        if cx < quarter:
            return "west"
        elif cx < 2 * quarter:
            return "south"
        elif cx < 3 * quarter:
            return "north"
        else:
            return "east"

    def infer_frame(self, frame: np.ndarray) -> dict:
        """Run detection + tracking on a single BGR frame."""
        t0 = time.time()
        results = self.model.track(
            frame,
            conf=self.conf,
            iou=self.iou,
            persist=True,
            tracker="bytetrack.yaml",
            device=self.device,
            verbose=False,
        )
        elapsed = (time.time() - t0) * 1000

        r = results[0]
        detections = []
        class_counts = defaultdict(int)
        lane_counts = defaultdict(int)
        ambulance_detected = False

        if r.boxes is not None and len(r.boxes):
            for box in r.boxes:
                cls_id = int(box.cls)
                cls_name = r.names.get(cls_id, "unknown")
                conf_val = float(box.conf)
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                track_id = int(box.id) if box.id is not None else -1
                cx = (x1 + x2) // 2

                lane = self._estimate_lane(cx, frame.shape[1])
                class_counts[cls_name] += 1
                lane_counts[lane] += 1

                if cls_name == "ambulance" and conf_val > 0.5:
                    ambulance_detected = True

                # Track centroid history
                if track_id > 0:
                    self.track_history[track_id].append((cx, (y1 + y2) // 2))

                detections.append({
                    "class": cls_name,
                    "confidence": round(conf_val, 3),
                    "bbox": [x1, y1, x2, y2],
                    "track_id": track_id,
                    "lane": lane,
                })

        total = sum(class_counts.values())
        density = "low" if total < 10 else ("medium" if total < 25 else "high")

        return {
            "detections": detections,
            "class_counts": dict(class_counts),
            "lane_counts": dict(lane_counts),
            "total_vehicles": total,
            "density_class": density,
            "ambulance_detected": ambulance_detected,
            "inference_ms": round(elapsed, 1),
        }

    def draw_results(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """Annotate frame with bounding boxes, labels, and HUD."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        for det in result["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            cls = det["class"]
            tid = det["track_id"]
            conf = det["confidence"]
            color = EMERGENCY_COLOR if cls == "ambulance" else COLORS.get(cls, (180, 180, 180))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw track trail
            if tid > 0 and tid in self.track_history:
                pts = list(self.track_history[tid])
                for i in range(1, len(pts)):
                    if pts[i - 1] and pts[i]:
                        cv2.line(annotated, pts[i - 1], pts[i], color, 1)

            label = f"#{tid} {cls} {conf:.2f}" if tid > 0 else f"{cls} {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - lh - 6), (x1 + lw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # ── HUD overlay ───────────────────────────────────────
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (270, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)

        density_colors = {"low": (0, 200, 0), "medium": (0, 165, 255), "high": (0, 0, 255)}
        dc = result["density_class"]
        y_pos = 20
        cv2.putText(annotated, f"Vehicles: {result['total_vehicles']}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += 25
        cv2.putText(annotated, f"Density: {dc.upper()}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, density_colors[dc], 2)
        y_pos += 25
        cv2.putText(annotated, f"FPS: {1000 / max(result['inference_ms'], 1):.1f}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_pos += 25
        for cls_name, cnt in result["class_counts"].items():
            cv2.putText(annotated, f"  {cls_name}: {cnt}", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            y_pos += 20

        if result["ambulance_detected"]:
            cv2.putText(annotated, "🚨 AMBULANCE — EMERGENCY OVERRIDE", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return annotated


def run(source, show: bool = True, save: bool = False, output: str = "output.mp4"):
    """Main inference loop."""
    detector = TrafficDetector()

    cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
    assert cap.isOpened(), f"Cannot open source: {source}"

    writer = None
    if save:
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w, h))

    print("▶️  Inference started. Press Q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.infer_frame(frame)
        annotated = detector.draw_results(frame, result)

        if save and writer:
            writer.write(annotated)
        if show:
            cv2.imshow("Traffic AI — Press Q to quit", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("✅ Inference complete.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="Video source (0=webcam, path, rtsp)")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--output", default="output.mp4")
    args = ap.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    run(src, show=not args.no_show, save=args.save, output=args.output)
