"""
detector.py
-----------
Wraps the Obico / The-Spaghetti-Detective YOLOv4-tiny model.

The model produces a single class: "failure" (spaghetti / detached print).
We run it through OpenCV's DNN module (ONNX backend) so we don't need
darknet compiled on the Pi.

Detection result:
    DetectionResult(label, confidence, bbox)
    bbox = (x_center, y_center, width, height) normalised 0–1

The detector also does basic OpenCV heuristics for failure types that the
Obico weights don't distinguish (layer shift, bed adhesion loss, etc.).
When the model fires we run the heuristics to narrow down the failure type
so the advisor can give more specific suggestions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Label the Obico model outputs (single class)
OBICO_LABEL = "spaghetti"

# Heuristic-only labels (not from Obico weights)
HEURISTIC_LABELS = {
    "layer_shift",
    "bed_adhesion_loss",
    "under_extrusion",
    "over_extrusion",
    "blob_stringing",
}


@dataclass
class Detection:
    label: str
    confidence: float
    # (x1, y1, x2, y2) in pixels
    box: Tuple[int, int, int, int] = field(default_factory=lambda: (0, 0, 0, 0))
    # Extra heuristic classifications discovered on the same frame
    heuristic_labels: List[str] = field(default_factory=list)

    def __str__(self):
        extra = (
            f" [{', '.join(self.heuristic_labels)}]" if self.heuristic_labels else ""
        )
        x1, y1, x2, y2 = self.box
        return (
            f"Detection(label={self.label}, conf={self.confidence:.2f}, "
            f"box=({x1},{y1})-({x2},{y2}){extra})"
        )


class PrintFailureDetector:
    """
    Loads the Obico ONNX model and runs inference + CV heuristics on frames.
    """

    # Obico model input size
    INPUT_W = 416
    INPUT_H = 416

    def __init__(self, cfg: dict):
        self._cfg = cfg
        self._conf_thresh: float = cfg.get("confidence_threshold", 0.3)
        self._nms_thresh: float = cfg.get("nms_threshold", 0.4)
        self._force_cpu: bool = cfg.get("force_cpu", False)
        self._net: cv2.dnn.Net | None = None
        self._output_layers: list[str] = []
        self._loaded = False

    # ── lifecycle ────────────────────────────────────────────────────────────

    def load(self):
        weights = Path(self._cfg.get("weights", "model/model.onnx"))
        if not weights.exists():
            raise FileNotFoundError(
                f"Model weights not found at {weights}. "
                "Run: python scripts/download_model.py"
            )

        logger.info("Loading model from %s", weights)
        self._net = cv2.dnn.readNetFromONNX(str(weights))

        if self._force_cpu:
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            logger.info("Model forced to CPU")
        else:
            # Use OpenCV default – will pick up NEON on Pi 5 automatically
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Grab output layer names from the network
        layer_names = self._net.getLayerNames()
        unconnected = self._net.getUnconnectedOutLayers()
        # OpenCV 4.x returns flat array
        if isinstance(unconnected[0], (list, np.ndarray)):
            self._output_layers = [layer_names[i[0] - 1] for i in unconnected]
        else:
            self._output_layers = [layer_names[i - 1] for i in unconnected]

        self._loaded = True
        logger.info("Model loaded. Output layers: %s", self._output_layers)

    # ── inference ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run the full detection pipeline on a BGR frame.
        Returns a (possibly empty) list of Detection objects.
        """
        if not self._loaded:
            raise RuntimeError("Call load() before detect()")

        h, w = frame.shape[:2]

        # --- Obico model inference ---
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1 / 255.0,
            size=(self.INPUT_W, self.INPUT_H),
            swapRB=False,  # already BGR
            crop=False,
        )
        self._net.setInput(blob)
        raw_outputs = self._net.forward(self._output_layers)

        boxes, confidences = [], []
        for output in raw_outputs:
            for detection in output:
                # detection layout: [cx, cy, bw, bh, obj_conf, class_conf…]
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id]) * float(detection[4])
                if confidence < self._conf_thresh:
                    continue
                cx = int(detection[0] * w)
                cy = int(detection[1] * h)
                bw = int(detection[2] * w)
                bh = int(detection[3] * h)
                x1 = max(0, cx - bw // 2)
                y1 = max(0, cy - bh // 2)
                boxes.append([x1, y1, bw, bh])
                confidences.append(confidence)

        # NMS
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self._conf_thresh, self._nms_thresh
        )

        detections: List[Detection] = []
        if len(indices) > 0:
            flat_idx = indices.flatten() if hasattr(indices, "flatten") else indices
            for i in flat_idx:
                x1, y1, bw, bh = boxes[i]
                det = Detection(
                    label=OBICO_LABEL,
                    confidence=confidences[i],
                    box=(x1, y1, x1 + bw, y1 + bh),
                )
                # Enrich with heuristic classification
                det.heuristic_labels = self._heuristic_classify(frame, det)
                detections.append(det)

        # Run heuristics even when Obico fires nothing (catches different failure
        # modes like layer shift that the spaghetti detector misses entirely)
        standalone = self._heuristic_standalone(frame)
        detections.extend(standalone)

        return detections

    # ── heuristics ───────────────────────────────────────────────────────────

    def _heuristic_classify(self, frame: np.ndarray, det: Detection) -> List[str]:
        """Given an existing Obico detection, narrow it down further."""
        labels: List[str] = []
        roi = frame[det.box[1] : det.box[3], det.box[0] : det.box[2]]
        if roi.size == 0:
            return labels

        # Fine strands → stringing / blob
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / edges.size
        if edge_density > 0.15:
            labels.append("blob_stringing")

        return labels

    def _heuristic_standalone(self, frame: np.ndarray) -> List[Detection]:
        """
        Purely heuristic detections not covered by Obico weights.
        Each heuristic is deliberately conservative to avoid false positives.
        """
        results: List[Detection] = []
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Layer shift ──────────────────────────────────────────────────────
        # Look for a strong, roughly-horizontal discontinuity line across the
        # centre of the print area.  We compare the left half vs right half
        # of the horizontal Sobel response.
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobelx_abs = np.abs(sobelx)
        strip = sobelx_abs[h // 4 : 3 * h // 4, :]
        col_means = strip.mean(axis=0)
        # A layer shift creates a strong vertical band of high X-gradient
        peak = float(col_means.max())
        if peak > 80:
            # Verify continuity: at least 40 px of consecutive high-gradient cols
            high_cols = (col_means > peak * 0.6).astype(np.uint8)
            kernel = np.ones(40, np.uint8)
            dilated = np.convolve(high_cols, kernel, mode="same")
            if dilated.max() >= 40:
                cx = int(col_means.argmax())
                results.append(
                    Detection(
                        label="layer_shift",
                        confidence=min(1.0, peak / 200.0),
                        box=(max(0, cx - 20), h // 4, min(w, cx + 20), 3 * h // 4),
                    )
                )

        # ── Bed adhesion loss ────────────────────────────────────────────────
        # The bottom strip of the frame should have consistent colour/texture
        # when a print is attached. A detached print leaves a bare bed visible.
        bottom_strip = gray[int(h * 0.85) :, :]
        # High variance in the bottom strip = unusual texture = possible detach
        variance = float(bottom_strip.var())
        if variance > 1800:
            results.append(
                Detection(
                    label="bed_adhesion_loss",
                    confidence=min(1.0, variance / 5000.0),
                    box=(0, int(h * 0.85), w, h),
                )
            )

        # ── Under / over extrusion ───────────────────────────────────────────
        # Analyse brightness of the printed area vs expected range.
        # Over-extrusion → unusually bright blobs; under → dark gaps.
        print_roi = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        mean_brightness = float(print_roi.mean())
        if mean_brightness > 220:
            results.append(
                Detection(
                    label="over_extrusion",
                    confidence=min(1.0, (mean_brightness - 220) / 35.0),
                    box=(w // 4, h // 4, 3 * w // 4, 3 * h // 4),
                )
            )
        elif mean_brightness < 30:
            results.append(
                Detection(
                    label="under_extrusion",
                    confidence=min(1.0, (30 - mean_brightness) / 30.0),
                    box=(w // 4, h // 4, 3 * w // 4, 3 * h // 4),
                )
            )

        return results
