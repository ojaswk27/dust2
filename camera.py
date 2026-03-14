"""
camera.py
---------
Provides a unified frame-capture iterator that works with:
  - Raspberry Pi camera module (picamera2 library)
  - Any USB/V4L2 webcam (OpenCV VideoCapture)

Usage:
    from camera import FrameSource
    with FrameSource(cfg) as cam:
        for frame in cam:
            # frame is a BGR numpy array (OpenCV convention)
            process(frame)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Generator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameSource:
    """Context-manager that yields BGR frames at the configured rate."""

    def __init__(self, cfg: dict):
        self._cfg = cfg
        self._backend: str = cfg.get("backend", "picamera2")
        self._width: int = cfg.get("width", 1280)
        self._height: int = cfg.get("height", 720)
        self._fps: float = cfg.get("fps", 2)
        self._interval: float = 1.0 / max(self._fps, 0.1)
        self._cap = None  # OpenCV capture (USB backend)
        self._picam = None  # picamera2 instance

    # ── context manager ──────────────────────────────────────────────────────

    def __enter__(self) -> "FrameSource":
        self._open()
        return self

    def __exit__(self, *_):
        self._close()

    # ── public iterator ──────────────────────────────────────────────────────

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        """Yields BGR frames throttled to the configured FPS."""
        while True:
            t0 = time.monotonic()
            frame = self._grab()
            if frame is not None:
                yield frame
            elapsed = time.monotonic() - t0
            sleep_for = self._interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    # ── internals ────────────────────────────────────────────────────────────

    def _open(self):
        if self._backend == "picamera2":
            self._open_picamera2()
        else:
            self._open_usb()

    def _open_picamera2(self):
        try:
            from picamera2 import Picamera2  # type: ignore

            self._picam = Picamera2()
            config = self._picam.create_still_configuration(
                main={"size": (self._width, self._height), "format": "BGR888"}
            )
            self._picam.configure(config)
            self._picam.start()
            logger.info("picamera2 opened (%dx%d)", self._width, self._height)
        except Exception as exc:
            raise RuntimeError(
                "Failed to open picamera2. Is the camera module connected and enabled? "
                f"Original error: {exc}"
            ) from exc

    def _open_usb(self):
        idx = self._cfg.get("usb_index", 0)
        self._cap = cv2.VideoCapture(idx)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open USB camera at index {idx}. "
                "Check that the device is connected."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        logger.info("USB camera %d opened (%dx%d)", idx, self._width, self._height)

    def _close(self):
        if self._picam is not None:
            try:
                self._picam.stop()
                self._picam.close()
            except Exception:
                pass
            self._picam = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("Camera released")

    def _grab(self) -> np.ndarray | None:
        if self._backend == "picamera2" and self._picam is not None:
            return self._picam.capture_array("main")

        if self._cap is not None:
            ok, frame = self._cap.read()
            if not ok:
                logger.warning("Failed to read frame from USB camera, retrying…")
                return None
            return frame

        return None
