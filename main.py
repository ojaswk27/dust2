"""
main.py
-------
PrintGuard – Raspberry Pi 5 print failure monitor for Bambu Lab FDM printers.

Entry point.  Ties together:
  - camera.FrameSource      – continuous frame capture
  - detector.PrintFailureDetector – Obico ONNX model + CV heuristics
  - bambu_mqtt.BambuMQTTClient    – local MQTT printer control
  - advisor.format_suggestions    – human-readable fix recommendations
  - Optional webhook notification & sound beep

Usage:
    python main.py [--config config.yaml]
    python main.py --dry-run          # analyse frames but do NOT send commands
    python main.py --once <image.jpg> # run on a single image and exit
"""

from __future__ import annotations

import argparse
import cv2
import datetime
import json
import logging
import logging.handlers
import os
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional

import yaml  # PyYAML

from bambu_mqtt import BambuMQTTClient
from camera import FrameSource
from detector import Detection, PrintFailureDetector
from llm_advisor import PrinterStateStore, get_llm_advice


# ── Logging setup ─────────────────────────────────────────────────────────────


def setup_logging(log_cfg: dict):
    level_name = log_cfg.get("level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_dir = Path(log_cfg.get("log_dir", "logs"))
    log_dir.mkdir(exist_ok=True)
    max_bytes = int(log_cfg.get("max_size_mb", 10)) * 1024 * 1024

    root = logging.getLogger()
    root.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    # Console
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    # Rotating file
    fh = logging.handlers.RotatingFileHandler(
        log_dir / "printguard.log",
        maxBytes=max_bytes,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)


logger = logging.getLogger(__name__)


# ── Config loading ─────────────────────────────────────────────────────────────


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(p) as f:
        cfg = yaml.safe_load(f)
    # Allow a local override file that wins over the committed defaults
    local = p.parent / "config.local.yaml"
    if local.exists():
        with open(local) as f:
            override = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, override)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ── Snapshot saving ───────────────────────────────────────────────────────────


def save_snapshot(frame, snapshot_dir: str, label: str = "failure") -> Path:
    d = Path(snapshot_dir)
    d.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = d / f"{ts}_{label}.jpg"
    cv2.imwrite(str(path), frame)
    return path


# ── Webhook notification ──────────────────────────────────────────────────────


def send_webhook(url: str, payload: dict):
    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5):
            pass
        logger.info("Webhook sent to %s", url)
    except Exception as exc:
        logger.warning("Webhook failed: %s", exc)


# ── Beep ─────────────────────────────────────────────────────────────────────


def beep():
    """Best-effort terminal bell."""
    try:
        sys.stdout.write("\a")
        sys.stdout.flush()
    except Exception:
        pass


# ── Debounce counter ─────────────────────────────────────────────────────────


class DebounceWindow:
    """
    Fires only after `confirm_frames` consecutive frames all contain a
    detection. Resets when a clean frame is seen.
    """

    def __init__(self, confirm_frames: int = 3):
        self._required = confirm_frames
        self._count = 0
        self._fired = False  # prevent repeated alerts for the same failure

    def update(self, detections: List[Detection]) -> bool:
        """Returns True the first time the threshold is crossed."""
        if detections:
            self._count += 1
            if self._count >= self._required and not self._fired:
                self._fired = True
                return True
        else:
            self._count = 0
            self._fired = False  # reset so a new failure can trigger again
        return False


# ── Single-image mode ─────────────────────────────────────────────────────────


def run_once(image_path: str, detector: PrintFailureDetector, cfg: dict):
    frame = cv2.imread(image_path)
    if frame is None:
        logger.error("Could not read image: %s", image_path)
        sys.exit(1)
    detections = detector.detect(frame)
    if not detections:
        print("No failures detected in image.")
        return
    all_labels = _collect_labels(detections)
    print(f"\nDetected failures: {all_labels}\n")
    advice = get_llm_advice(all_labels, {}, cfg.get("advisor", {}))
    print(advice)


def _collect_labels(detections: List[Detection]) -> List[str]:
    labels: List[str] = []
    for det in detections:
        if det.label not in labels:
            labels.append(det.label)
        for hl in det.heuristic_labels:
            if hl not in labels:
                labels.append(hl)
    return labels


# ── Main loop ─────────────────────────────────────────────────────────────────


def run(cfg: dict, dry_run: bool = False):
    cam_cfg = cfg.get("camera", {})
    model_cfg = cfg.get("model", {})
    bambu_cfg = cfg.get("bambu", {})
    notif_cfg = cfg.get("notifications", {})
    advisor_cfg = cfg.get("advisor", {})

    confirm_frames = model_cfg.get("confirm_frames", 3)
    save_snaps = cam_cfg.get("save_snapshots", True)
    snapshot_dir = cam_cfg.get("snapshot_dir", "snapshots")

    # Load model
    detector = PrintFailureDetector(model_cfg)
    detector.load()

    debounce = DebounceWindow(confirm_frames)

    # State store receives live printer telemetry via the MQTT on_status callback
    state_store = PrinterStateStore()

    if dry_run:
        logger.warning("DRY RUN mode – printer commands will NOT be sent")
        printer: Optional[BambuMQTTClient] = None
    else:
        printer = BambuMQTTClient(bambu_cfg, on_status=state_store.update)
        printer.connect()
        # Ask the printer to push its full current state immediately
        printer.request_full_status()

    try:
        with FrameSource(cam_cfg) as cam:
            logger.info(
                "PrintGuard monitoring started (confirm_frames=%d, action=%s)",
                confirm_frames,
                "DRY_RUN" if dry_run else bambu_cfg.get("on_failure", "pause"),
            )
            for frame in cam:
                detections = detector.detect(frame)
                frame_has_failure = len(detections) > 0

                if frame_has_failure:
                    all_labels = _collect_labels(detections)
                    logger.info("Frame: detections=%s", [str(d) for d in detections])

                should_act = debounce.update(detections)
                if not should_act:
                    continue

                # ── Confirmed failure ─────────────────────────────────────
                all_labels = _collect_labels(detections)
                logger.warning(
                    "FAILURE CONFIRMED after %d frames: %s", confirm_frames, all_labels
                )

                # Save snapshot
                snap_path: Optional[Path] = None
                if save_snaps:
                    snap_path = save_snapshot(
                        frame, snapshot_dir, label="_".join(all_labels)
                    )
                    logger.info("Snapshot saved: %s", snap_path)

                # Send printer command first, then get advice (LLM may take a moment)
                if printer:
                    printer.on_failure_detected()
                else:
                    logger.warning("[DRY RUN] Would have sent printer command here")

                # Get AI advice with live printer state
                advice = get_llm_advice(
                    all_labels,
                    state_store.snapshot(),
                    advisor_cfg,
                )
                logger.warning("\n%s", advice)

                # Sound
                if notif_cfg.get("beep_on_failure", True):
                    beep()

                # Webhook
                if notif_cfg.get("webhook_enabled", False):
                    payload = {
                        "event": "print_failure",
                        "labels": all_labels,
                        "snapshot": str(snap_path) if snap_path else None,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "suggestions": advice,
                    }
                    send_webhook(notif_cfg["webhook_url"], payload)

    except KeyboardInterrupt:
        logger.info("Stopped by user (Ctrl+C)")
    finally:
        if printer:
            printer.disconnect()
        logger.info("PrintGuard shut down")


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="PrintGuard – 3D print failure monitor"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML (default: config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run detection but do not send any command to the printer",
    )
    parser.add_argument(
        "--once",
        metavar="IMAGE",
        help="Run detection on a single image file and print suggestions, then exit",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}))

    if args.once:
        model_cfg = cfg.get("model", {})
        detector = PrintFailureDetector(model_cfg)
        detector.load()
        run_once(args.once, detector, cfg)
    else:
        run(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
