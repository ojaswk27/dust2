"""
demo.py
-------
PrintGuard video demo — run failure detection on a local video file.

No printer connection required. Works on any machine with OpenCV installed.
Designed for the Raspberry Pi Cam 3 output resolution but handles any input.

Usage:
    python demo.py <video_file> [options]

Examples:
    python demo.py sample_fail.mp4
    python demo.py sample_fail.mp4 --every 3 --conf 0.25
    python demo.py sample_fail.mp4 --no-display          # headless / SSH
    python demo.py sample_fail.mp4 --weights model/model.onnx --confirm 2

Controls (when display is open):
    Space     pause / unpause
    →         step one frame while paused
    s         save current annotated frame as snapshot manually
    q / Esc   quit

──────────────────────────────────────────────────────────────────────────────
No sample video? Grab one with yt-dlp:

    pip install yt-dlp

    # Classic spaghetti failure timelapse (CC / public):
    yt-dlp -o sample_fail.mp4 "https://www.youtube.com/watch?v=D6_VCBMEiMo"

    # Or search YouTube for: "3d print spaghetti failure timelapse"
    # Or browse the Obico public gallery: https://app.obico.io/ent/publics/
    # Or Reddit r/3Dprinting filtered by flair: "fail"
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import datetime
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ── Reuse existing modules (no MQTT / camera / config needed) ─────────────────
from detector import Detection, PrintFailureDetector
from advisor import format_suggestions

# Import helpers from main without triggering the __main__ block
from main import DebounceWindow, _collect_labels, save_snapshot


# ── Colours (BGR) ─────────────────────────────────────────────────────────────
C_RED = (0, 0, 220)
C_YELLOW = (0, 210, 255)
C_GREEN = (0, 150, 0)
C_DARK_RED = (0, 0, 160)
C_DARK_GRN = (0, 100, 0)
C_WHITE = (255, 255, 255)
C_BLACK = (0, 0, 0)
C_ORANGE = (0, 140, 255)
C_GREY = (80, 80, 80)


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PrintGuard demo — run failure detection on a video file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Sample video tip:\n"
            "  pip install yt-dlp\n"
            "  yt-dlp -o sample_fail.mp4 "
            '"https://www.youtube.com/watch?v=D6_VCBMEiMo"\n'
            "  Or search YouTube: '3d print spaghetti failure timelapse'"
        ),
    )
    parser.add_argument("video", help="Path to video file (.mp4, .avi, .mov, …)")
    parser.add_argument(
        "--every",
        type=int,
        default=5,
        metavar="N",
        help="Run detector every N frames (default: 5)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        metavar="FLOAT",
        help="Confidence threshold 0–1 (default: 0.3)",
    )
    parser.add_argument(
        "--confirm",
        type=int,
        default=3,
        metavar="N",
        help="Consecutive positive frames before alerting (default: 3)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="model/model.onnx",
        help="Path to ONNX weights (default: model/model.onnx)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Headless mode — no window, just terminal output + snapshots",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1280,
        help="Max display window width in px (default: 1280)",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=720,
        help="Max display window height in px (default: 720)",
    )
    return parser.parse_args()


# ── Frame scaling ─────────────────────────────────────────────────────────────


def scale_frame(frame: np.ndarray, max_w: int, max_h: int) -> Tuple[np.ndarray, float]:
    """
    Scale frame down so it fits within max_w × max_h, preserving aspect ratio.
    Returns (scaled_frame, scale_factor).  scale_factor < 1 means it was shrunk.
    """
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale == 1.0:
        return frame, 1.0
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


# ── Overlay drawing ───────────────────────────────────────────────────────────


def _text_with_shadow(
    img: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    scale: float = 0.55,
    thickness: int = 1,
    colour=C_WHITE,
):
    """Draw text with a 1-pixel black drop shadow for readability on any bg."""
    x, y = pos
    cv2.putText(
        img,
        text,
        (x + 1, y + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        C_BLACK,
        thickness + 1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        colour,
        thickness,
        cv2.LINE_AA,
    )


def draw_overlay(
    frame: np.ndarray,
    detections: List[Detection],
    debounce_count: int,
    confirm_required: int,
    frame_idx: int,
    total_frames: int,
    display_fps: float,
    paused: bool,
    scale: float,
) -> np.ndarray:
    """
    Draw all overlay elements onto a copy of the (already-scaled) frame.
    Returns the annotated frame.
    """
    out = frame.copy()
    h, w = out.shape[:2]
    banner_h = 34

    has_failure = len(detections) > 0

    # ── 1. Top status banner ─────────────────────────────────────────────────
    banner_colour = C_DARK_RED if has_failure else C_DARK_GRN
    cv2.rectangle(out, (0, 0), (w, banner_h), banner_colour, -1)

    if has_failure:
        labels = _collect_labels(detections)
        banner_text = f"FAILURE: {', '.join(labels)}"
        text_colour = (80, 80, 255)  # light red-ish
    else:
        banner_text = "OK  —  no failures detected"
        text_colour = (100, 255, 100)  # light green

    _text_with_shadow(
        out, banner_text, (10, 23), scale=0.65, thickness=1, colour=text_colour
    )

    # ── 2. Bounding boxes + labels ───────────────────────────────────────────
    for det in detections:
        x1, y1, x2, y2 = [int(v * scale) for v in det.box]
        # Obico model → red; pure heuristic → yellow
        box_colour = C_RED if det.label == "spaghetti" else C_YELLOW
        cv2.rectangle(out, (x1, y1 + banner_h), (x2, y2 + banner_h), box_colour, 2)

        conf_pct = int(det.confidence * 100)
        tag = f"{det.label} {conf_pct}%"
        tag_y = max(y1 + banner_h - 6, banner_h + 14)
        _text_with_shadow(out, tag, (x1 + 2, tag_y), scale=0.5, colour=box_colour)

        # Heuristic sub-labels (from Obico detection enrichment)
        for i, hl in enumerate(det.heuristic_labels):
            _text_with_shadow(
                out,
                f"  +{hl}",
                (x1 + 2, tag_y + 16 * (i + 1)),
                scale=0.42,
                colour=C_YELLOW,
            )

    # ── 3. Frame counter (bottom-left) ───────────────────────────────────────
    total_str = str(total_frames) if total_frames > 0 else "?"
    frame_text = f"frame {frame_idx} / {total_str}   {display_fps:.1f} fps"
    _text_with_shadow(out, frame_text, (8, h - 10), scale=0.48)

    # ── 4. Debounce indicator (bottom-right) ─────────────────────────────────
    if debounce_count > 0 and debounce_count < confirm_required:
        db_text = f"confirming: {debounce_count} / {confirm_required}"
        (tw, _), _ = cv2.getTextSize(db_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        _text_with_shadow(
            out, db_text, (w - tw - 10, h - 10), scale=0.48, colour=C_ORANGE
        )

    # ── 5. Paused indicator (centre) ─────────────────────────────────────────
    if paused:
        label = "[ PAUSED ]"
        font_scale = 1.4
        thick = 2
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick
        )
        cx = (w - tw) // 2
        cy = (h + th) // 2
        # semi-transparent dark backdrop
        overlay = out.copy()
        pad = 14
        cv2.rectangle(
            overlay, (cx - pad, cy - th - pad), (cx + tw + pad, cy + pad), C_GREY, -1
        )
        cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
        cv2.putText(
            out,
            label,
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            C_WHITE,
            thick,
            cv2.LINE_AA,
        )

    return out


# ── Main processing loop ──────────────────────────────────────────────────────


def process_video(
    cap: cv2.VideoCapture,
    detector: PrintFailureDetector,
    args: argparse.Namespace,
):
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_dur = 1.0 / video_fps  # target seconds per frame
    every_n = max(1, args.every)

    debounce = DebounceWindow(args.confirm)
    last_dets: List[Detection] = []
    debounce_count = 0  # local mirror for overlay

    frame_idx = 0
    paused = False
    win_name = "PrintGuard Demo"
    snapshot_dir = "snapshots"
    # Initialised before use; set here so LSP / type-checkers are happy
    raw_frame: np.ndarray = np.zeros((480, 640, 3), dtype=np.uint8)
    t_frame_start: float = time.monotonic()

    # fps measurement
    fps_t0 = time.monotonic()
    fps_count = 0
    display_fps = video_fps

    if not args.no_display:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    print()
    print("─" * 74)
    print(
        f"  PrintGuard Demo  |  {Path(args.video).name}"
        f"  |  every {every_n} frames  |  conf ≥ {args.conf}"
    )
    print(
        f"  {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} source"
        f"  →  max {args.max_width}×{args.max_height} display"
        f"  |  {video_fps:.1f} fps  |  {total_frames} frames"
    )
    print()
    print("  Overlay colours:")
    print("    RED box    = spaghetti / detachment  (Obico ONNX model)")
    print("    YELLOW box = layer shift / adhesion / extrusion  (CV heuristics)")
    print("    GREEN top  = no failures on this frame")
    print("    RED top    = failure detected this frame")
    print()
    if not args.no_display:
        print("  Controls:  SPACE = pause   → = step   s = snapshot   q/Esc = quit")
    print("─" * 74)
    print()

    while True:
        if not paused:
            t_frame_start = time.monotonic()
            ok, raw_frame = cap.read()
            if not ok:
                print("\nEnd of video.")
                break
            frame_idx += 1

            # ── Run detector every N-th frame ─────────────────────────────
            if frame_idx % every_n == 0:
                last_dets = detector.detect(raw_frame)
                fired = debounce.update(last_dets)
                # Keep a local count for the overlay
                if last_dets:
                    debounce_count = min(debounce_count + 1, args.confirm)
                else:
                    debounce_count = 0

                if fired:
                    labels = _collect_labels(last_dets)
                    ts = datetime.datetime.now().strftime("%H:%M:%S")
                    print(
                        f"\n[{ts}] FAILURE CONFIRMED at frame {frame_idx}"
                        f"  ({', '.join(labels)})"
                    )
                    print()
                    print(format_suggestions(labels))
                    snap = save_snapshot(
                        raw_frame,
                        snapshot_dir,
                        label="demo_" + "_".join(labels),
                    )
                    print(f"\n  Snapshot saved → {snap}\n")
                    debounce_count = 0  # reset visual counter after alert

            # ── fps measurement ───────────────────────────────────────────
            fps_count += 1
            elapsed_fps = time.monotonic() - fps_t0
            if elapsed_fps >= 1.0:
                display_fps = fps_count / elapsed_fps
                fps_count = 0
                fps_t0 = time.monotonic()

        # ── Build display frame ───────────────────────────────────────────
        if not args.no_display:
            scaled, sf = scale_frame(raw_frame, args.max_width, args.max_height)
            annotated = draw_overlay(
                scaled,
                last_dets,
                debounce_count,
                args.confirm,
                frame_idx,
                total_frames,
                display_fps,
                paused,
                sf,
            )
            cv2.imshow(win_name, annotated)

        # ── Keypress handling ─────────────────────────────────────────────
        if not args.no_display:
            wait_ms = (
                1
                if paused
                else max(
                    1, int((frame_dur - (time.monotonic() - t_frame_start)) * 1000)
                )
            )
            key = cv2.waitKey(wait_ms) & 0xFF
            if key in (ord("q"), 27):  # q or Esc
                print("Quit by user.")
                break
            elif key == ord(" "):  # Space — pause/unpause
                paused = not paused
                if paused:
                    print(f"[paused at frame {frame_idx}]")
            elif key == ord("s"):  # s — manual snapshot
                snap = save_snapshot(raw_frame, snapshot_dir, label="demo_manual")
                print(f"  Manual snapshot → {snap}")
            elif key == 83:
                # Right arrow key on Linux (cv2.waitKey returns 83)
                if paused:
                    ok2, raw_frame = cap.read()
                    if ok2:
                        frame_idx += 1
                        # Run detector on stepped frame too
                        last_dets = detector.detect(raw_frame)
                        debounce_count = len(last_dets)
                    paused = True  # stay paused after step
        else:
            # Headless: pace to real video speed
            elapsed = time.monotonic() - t_frame_start
            sleep_for = frame_dur - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    # Validate video file
    if not Path(args.video).exists():
        print(f"Error: video file not found: {args.video}", file=sys.stderr)
        print()
        print("No sample video? Download one with yt-dlp:")
        print("  pip install yt-dlp")
        print(
            '  yt-dlp -o sample_fail.mp4 "https://www.youtube.com/watch?v=D6_VCBMEiMo"'
        )
        sys.exit(1)

    # Validate weights
    if not Path(args.weights).exists():
        print(f"Error: model weights not found: {args.weights}", file=sys.stderr)
        print("Run setup.sh to download them, or pass --weights <path>")
        sys.exit(1)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: could not open video: {args.video}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model from {args.weights} …", end=" ", flush=True)
    detector = PrintFailureDetector(
        {
            "weights": args.weights,
            "confidence_threshold": args.conf,
            "nms_threshold": 0.4,
            "force_cpu": False,
        }
    )
    detector.load()
    print("done.")

    try:
        process_video(cap, detector, args)
    except KeyboardInterrupt:
        print("\nStopped.")
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
