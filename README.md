# PrintGuard

Raspberry Pi 5 print failure monitor for Bambu Lab FDM printers.

Captures frames from a Pi camera module (or USB webcam), runs the
[Obico (The Spaghetti Detective)](https://github.com/TheSpaghettiDetective/obico-server)
YOLOv4-tiny ONNX model + additional OpenCV heuristics to detect print failures,
then automatically pauses or stops the printer via local MQTT and prints
actionable fix suggestions.

---

## Detected failure modes

| Failure | Source |
|---|---|
| Spaghetti / print detachment | Obico ONNX model |
| Stringing / blobs | Obico + edge-density heuristic |
| Layer shift | OpenCV Sobel heuristic |
| Bed adhesion loss | OpenCV variance heuristic |
| Under-extrusion | OpenCV brightness heuristic |
| Over-extrusion | OpenCV brightness heuristic |

You can train your own weights on top of the Obico base (see **Custom weights** below).

---

## Hardware requirements

- Raspberry Pi 5 (4 GB or 8 GB recommended)
- Raspberry Pi Camera Module 3 (or any USB webcam)
- Pi and Bambu printer on the same local network
- Bambu printer with **LAN mode** enabled

---

## Quick start

```bash
# 1. Clone or copy this repo onto the Pi
git clone <your-repo-url> printguard
cd printguard

# 2. Run the setup script (installs deps, downloads Obico weights, creates systemd service)
chmod +x setup.sh
./setup.sh

# 3. Fill in your printer details
nano config.local.yaml
#  - bambu.ip          : printer LAN IP  (find it in Bambu Studio or router)
#  - bambu.device_id   : printer serial  (Settings → Device in Bambu Studio)
#  - bambu.access_code : 8-char code     (printer touchscreen → Settings → LAN mode)

# 4. Test detection without touching the printer
.venv/bin/python main.py --dry-run

# 5. Run for real
.venv/bin/python main.py

# Or as a background service
systemctl --user start printguard
```

---

## Configuration reference

All settings live in `config.yaml` (defaults) and `config.local.yaml` (your overrides).
`config.local.yaml` wins for any key it defines and is `.gitignore`d.

Key settings:

| Key | Default | Description |
|---|---|---|
| `bambu.on_failure` | `pause` | `pause` or `stop` when failure detected |
| `camera.backend` | `picamera2` | `picamera2` or `usb` |
| `camera.fps` | `2` | Frames per second to analyse |
| `model.confidence_threshold` | `0.3` | Minimum confidence to record a detection |
| `model.confirm_frames` | `3` | Consecutive positive frames before acting |
| `notifications.webhook_enabled` | `false` | POST JSON payload to a URL on failure |

---

## CLI options

```
python main.py --help

  --config CONFIG   Path to config YAML (default: config.yaml)
  --dry-run         Detect failures but do NOT send printer commands
  --once IMAGE      Run on a single image file and print suggestions, then exit
```

**Test a saved image:**
```bash
.venv/bin/python main.py --once snapshots/20260314_120000_spaghetti.jpg
```

---

## Custom / fine-tuned weights

The Obico model only classifies spaghetti/detachment. To train your own weights:

1. Collect labelled frames from your printer (the `snapshots/` folder is a good source).
2. Use [Roboflow](https://roboflow.com) or [Label Studio](https://labelstud.io) to annotate.
3. Train YOLOv4-tiny or YOLOv8n using [Ultralytics](https://github.com/ultralytics/ultralytics).
4. Export the model to ONNX:
   ```bash
   yolo export model=best.pt format=onnx imgsz=416
   ```
5. Update `model.weights` in `config.local.yaml` to point to your new `.onnx`.

Public datasets worth fine-tuning on:
- [3D Printing Failures (Roboflow Universe)](https://universe.roboflow.com/search?q=3d+printing+failure)
- [Obico training data](https://github.com/TheSpaghettiDetective/obico-server/tree/release/ml_api)

---

## Project structure

```
printguard/
├── main.py          # Entry point and main monitoring loop
├── camera.py        # Frame capture (picamera2 / USB)
├── detector.py      # Obico ONNX model + OpenCV heuristics
├── bambu_mqtt.py    # Bambu Lab local MQTT client
├── advisor.py       # Failure → fix suggestions knowledge base
├── config.yaml      # Default configuration (commit this)
├── config.local.yaml# Your secrets & overrides (gitignored)
├── requirements.txt
├── setup.sh         # Pi 5 install + systemd service
├── model/           # ONNX weights downloaded by setup.sh
├── snapshots/       # Auto-saved failure frames
└── logs/            # Rotating log files
```

---

## Bambu LAN mode MQTT notes

- Local MQTT server: `<printer-ip>:8883` (TLS, self-signed cert – verification skipped)
- Username: `bblp` / Password: LAN access code
- Stop command: `{"print": {"command": "stop", ...}}` at QoS 1
- Pause command: `{"print": {"command": "pause", ...}}` at QoS 1
- Protocol reference: <https://github.com/Doridian/OpenBambuAPI/blob/main/mqtt.md>

---

## Licence

MIT
