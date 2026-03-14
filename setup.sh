#!/usr/bin/env bash
# setup.sh – Install PrintGuard dependencies on Raspberry Pi 5
# Tested on: Debian Trixie (Raspberry Pi OS based on Debian 13)
# Run as a regular user (sudo is invoked internally where needed):
#   chmod +x setup.sh && ./setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
MODEL_DIR="$SCRIPT_DIR/model"
MODEL_URL="https://github.com/TheSpaghettiDetective/obico-server/raw/release/ml_api/model/model.onnx"
CFG_URL="https://github.com/TheSpaghettiDetective/obico-server/raw/release/ml_api/model/model.cfg"
META_URL="https://github.com/TheSpaghettiDetective/obico-server/raw/release/ml_api/model/model.meta"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }

# ── 1. System packages ───────────────────────────────────────────────────────
info "Updating apt and installing system dependencies…"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    python3-opencv \
    python3-picamera2 \
    libopenblas-dev \
    libjpeg-dev \
    libpng-dev \
    libgl1 \
    wget \
    curl

# ── 2. Python virtual environment ────────────────────────────────────────────
info "Creating Python virtual environment at $VENV_DIR…"
python3 -m venv --system-site-packages "$VENV_DIR"
# --system-site-packages lets the venv pick up python3-opencv & python3-picamera2
# which are easier to install via apt on Pi OS than via pip

info "Installing Python packages…"
"$VENV_DIR/bin/pip" install --upgrade pip wheel
"$VENV_DIR/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"

# ── 3. Download model weights ─────────────────────────────────────────────────
mkdir -p "$MODEL_DIR"

download_if_missing() {
    local url="$1"
    local dest="$2"
    if [ -f "$dest" ]; then
        info "Already present: $dest (skipping download)"
    else
        info "Downloading $(basename "$dest")…"
        wget -q --show-progress -O "$dest" "$url" || {
            warn "Download failed for $url"
            warn "You can manually place the file at $dest"
        }
    fi
}

download_if_missing "$MODEL_URL" "$MODEL_DIR/model.onnx"
download_if_missing "$CFG_URL"   "$MODEL_DIR/model.cfg"
download_if_missing "$META_URL"  "$MODEL_DIR/model.meta"

# ── 4. Config file ───────────────────────────────────────────────────────────
LOCAL_CFG="$SCRIPT_DIR/config.local.yaml"
if [ ! -f "$LOCAL_CFG" ]; then
    info "Creating config.local.yaml from template…"
    cp "$SCRIPT_DIR/config.yaml" "$LOCAL_CFG"
    warn "IMPORTANT: Edit config.local.yaml and fill in your Bambu printer IP,"
    warn "           device ID, and LAN access code before running PrintGuard."
else
    info "config.local.yaml already exists – not overwriting"
fi

# ── 5. Create required directories ───────────────────────────────────────────
mkdir -p "$SCRIPT_DIR/logs" "$SCRIPT_DIR/snapshots"

# ── 6. Optional: install systemd service ────────────────────────────────────
SYSTEMD_DIR="$HOME/.config/systemd/user"
SERVICE_FILE="$SYSTEMD_DIR/printguard.service"
if [ ! -f "$SERVICE_FILE" ]; then
    info "Installing systemd user service…"
    mkdir -p "$SYSTEMD_DIR"
    cat > "$SERVICE_FILE" << EOF
[Unit]
Description=PrintGuard 3D print failure monitor
After=network-online.target
Wants=network-online.target

[Service]
WorkingDirectory=${SCRIPT_DIR}
ExecStart=${VENV_DIR}/bin/python main.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload
    systemctl --user enable printguard.service
    info "Service installed. Start with:  systemctl --user start printguard"
    info "View logs with:               journalctl --user -u printguard -f"
else
    info "systemd service already exists – not overwriting"
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
info "Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Edit config.local.yaml with your printer details"
echo "  2. Position the Pi camera so the print bed is clearly visible"
echo "  3. Run a test: ${VENV_DIR}/bin/python main.py --dry-run"
echo "  4. Run for real: ${VENV_DIR}/bin/python main.py"
echo "     or start the systemd service: systemctl --user start printguard"
echo ""
