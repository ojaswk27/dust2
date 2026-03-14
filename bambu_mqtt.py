"""
bambu_mqtt.py
-------------
Local MQTT client for Bambu Lab printers.

Protocol reference: https://github.com/Doridian/OpenBambuAPI/blob/main/mqtt.md

Connection details:
  - Host: <printer LAN IP>  Port: 8883  TLS: yes (self-signed cert – skip verify)
  - Username: bblp
  - Password: <LAN access code>  (8 chars shown on printer touchscreen)

Topics:
  Subscribe: device/<DEVICE_ID>/report    (printer → Pi)
  Publish:   device/<DEVICE_ID>/request   (Pi → printer)  QoS=1 for stop/pause
"""

from __future__ import annotations

import json
import logging
import ssl
import threading
import time
from typing import Callable, Optional

import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)

# Sequence counter – Bambu expects incrementing integers
_seq = 0
_seq_lock = threading.Lock()


def _next_seq() -> str:
    global _seq
    with _seq_lock:
        _seq += 1
        return str(_seq)


class BambuMQTTClient:
    """
    Manages the local MQTT connection to a Bambu Lab printer and exposes
    simple control commands (pause / stop / resume).

    Parameters
    ----------
    cfg : dict
        The 'bambu' section from config.yaml.
    on_status : callable, optional
        Called with the parsed JSON payload whenever the printer sends a
        push_status report.  Signature: on_status(payload: dict) -> None
    """

    PORT = 8883

    def __init__(self, cfg: dict, on_status: Optional[Callable[[dict], None]] = None):
        self._ip: str = cfg["ip"]
        self._device_id: str = cfg["device_id"]
        self._access_code: str = cfg["access_code"]
        self._on_failure_action: str = cfg.get("on_failure", "pause")
        self._on_status = on_status

        self._topic_report = f"device/{self._device_id}/report"
        self._topic_request = f"device/{self._device_id}/request"

        self._client: mqtt.Client = mqtt.Client(
            client_id=f"printguard_{int(time.time())}",
            protocol=mqtt.MQTTv311,
        )
        self._connected = threading.Event()
        self._running = False
        self._setup_client()

    # ── connection lifecycle ─────────────────────────────────────────────────

    def connect(self):
        """Connect to the printer and start the network loop in a background thread."""
        logger.info("Connecting to Bambu printer at %s:%d", self._ip, self.PORT)
        self._running = True
        self._client.loop_start()
        self._client.connect(self._ip, self.PORT, keepalive=60)
        if not self._connected.wait(timeout=15):
            self._running = False
            raise ConnectionError(
                f"Timed out connecting to Bambu printer at {self._ip}:{self.PORT}. "
                "Check that LAN mode is enabled on the printer and the IP / access code are correct."
            )
        logger.info("Connected to Bambu printer")

    def disconnect(self):
        self._running = False
        self._client.loop_stop()
        self._client.disconnect()
        logger.info("Disconnected from Bambu printer")

    def __enter__(self) -> "BambuMQTTClient":
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    # ── printer control commands ─────────────────────────────────────────────

    def pause(self) -> bool:
        """Pause the current print. Returns True if the publish succeeded."""
        return self._publish_command("pause")

    def stop(self) -> bool:
        """Stop (cancel) the current print. Returns True if the publish succeeded."""
        return self._publish_command("stop")

    def resume(self) -> bool:
        """Resume a paused print. Returns True if the publish succeeded."""
        return self._publish_command("resume")

    def on_failure_detected(self) -> bool:
        """
        Execute the configured action (pause or stop) when a failure is detected.
        Returns True on success.
        """
        if self._on_failure_action == "stop":
            logger.warning("Failure detected – sending STOP to printer")
            return self.stop()
        else:
            logger.warning("Failure detected – sending PAUSE to printer")
            return self.pause()

    def request_full_status(self):
        """Ask the printer for its full current status (useful on startup)."""
        payload = {
            "pushing": {
                "sequence_id": _next_seq(),
                "command": "pushall",
                "version": 1,
                "push_target": 1,
            }
        }
        self._publish(payload, qos=0)

    # ── internals ────────────────────────────────────────────────────────────

    def _setup_client(self):
        # Bambu printers use self-signed TLS – skip server cert verification
        tls_ctx = ssl.create_default_context()
        tls_ctx.check_hostname = False
        tls_ctx.verify_mode = ssl.CERT_NONE
        self._client.tls_set_context(tls_ctx)

        self._client.username_pw_set(username="bblp", password=self._access_code)

        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected.set()
            client.subscribe(self._topic_report, qos=0)
            logger.debug("Subscribed to %s", self._topic_report)
        else:
            logger.error("MQTT connect failed, rc=%d", rc)

    def _on_disconnect(self, client, userdata, rc):
        self._connected.clear()
        if rc != 0 and self._running:
            logger.warning(
                "Unexpected MQTT disconnect (rc=%d), will auto-reconnect", rc
            )

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
        except Exception:
            return

        if self._on_status is not None:
            try:
                self._on_status(payload)
            except Exception as exc:
                logger.debug("on_status callback raised: %s", exc)

    def _publish_command(self, command: str) -> bool:
        payload = {
            "print": {
                "sequence_id": _next_seq(),
                "command": command,
                "param": "",
            }
        }
        return self._publish(payload, qos=1)

    def _publish(self, payload: dict, qos: int = 0) -> bool:
        if not self._connected.is_set():
            logger.error("Cannot publish – not connected to printer")
            return False
        result = self._client.publish(
            self._topic_request,
            json.dumps(payload),
            qos=qos,
        )
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            logger.error("MQTT publish failed, rc=%d", result.rc)
            return False
        logger.debug("Published to %s: %s", self._topic_request, payload)
        return True
