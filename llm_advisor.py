"""
llm_advisor.py
--------------
LLM-powered failure advisor for PrintGuard.

Replaces the static advisor.py knowledge base with a Gemini API call that
receives live printer state from Bambu MQTT, giving context-specific
suggestions rather than generic ones.

Public API
----------
PrinterStateStore   – thread-safe accumulator; pass .update to BambuMQTTClient
                      as the on_status callback.
get_llm_advice()    – build prompt, call Gemini, return formatted advice string.
                      Falls back to static advisor.py on any error.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Printer state store ────────────────────────────────────────────────────────


class PrinterStateStore:
    """
    Continuously updated snapshot of the printer's live state, populated by
    the MQTT on_status callback.

    Thread-safe: update() can be called from the MQTT network thread while
    snapshot() is called from the main loop thread.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._state: Dict[str, Any] = {}

    # ── MQTT callback ──────────────────────────────────────────────────────

    def update(self, payload: dict) -> None:
        """
        Called by BambuMQTTClient whenever a report arrives.
        Extracts the fields we care about from the 'print' sub-dict.
        """
        print_data: dict = payload.get("print", {})
        if not print_data:
            return

        extracted: Dict[str, Any] = {}

        # Temperatures
        for key in (
            "nozzle_temper",
            "nozzle_target_temper",
            "bed_temper",
            "bed_target_temper",
            "chamber_temper",
        ):
            if key in print_data:
                extracted[key] = print_data[key]

        # Fan speeds (0–255 raw values)
        for key in (
            "cooling_fan_speed",
            "big_fan1_speed",
            "big_fan2_speed",
            "heatbreak_fan_speed",
        ):
            if key in print_data:
                extracted[key] = print_data[key]

        # Print progress
        for key in (
            "gcode_state",
            "layer_num",
            "total_layer_num",
            "mc_percent",
            "mc_remaining_time",
            "gcode_file",
            "subtask_name",
        ):
            if key in print_data:
                extracted[key] = print_data[key]

        # Speed
        for key in ("spd_lvl", "spd_mag"):
            if key in print_data:
                extracted[key] = print_data[key]

        # Nozzle diameter
        if "nozzle_diameter" in print_data:
            extracted["nozzle_diameter"] = print_data["nozzle_diameter"]

        # Errors / HMS alerts
        if "hms" in print_data:
            extracted["hms"] = print_data["hms"]
        if "fail_reason" in print_data:
            extracted["fail_reason"] = print_data["fail_reason"]
        if "print_error" in print_data:
            extracted["print_error"] = print_data["print_error"]

        # Active filament from AMS
        ams_data = print_data.get("ams", {})
        if ams_data:
            active_tray_idx = str(ams_data.get("tray_now", ""))
            filament_info = _extract_active_filament(ams_data, active_tray_idx)
            if filament_info:
                extracted["filament"] = filament_info

        with self._lock:
            self._state.update(extracted)

    # ── Snapshot ───────────────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        """Return a thread-safe copy of the current printer state."""
        with self._lock:
            return dict(self._state)

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._state) == 0


def _extract_active_filament(ams_data: dict, active_tray_idx: str) -> Optional[dict]:
    """Pull the active filament tray info out of the AMS payload."""
    try:
        for ams_unit in ams_data.get("ams", []):
            for tray in ams_unit.get("tray", []):
                if str(tray.get("id", "")) == active_tray_idx:
                    return {
                        "type": tray.get("tray_type", "unknown"),
                        "color": tray.get("tray_color", ""),
                        "nozzle_temp_min": tray.get("nozzle_temp_min"),
                        "nozzle_temp_max": tray.get("nozzle_temp_max"),
                        "bed_temp": tray.get("bed_temp"),
                    }
    except Exception:
        pass
    return None


# ── Prompt builder ─────────────────────────────────────────────────────────────

_SPEED_LEVEL_NAMES = {1: "Silent", 2: "Standard", 3: "Sport", 4: "Ludicrous"}


def _fan_pct(raw) -> str:
    """Convert a 0–255 fan value to a percentage string."""
    try:
        return f"{round(int(raw) / 255 * 100)}%"
    except Exception:
        return str(raw)


def _build_prompt(
    labels: List[str],
    detections_detail: Optional[List[dict]],
    state: Dict[str, Any],
    gcode_snippet: Optional[str],
) -> str:
    """
    Build a rich prompt that gives Gemini everything it needs to produce
    printer-specific fix suggestions rather than generic advice.
    """
    lines = [
        "You are an expert 3D printing technician assistant for Bambu Lab FDM printers.",
        "A print failure has just been detected and the printer has been paused.",
        "Provide concise, specific, actionable fix suggestions based on the exact printer state below.",
        "Do NOT give generic advice. Reference the actual temperatures, fan speeds, and filament type.",
        "",
        "## Detected failures",
    ]

    if detections_detail:
        for d in detections_detail:
            conf = d.get("confidence", "")
            conf_str = f" (confidence: {conf:.0%})" if conf else ""
            lines.append(f"  - {d['label']}{conf_str}")
    else:
        for label in labels:
            lines.append(f"  - {label}")

    lines.append("")
    lines.append("## Live printer state at time of failure")

    # Temperatures
    nozzle = state.get("nozzle_temper")
    nozzle_target = state.get("nozzle_target_temper")
    bed = state.get("bed_temper")
    bed_target = state.get("bed_target_temper")
    chamber = state.get("chamber_temper")

    if nozzle is not None:
        lines.append(f"  Nozzle temperature:  {nozzle} °C (target: {nozzle_target} °C)")
    if bed is not None:
        lines.append(f"  Bed temperature:     {bed} °C (target: {bed_target} °C)")
    if chamber is not None:
        lines.append(f"  Chamber temperature: {chamber} °C")

    # Fans
    cooling = state.get("cooling_fan_speed")
    aux = state.get("big_fan1_speed")
    chamber_fan = state.get("big_fan2_speed")
    if cooling is not None:
        lines.append(f"  Part cooling fan:    {_fan_pct(cooling)}")
    if aux is not None:
        lines.append(f"  Aux fan:             {_fan_pct(aux)}")
    if chamber_fan is not None:
        lines.append(f"  Chamber fan:         {_fan_pct(chamber_fan)}")

    # Filament
    filament = state.get("filament")
    if filament:
        ftype = filament.get("type", "unknown")
        fcolor = filament.get("color", "")
        fmin = filament.get("nozzle_temp_min")
        fmax = filament.get("nozzle_temp_max")
        fbed = filament.get("bed_temp")
        temp_range = f", recommended nozzle {fmin}–{fmax} °C" if fmin and fmax else ""
        bed_rec = f", recommended bed {fbed} °C" if fbed else ""
        color_str = f" ({fcolor})" if fcolor else ""
        lines.append(f"  Filament:            {ftype}{color_str}{temp_range}{bed_rec}")

    # Nozzle diameter
    nozzle_dia = state.get("nozzle_diameter")
    if nozzle_dia:
        lines.append(f"  Nozzle diameter:     {nozzle_dia} mm")

    # Print progress
    layer = state.get("layer_num")
    total_layers = state.get("total_layer_num")
    pct = state.get("mc_percent")
    gcode_file = state.get("gcode_file") or state.get("subtask_name")
    if layer is not None:
        layer_str = f"{layer}" + (f" / {total_layers}" if total_layers else "")
        lines.append(f"  Layer:               {layer_str}")
    if pct is not None:
        lines.append(f"  Progress:            {pct}%")
    if gcode_file:
        lines.append(f"  GCode file:          {gcode_file}")

    # Speed
    spd_lvl = state.get("spd_lvl")
    spd_mag = state.get("spd_mag")
    if spd_lvl is not None or spd_mag is not None:
        spd_name = (
            _SPEED_LEVEL_NAMES.get(spd_lvl, f"level {spd_lvl}") if spd_lvl else ""
        )
        spd_pct = f" ({spd_mag}%)" if spd_mag else ""
        lines.append(f"  Print speed:         {spd_name}{spd_pct}")

    # HMS / errors
    hms = state.get("hms")
    if hms:
        lines.append(f"  HMS alerts:          {hms}")
    fail_reason = state.get("fail_reason")
    if fail_reason:
        lines.append(f"  Fail reason code:    {fail_reason}")

    if not state:
        lines.append(
            "  (No live printer state available — MQTT not connected or no data received yet)"
        )

    # GCode snippet
    if gcode_snippet:
        lines.append("")
        lines.append("## Last GCode lines before failure")
        lines.append("```")
        lines.append(gcode_snippet.strip())
        lines.append("```")

    lines += [
        "",
        "## Your task",
        "1. Diagnose the most likely root cause of this specific failure given the printer state above.",
        "2. List 3–5 concrete fix steps the user should take RIGHT NOW (before the next print attempt).",
        "3. List 2–3 slicer/settings changes to prevent recurrence.",
        "4. Keep each point to one sentence. Be direct. Reference the actual numbers "
        "(e.g. 'your nozzle is at X °C which is Y °C below the minimum for this filament').",
    ]
    return "\n".join(lines)


# ── GCode loader ───────────────────────────────────────────────────────────────


def _load_gcode_snippet(
    gcode_file: str, gcode_dir: str, max_lines: int = 20
) -> Optional[str]:
    """
    Try to load the last `max_lines` lines from the gcode file.
    Returns None if the file cannot be found or read.
    """
    if not gcode_file or not gcode_dir:
        return None
    try:
        base = Path(gcode_file).name
        path = Path(gcode_dir) / base
        if not path.exists():
            logger.debug("GCode file not found at %s", path)
            return None
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        snippet = "".join(all_lines[-max_lines:])
        logger.debug(
            "Loaded gcode snippet from %s (%d lines)",
            path,
            min(max_lines, len(all_lines)),
        )
        return snippet
    except Exception as exc:
        logger.debug("Could not load gcode file: %s", exc)
        return None


# ── Gemini call ────────────────────────────────────────────────────────────────


def _call_gemini(prompt: str, api_key: str, model: str, timeout: int) -> str:
    """Call the Gemini API and return the response text."""
    # Imported lazily so the module loads fine even without the package installed
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(model)
    response = gemini_model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=1024,
            temperature=0.3,  # low = focused, less hallucination
        ),
        request_options={"timeout": timeout},
    )
    return response.text.strip()


# ── Public API ─────────────────────────────────────────────────────────────────


def get_llm_advice(
    labels: List[str],
    printer_state: Dict[str, Any],
    advisor_cfg: dict,
    detections_detail: Optional[List[dict]] = None,
) -> str:
    """
    Generate context-specific fix suggestions using the Gemini API.

    Parameters
    ----------
    labels : list of str
        Failure labels detected (e.g. ["spaghetti", "layer_shift"]).
    printer_state : dict
        Snapshot from PrinterStateStore.snapshot().
    advisor_cfg : dict
        The 'advisor' section from config.yaml.
    detections_detail : list of dict, optional
        Richer detection info with confidence scores; keys: 'label', 'confidence'.

    Returns
    -------
    str
        Multi-line advice string, either from Gemini or the static fallback.
    """
    backend = advisor_cfg.get("backend", "gemini")
    fallback = advisor_cfg.get("fallback_to_static", True)

    if backend == "static":
        logger.info("Advisor backend is 'static', using knowledge base directly")
        return _static_fallback(labels)

    api_key: str = advisor_cfg.get("gemini_api_key", "")
    model: str = advisor_cfg.get("gemini_model", "gemini-2.0-flash")
    timeout: int = int(advisor_cfg.get("timeout_seconds", 30))
    gcode_dir: str = advisor_cfg.get("gcode_dir", "")

    if not api_key:
        logger.warning("No Gemini API key configured — falling back to static advisor")
        return _static_fallback(labels)

    # Try to load a gcode snippet if we know the filename and the dir is configured
    gcode_file = printer_state.get("gcode_file") or printer_state.get(
        "subtask_name", ""
    )
    gcode_snippet = _load_gcode_snippet(str(gcode_file), gcode_dir)

    prompt = _build_prompt(labels, detections_detail, printer_state, gcode_snippet)
    logger.debug("Gemini prompt:\n%s", prompt)

    try:
        logger.info("Calling Gemini API (model=%s, timeout=%ds)…", model, timeout)
        response_text = _call_gemini(prompt, api_key, model, timeout)
        header = f"=== PrintGuard AI Failure Report ({'  |  '.join(labels)}) ===\n"
        return header + response_text
    except Exception as exc:
        logger.error("Gemini API call failed: %s", exc)
        if fallback:
            logger.warning("Falling back to static advisor")
            return _static_fallback(labels)
        return f"[LLM advisor error: {exc}]\n" + _static_fallback(labels)


def _static_fallback(labels: List[str]) -> str:
    """Delegate to the original static knowledge base in advisor.py."""
    try:
        from advisor import format_suggestions

        return format_suggestions(labels)
    except Exception as exc:
        logger.error("Static fallback also failed: %s", exc)
        return (
            f"Failures detected: {', '.join(labels)}. "
            "Please inspect the print manually."
        )
