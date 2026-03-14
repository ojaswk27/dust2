"""
advisor.py
----------
Maps a detected failure label to a structured list of actionable suggestions.

Each suggestion has:
  - title   : short problem statement
  - causes  : list of likely causes
  - fixes   : ordered list of things to try (most impactful first)
  - severity: "low" | "medium" | "high"  (used by main.py to decide action)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Suggestion:
    title: str
    severity: str  # "low" | "medium" | "high"
    causes: List[str] = field(default_factory=list)
    fixes: List[str] = field(default_factory=list)

    def format(self) -> str:
        lines = [
            f"  [{self.severity.upper()}] {self.title}",
            "  Likely causes:",
        ]
        for c in self.causes:
            lines.append(f"    • {c}")
        lines.append("  Suggested fixes (try in order):")
        for i, f in enumerate(self.fixes, 1):
            lines.append(f"    {i}. {f}")
        return "\n".join(lines)


# ── Knowledge base ─────────────────────────────────────────────────────────────

_ADVICE: Dict[str, Suggestion] = {
    "spaghetti": Suggestion(
        title="Spaghetti / detached print detected",
        severity="high",
        causes=[
            "Print detached from the build plate mid-print",
            "First layer adhesion was insufficient",
            "Warping caused the print to peel from the bed",
            "Vibration or external disturbance",
        ],
        fixes=[
            "Clean the build plate with IPA before the next print",
            "Increase first-layer height or slow down first-layer speed",
            "Raise bed temperature by 5 °C (typical: PLA 60 °C, PETG/ABS 80–100 °C)",
            "Apply a thin layer of glue stick or hairspray to the plate",
            "Add a brim or raft in your slicer for better bed coverage",
            "Check for and fix warped build plate (use auto bed levelling)",
            "Reduce part cooling fan speed for the first 3–5 layers",
        ],
    ),
    "layer_shift": Suggestion(
        title="Layer shift detected",
        severity="high",
        causes=[
            "Print head collided with the print (knock-over or stringing)",
            "Loose or skipping stepper motor belt",
            "Extruder or toolhead stall / overheating motor driver",
            "Print speed too high causing inertia-driven shift",
            "Mechanical obstruction in X/Y axis travel",
        ],
        fixes=[
            "Inspect and tension the X and Y drive belts",
            "Reduce print speed or acceleration in slicer settings",
            "Check that no cables, clips, or debris obstruct head movement",
            "Ensure the printer is on a stable, vibration-free surface",
            "Check stepper motor driver temperature — add cooling if hot",
            "Reduce layer height / enable smooth/silent mode if available",
            "Update printer firmware (Bambu releases motion-control improvements)",
        ],
    ),
    "bed_adhesion_loss": Suggestion(
        title="Bed adhesion loss / print detaching",
        severity="high",
        causes=[
            "Bed surface contaminated (oils from fingers, dust)",
            "First layer Z offset too high (gap between nozzle and bed)",
            "Bed temperature too low for the filament type",
            "Insufficient first-layer width or slow first-layer extrusion",
            "Ambient draught or enclosure temperature drop",
        ],
        fixes=[
            "Wipe build plate with 90%+ IPA and a lint-free cloth",
            "Re-run bed levelling / first-layer calibration",
            "Increase bed temperature by 5–10 °C for the first layer",
            "Lower the Z offset by 0.05 mm increments until first layer squishes properly",
            "Add a brim (minimum 5 mm) in the slicer",
            "Close the printer enclosure or move away from draughts",
            "Increase first-layer line width to 120–140% in the slicer",
        ],
    ),
    "under_extrusion": Suggestion(
        title="Under-extrusion detected",
        severity="medium",
        causes=[
            "Partial clog or nozzle debris",
            "Filament grinding / slipping in the extruder",
            "Print speed or flow rate too high for the hotend capacity",
            "Wet/moisture-absorbed filament causing steam voids",
            "Incorrect filament diameter setting in slicer",
        ],
        fixes=[
            "Perform a cold pull to clear a partial clog",
            "Increase hotend temperature by 5–10 °C",
            "Reduce print speed by 20–30%",
            "Check extruder tension and clean gear teeth",
            "Dry your filament (60 °C for 4–6 h in a food dehydrator or oven)",
            "Increase flow / extrusion multiplier by 5% steps in slicer",
            "Verify filament diameter matches slicer setting (1.75 vs 2.85 mm)",
        ],
    ),
    "over_extrusion": Suggestion(
        title="Over-extrusion / excessive material detected",
        severity="medium",
        causes=[
            "Flow rate / extrusion multiplier set too high",
            "Incorrect filament diameter (e.g. 1.80 mm measured as 1.75)",
            "Temperature too high causing runny melt zone",
            "E-steps (extruder steps/mm) not calibrated",
        ],
        fixes=[
            "Reduce flow / extrusion multiplier by 5% steps in slicer",
            "Measure actual filament diameter with calipers and update slicer",
            "Lower print temperature by 5 °C",
            "Calibrate extruder E-steps / flow rate using a tower or line test",
            "Enable linear advance / pressure advance (Bambu has this built-in)",
        ],
    ),
    "blob_stringing": Suggestion(
        title="Stringing / blobs detected",
        severity="low",
        causes=[
            "Retraction distance or speed not optimised for this filament",
            "Print temperature too high (low viscosity during travel)",
            "Travel moves too slow across open gaps",
            "Z-hop enabled when not needed (adds blob on restart)",
        ],
        fixes=[
            "Increase retraction distance by 0.5 mm steps (start at 0.8–1.5 mm for direct drive)",
            "Lower print temperature by 5–10 °C",
            "Enable or increase 'Wipe on retract' in slicer",
            "Increase travel speed to 200–250 mm/s",
            "Enable 'Avoid crossing perimeters' / 'Combing' in slicer",
            "Disable Z-hop if enabled (it can add ooze on lowering)",
            "Dry filament — wet filament strings aggressively",
        ],
    ),
}

# Fallback for unknown labels
_FALLBACK = Suggestion(
    title="Unknown print anomaly",
    severity="medium",
    causes=["Unclassified visual anomaly detected by the camera"],
    fixes=[
        "Inspect the print visually",
        "Review the camera snapshot saved in the snapshots/ directory",
        "Compare against a known-good print of the same model",
        "Check printer logs / Bambu Studio for error codes",
    ],
)


# ── Public API ─────────────────────────────────────────────────────────────────


def get_suggestions(labels: List[str]) -> List[Suggestion]:
    """
    Return a deduplicated list of Suggestion objects for a set of failure labels.

    Parameters
    ----------
    labels : list of str
        One or more failure labels (e.g. ["spaghetti", "blob_stringing"]).

    Returns
    -------
    list of Suggestion, ordered by descending severity.
    """
    _ORDER = {"high": 0, "medium": 1, "low": 2}
    seen: set[str] = set()
    suggestions: List[Suggestion] = []
    for label in labels:
        if label in seen:
            continue
        seen.add(label)
        suggestions.append(_ADVICE.get(label, _FALLBACK))
    suggestions.sort(key=lambda s: _ORDER.get(s.severity, 3))
    return suggestions


def format_suggestions(labels: List[str]) -> str:
    """
    Human-readable multi-line string of all suggestions for the given labels.
    """
    suggs = get_suggestions(labels)
    if not suggs:
        return "No specific suggestions available."
    parts = [f"=== PrintGuard Failure Report ({'  |  '.join(labels)}) ==="]
    for s in suggs:
        parts.append(s.format())
    return "\n".join(parts)
