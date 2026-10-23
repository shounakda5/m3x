"""
Regenerate PilotPhaseC/results/RESULTS_SUMMARY.md from all manifest.json files.

Run from the m3x repo root:
    .venv/bin/python -m PilotPhaseC.src.summarize_results
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PilotPhaseC.config import RESULTS_DIR


def _find_manifests(root: Path) -> list[Path]:
    return sorted(root.glob("**/manifest.json"))


def _one_line_metrics(manifest: dict) -> str:
    task = manifest.get("task")
    cv = manifest.get("cv_metrics", {})
    pooled = cv.get("pooled_oof", {})
    if task == "classification":
        return (
            f"acc={pooled.get('accuracy', 0):.3f}  "
            f"f1_macro={pooled.get('f1_macro', 0):.3f}  "
            f"n_windows={pooled.get('n_windows', manifest.get('n_windows', '?'))}"
        )
    if task == "regression":
        return (
            f"MAE={pooled.get('mae', 0):.3f}  "
            f"RMSE={pooled.get('rmse', 0):.3f}  "
            f"R2={pooled.get('r2', 0):.4f}  "
            f"n_windows={manifest.get('n_windows', '?')}"
        )
    return ""


def main():
    manifests = _find_manifests(RESULTS_DIR)
    lines = [
        "# Pilot Phase C — results summary",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Window-level pooled out-of-fold metrics from each run's `manifest.json`.",
        "For per-class tables and confusion matrices, open that run's `cv_report.txt`.",
        "",
        "| Participant | Lineage | Task | Pooled OOF metrics | Results folder |",
        "|---|---|---|---|---|",
    ]
    for path in manifests:
        rel_dir = path.parent.relative_to(RESULTS_DIR)
        parts = rel_dir.parts
        if len(parts) < 3:
            continue
        participant, lineage, task_dir = parts[0], parts[1], parts[2]
        task = "classification" if task_dir == "classification" else "regression"
        m = json.loads(path.read_text(encoding="utf-8"))
        metrics = _one_line_metrics(m)
        lines.append(f"| {participant} | {lineage} | {task} | {metrics} | `{rel_dir}` |")

    out = RESULTS_DIR / "RESULTS_SUMMARY.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
