"""
train.py -- Train a model on a frozen Pilot Phase C dataset snapshot and
write a versioned, traceable results directory.

The lineage plumbing here (which dataset snapshot, which git commit, which
hyperparameters, what metrics came out) is fully wired up now, so every
model version this phase produces is reproducible from day one. The actual
feature extraction / model choice is left as a TODO -- it depends on the
real data's shape (raw time-series vs. tabular features), which isn't known
until the first batch arrives. Wire it up using the existing src/models/*
classes (CNNClassifier, RFClassifier, etc.) the same way the other two
pipelines do -- there's no need to reimplement model code here.

Usage
-----
    python -m PilotPhaseC.src.train --dataset-version v1 --output-version v1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PilotPhaseC.config import RESULTS_DIR, N_FOLDS, RANDOM_STATE
from PilotPhaseC.src.loader import load_dataset

# TODO: import the model(s) this phase actually needs once the data shape is
# known, e.g.:
#   from src.models.cnn_model import CNNClassifier
#   from src.evaluation.cross_val import run_cross_val


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Train a model on a Pilot Phase C dataset snapshot.")
    parser.add_argument("--dataset-version", required=True, help="Dataset snapshot to train on, e.g. v1.")
    parser.add_argument("--output-version", required=True, help="Name for this results version, e.g. v1.")
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    args = parser.parse_args()

    out = RESULTS_DIR / args.output_version
    if out.exists():
        raise FileExistsError(f"{out} already exists -- pick a new --output-version.")
    out.mkdir(parents=True)

    print(f"Loading dataset snapshot '{args.dataset_version}' ...")
    df = load_dataset(args.dataset_version)
    print(f"  {len(df)} rows loaded")

    # ------------------------------------------------------------------
    # TODO: feature/label extraction, model selection, cross-validation,
    # and model export all depend on the real data's columns and shape.
    # Fill in once the first batch has been ingested -- follow the pattern
    # in train_cnn.py (raw time-series) or train.py (tabular features) at
    # the repo root.
    # ------------------------------------------------------------------
    metrics = {}  # TODO: populate with real CV results

    manifest = {
        "output_version": args.output_version,
        "dataset_version": args.dataset_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "n_folds": args.n_folds,
        "random_state": RANDOM_STATE,
        "metrics": metrics,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifest written to {out / 'manifest.json'}")
    print("NOTE: training logic is still a TODO -- see the comment block above.")


if __name__ == "__main__":
    main()
