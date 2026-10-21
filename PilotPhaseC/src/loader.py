"""
loader.py -- Ingest pre-labeled Pilot Phase C data batches into a frozen,
versioned dataset snapshot.

Unlike the original pipeline (src/data/classify_survey.py + merge_labels.py)
and the Pilot B cohort (pilot_b/src/loader.py), this phase's incoming data
already carries its labels -- there is no survey-matching or burst-labeling
step. This module's only job is: validate each incoming batch's schema,
combine the batches you choose into one dataset, and freeze that as a
versioned snapshot with a manifest recording exactly what went into it.

Pipeline
--------
    PilotPhaseC/data/incoming/<batch_id>/        (raw drops, never edited in place)
        manifest.json                            (batch-level metadata, see below)
        *.csv                                     (pre-labeled rows)
            |
            v  validate_batch() + combine_batches()
            |
    PilotPhaseC/data/dataset/<version>/
        dataset.csv                               (frozen combined snapshot)
        manifest.json                             (which batches -> this snapshot,
                                                     row/participant counts, checksums)

Usage
-----
    python -m PilotPhaseC.src.loader --batches 2026-10-21_batch1 --version v1

TODO before the first real batch lands
---------------------------------------
    * REQUIRED_COLUMNS below is a placeholder -- set it to the actual column
      names once the first batch's format is known (which column holds the
      label, participant/session identifiers, timestamps, etc).
    * PilotPhaseC/config.py's CLASSES list needs to be filled in from the
      same source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PilotPhaseC.config import INCOMING_DIR, DATASET_DIR

# TODO: replace with the real pre-labeled schema once the first batch arrives.
# At minimum this should include whatever column holds the label.
REQUIRED_COLUMNS: list[str] = []


# ---------------------------------------------------------------------------
# Batch-level validation
# ---------------------------------------------------------------------------

def load_batch_manifest(batch_dir: Path) -> dict:
    manifest_path = batch_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{batch_dir} has no manifest.json -- every incoming batch needs one "
            f"(source, participants, date range, label scheme; see README.md)."
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def validate_batch(batch_dir: Path) -> pd.DataFrame:
    """
    Load and schema-check every CSV in one incoming batch folder.

    Raises if a required column is missing or if the batch has no manifest.
    Returns the batch's rows concatenated into a single DataFrame, tagged
    with its batch_id for traceability in the combined dataset.
    """
    manifest = load_batch_manifest(batch_dir)
    csv_paths = sorted(batch_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"{batch_dir} has a manifest but no CSV files.")

    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing required column(s): {missing}")
        df["_source_file"] = path.name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["_batch_id"] = batch_dir.name
    print(f"  {batch_dir.name}: {len(combined)} rows from {len(csv_paths)} file(s) "
          f"(source: {manifest.get('source', 'unknown')})")
    return combined


# ---------------------------------------------------------------------------
# Combine batches into a versioned snapshot
# ---------------------------------------------------------------------------

def combine_batches(batch_ids: list[str]) -> pd.DataFrame:
    frames = []
    for batch_id in batch_ids:
        batch_dir = INCOMING_DIR / batch_id
        if not batch_dir.is_dir():
            raise FileNotFoundError(f"No such incoming batch: {batch_dir}")
        frames.append(validate_batch(batch_dir))
    return pd.concat(frames, ignore_index=True)


def _file_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def write_dataset_snapshot(df: pd.DataFrame, batch_ids: list[str], version: str) -> Path:
    """Freeze a combined dataset as PilotPhaseC/data/dataset/<version>/, with a lineage manifest."""
    out_dir = DATASET_DIR / version
    if out_dir.exists():
        raise FileExistsError(
            f"{out_dir} already exists -- dataset snapshots are immutable once cut. "
            f"Use a new version name."
        )
    out_dir.mkdir(parents=True)

    dataset_path = out_dir / "dataset.csv"
    df.to_csv(dataset_path, index=False)

    manifest = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_batches": batch_ids,
        "n_rows": len(df),
        "dataset_checksum_sha256_16": _file_checksum(dataset_path),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nWrote dataset snapshot: {dataset_path} ({len(df)} rows)")
    print(f"Manifest: {out_dir / 'manifest.json'}")
    return dataset_path


def load_dataset(version: str) -> pd.DataFrame:
    """Load a previously frozen dataset snapshot by version name."""
    dataset_path = DATASET_DIR / version / "dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"No dataset snapshot at {dataset_path}. Cut one first with "
            f"`python -m PilotPhaseC.src.loader --batches <batch_id> [...] --version {version}`."
        )
    return pd.read_csv(dataset_path)


def main():
    parser = argparse.ArgumentParser(description="Combine incoming Pilot Phase C batches into a versioned dataset snapshot.")
    parser.add_argument("--batches", nargs="+", required=True, help="Batch IDs under data/incoming/ to combine.")
    parser.add_argument("--version", required=True, help="Name for the resulting dataset snapshot, e.g. v1.")
    args = parser.parse_args()

    print(f"Combining batches: {args.batches}")
    df = combine_batches(args.batches)
    write_dataset_snapshot(df, args.batches, args.version)


if __name__ == "__main__":
    main()
