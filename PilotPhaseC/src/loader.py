"""
loader.py -- Parse one lineage's pre-labeled Excel files into a windowed
dataset (X, y_label, y_stress, groups), with an on-disk cache so repeated
runs (classifier + regressor, tuning + CV) don't re-parse Excel every time.

Each row belongs to exactly one window (Global_Window) of WINDOW_SAMPLES
rows; Row_In_Window sorts rows within that window into the correct temporal
order before stacking into an (WINDOW_SAMPLES, n_channels) array. Every
window is validated to have exactly WINDOW_SAMPLES rows and a single,
unambiguous label and stress score -- malformed windows are dropped with a
loud warning rather than silently trained on.

Usage
-----
    from PilotPhaseC.src.loader import load_lineage_dataset
    data = load_lineage_dataset("ring01_pid104", "all_continuous")
    data.X, data.y_label, data.y_stress, data.groups
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PilotPhaseC.config import (
    PARTICIPANTS,
    DATASET_DIR,
    WINDOW_SAMPLES,
    ROW_IN_WINDOW_COL,
    WINDOW_ID_COL,
    GROUP_COL,
    CLASS_LABEL_COL,
    REGRESSION_LABEL_COL,
    RAW_CHANNELS,
)

REQUIRED_COLUMNS = [
    WINDOW_ID_COL, ROW_IN_WINDOW_COL, GROUP_COL,
    CLASS_LABEL_COL, REGRESSION_LABEL_COL,
] + RAW_CHANNELS


@dataclass
class LineageDataset:
    X: np.ndarray            # (N, WINDOW_SAMPLES, len(RAW_CHANNELS)) float32
    y_label: np.ndarray      # (N,) str   -- classification target
    y_stress: np.ndarray     # (N,) float32 -- regression target, 0-100
    groups: np.ndarray       # (N,) str   -- Survey_ID, for leakage-safe CV
    source_file: np.ndarray  # (N,) str   -- which Excel file this window came from
    n_windows_dropped: int = 0
    meta: dict = field(default_factory=dict)


def _lineage_dir(participant: str, lineage: str) -> Path:
    p = PARTICIPANTS[participant]
    return p["dir"] / p["lineages"][lineage]


def _load_one_file(path: Path) -> LineageDataset:
    df = pd.read_excel(path, header=0)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required column(s): {missing}")

    X_list, y_label_list, y_stress_list, group_list = [], [], [], []
    n_dropped = 0

    for window_id, win in df.groupby(WINDOW_ID_COL, sort=False):
        win = win.sort_values(ROW_IN_WINDOW_COL)

        if len(win) != WINDOW_SAMPLES:
            warnings.warn(
                f"{path.name}: window {window_id} has {len(win)} rows "
                f"(expected {WINDOW_SAMPLES}) -- dropped."
            )
            n_dropped += 1
            continue

        labels = win[CLASS_LABEL_COL].unique()
        stresses = win[REGRESSION_LABEL_COL].unique()
        groups_in_window = win[GROUP_COL].unique()
        if len(labels) != 1 or len(stresses) != 1 or len(groups_in_window) != 1:
            warnings.warn(
                f"{path.name}: window {window_id} has ambiguous "
                f"label/stress/group ({len(labels)}/{len(stresses)}/{len(groups_in_window)} "
                f"distinct values) -- dropped."
            )
            n_dropped += 1
            continue

        X_list.append(win[RAW_CHANNELS].to_numpy(dtype=np.float32))
        y_label_list.append(labels[0])
        y_stress_list.append(float(stresses[0]))
        group_list.append(groups_in_window[0])

    n = len(X_list)
    return LineageDataset(
        X=np.stack(X_list, axis=0) if n else np.empty((0, WINDOW_SAMPLES, len(RAW_CHANNELS)), dtype=np.float32),
        y_label=np.array(y_label_list, dtype=object),
        y_stress=np.array(y_stress_list, dtype=np.float32),
        groups=np.array(group_list, dtype=object),
        source_file=np.array([path.name] * n, dtype=object),
        n_windows_dropped=n_dropped,
    )


def _cache_path(participant: str, lineage: str) -> Path:
    return DATASET_DIR / participant / f"{lineage}.npz"


def load_lineage_dataset(
    participant: str, lineage: str, use_cache: bool = True, force_reload: bool = False,
) -> LineageDataset:
    cache_path = _cache_path(participant, lineage)

    if use_cache and not force_reload and cache_path.exists():
        npz = np.load(cache_path, allow_pickle=True)
        return LineageDataset(
            X=npz["X"], y_label=npz["y_label"], y_stress=npz["y_stress"],
            groups=npz["groups"], source_file=npz["source_file"],
            n_windows_dropped=int(npz["n_windows_dropped"]),
        )

    lineage_dir = _lineage_dir(participant, lineage)
    files = sorted(lineage_dir.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No .xlsx files found in {lineage_dir}")

    parts = [_load_one_file(f) for f in files]
    data = LineageDataset(
        X=np.concatenate([p.X for p in parts], axis=0),
        y_label=np.concatenate([p.y_label for p in parts]),
        y_stress=np.concatenate([p.y_stress for p in parts]),
        groups=np.concatenate([p.groups for p in parts]),
        source_file=np.concatenate([p.source_file for p in parts]),
        n_windows_dropped=sum(p.n_windows_dropped for p in parts),
        meta={"files": [f.name for f in files]},
    )

    # Sanity check: Survey_ID (the CV group key) must be globally unique
    # across files within a lineage -- otherwise two different survey
    # events would get silently merged into one CV group.
    seen = {}
    for src, grp in zip(data.source_file, data.groups):
        if grp in seen and seen[grp] != src:
            raise ValueError(
                f"Survey_ID {grp!r} appears in both {seen[grp]} and {src} -- "
                f"the group key is not globally unique within this lineage, "
                f"cross-validation grouping would be wrong."
            )
        seen[grp] = src

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            X=data.X, y_label=data.y_label, y_stress=data.y_stress,
            groups=data.groups, source_file=data.source_file,
            n_windows_dropped=data.n_windows_dropped,
        )

    return data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parse a Pilot Phase C lineage and report dataset stats.")
    parser.add_argument("--participant", default="ring01_pid104")
    parser.add_argument("--lineage", required=True, choices=["all_continuous", "labeled_last5min"])
    parser.add_argument("--force-reload", action="store_true")
    args = parser.parse_args()

    data = load_lineage_dataset(args.participant, args.lineage, force_reload=args.force_reload)
    print(f"X: {data.X.shape}  windows dropped: {data.n_windows_dropped}")
    print(f"groups: {len(np.unique(data.groups))} unique")
    print("Label counts:", pd.Series(data.y_label).value_counts().to_dict())
    print("Stress_0to100 range:", data.y_stress.min(), "-", data.y_stress.max())
