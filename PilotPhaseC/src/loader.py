"""
loader.py -- Parse one lineage's pre-labeled Excel files into a windowed
dataset (X, y_label, y_stress, groups), with an on-disk cache so repeated
runs (classifier + regressor, tuning + CV) don't re-parse Excel every time.
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
    MODEL_CHANNELS,
    PERIOD_COL,
)

_N_MODEL_CHANNELS = len(MODEL_CHANNELS)

REQUIRED_COLUMNS = [
    WINDOW_ID_COL, ROW_IN_WINDOW_COL, GROUP_COL,
    CLASS_LABEL_COL, REGRESSION_LABEL_COL,
] + RAW_CHANNELS


def _stack_window_features(win: pd.DataFrame) -> np.ndarray:
    """Excel columns plus LED3-LED1 and LED3-LED2 per sample."""
    base = win[RAW_CHANNELS].to_numpy(dtype=np.float32)
    led1, led2, led3 = base[:, 0], base[:, 1], base[:, 2]
    derived = np.stack([led3 - led1, led3 - led2], axis=1).astype(np.float32)
    return np.concatenate([base, derived], axis=1)


@dataclass
class LineageDataset:
    X: np.ndarray
    y_label: np.ndarray
    y_stress: np.ndarray
    groups: np.ndarray
    source_file: np.ndarray
    period: np.ndarray | None = None
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

    has_period = PERIOD_COL in df.columns
    X_list, y_label_list, y_stress_list, group_list, period_list = [], [], [], [], []
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
                f"label/stress/group -- dropped."
            )
            n_dropped += 1
            continue

        if has_period:
            periods = win[PERIOD_COL].unique()
            if len(periods) != 1:
                warnings.warn(f"{path.name}: window {window_id} ambiguous Period -- dropped.")
                n_dropped += 1
                continue
            period_val = str(periods[0])
        else:
            period_val = None

        X_list.append(_stack_window_features(win))
        y_label_list.append(labels[0])
        y_stress_list.append(float(stresses[0]))
        group_list.append(groups_in_window[0])
        if has_period:
            period_list.append(period_val)

    n = len(X_list)
    period_arr = np.array(period_list, dtype=object) if has_period and period_list else None
    return LineageDataset(
        X=np.stack(X_list, axis=0) if n else np.empty((0, WINDOW_SAMPLES, _N_MODEL_CHANNELS), dtype=np.float32),
        y_label=np.array(y_label_list, dtype=object),
        y_stress=np.array(y_stress_list, dtype=np.float32),
        groups=np.array(group_list, dtype=object),
        source_file=np.array([path.name] * n, dtype=object),
        period=period_arr,
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
        stale = npz["X"].shape[2] != _N_MODEL_CHANNELS or "period" not in npz.files
        if stale:
            warnings.warn(f"Cache {cache_path} is stale -- rebuilding.")
            force_reload = True
        else:
            period = npz["period"] if "period" in npz.files else None
            return LineageDataset(
                X=npz["X"], y_label=npz["y_label"], y_stress=npz["y_stress"],
                groups=npz["groups"], source_file=npz["source_file"],
                period=period,
                n_windows_dropped=int(npz["n_windows_dropped"]),
            )

    lineage_dir = _lineage_dir(participant, lineage)
    files = sorted(lineage_dir.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No .xlsx files found in {lineage_dir}")

    parts = [_load_one_file(f) for f in files]
    periods = [p.period for p in parts if p.period is not None]
    period_all = np.concatenate(periods) if periods and len(periods) == len(parts) else None
    data = LineageDataset(
        X=np.concatenate([p.X for p in parts], axis=0),
        y_label=np.concatenate([p.y_label for p in parts]),
        y_stress=np.concatenate([p.y_stress for p in parts]),
        groups=np.concatenate([p.groups for p in parts]),
        source_file=np.concatenate([p.source_file for p in parts]),
        period=period_all,
        n_windows_dropped=sum(p.n_windows_dropped for p in parts),
        meta={"files": [f.name for f in files]},
    )

    seen = {}
    for src, grp in zip(data.source_file, data.groups):
        if grp in seen and seen[grp] != src:
            raise ValueError(f"Survey_ID {grp!r} appears in both {seen[grp]} and {src}.")
        seen[grp] = src

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_kw = dict(
            X=data.X, y_label=data.y_label, y_stress=data.y_stress,
            groups=data.groups, source_file=data.source_file,
            n_windows_dropped=data.n_windows_dropped,
        )
        if data.period is not None:
            save_kw["period"] = data.period
        np.savez(cache_path, **save_kw)

    return data


def load_pooled_lineage_dataset(
    lineage: str,
    use_cache: bool = True,
    force_reload: bool = False,
    participant_ids: list[str] | None = None,
) -> LineageDataset:
    """Concatenate one lineage across multiple registered participants (generalized models).

    Survey_ID values are prefixed with the participant key so CV groups never mix
    windows from different people under the same ID.
    """
    from PilotPhaseC.config import PARTICIPANTS

    participant_ids = participant_ids or list(PARTICIPANTS.keys())
    unknown = [p for p in participant_ids if p not in PARTICIPANTS]
    if unknown:
        raise ValueError(f"Unknown participant id(s) in config.PARTICIPANTS: {unknown}")

    parts = []
    for pid in participant_ids:
        d = load_lineage_dataset(pid, lineage, use_cache=use_cache, force_reload=force_reload)
        if d.X.shape[0] == 0:
            warnings.warn(f"Participant {pid} has no windows for lineage {lineage} -- skipped.")
            continue
        groups = np.array([f"{pid}::{g}" for g in d.groups], dtype=object)
        source = np.array([f"{pid}/{s}" for s in d.source_file], dtype=object)
        parts.append(LineageDataset(
            X=d.X, y_label=d.y_label, y_stress=d.y_stress, groups=groups,
            source_file=source, period=d.period, n_windows_dropped=d.n_windows_dropped,
            meta={"participant": pid, **d.meta},
        ))

    if not parts:
        raise ValueError(f"No data loaded for lineage {lineage} across {participant_ids}")

    period_chunks = [p.period for p in parts if p.period is not None]
    period_all = np.concatenate(period_chunks) if period_chunks and len(period_chunks) == len(parts) else None
    return LineageDataset(
        X=np.concatenate([p.X for p in parts], axis=0),
        y_label=np.concatenate([p.y_label for p in parts]),
        y_stress=np.concatenate([p.y_stress for p in parts]),
        groups=np.concatenate([p.groups for p in parts]),
        source_file=np.concatenate([p.source_file for p in parts]),
        period=period_all,
        n_windows_dropped=sum(p.n_windows_dropped for p in parts),
        meta={"participants": participant_ids, "lineage": lineage},
    )
