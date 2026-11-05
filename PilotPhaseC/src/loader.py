"""
loader.py -- Parse one lineage's pre-labeled Excel/CSV files into a windowed
dataset (X, y_label, y_stress, groups), with an on-disk cache.
"""

from __future__ import annotations

import hashlib
import json
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PilotPhaseC.config import (
    PARTICIPANTS,
    DATASET_DIR,
    WINDOW_SAMPLES_LEGACY,
    WINDOW_SAMPLES_NEW,
    ROW_IN_WINDOW_COL,
    WINDOW_ID_COL,
    GROUP_COL,
    CLASS_LABEL_COL,
    REGRESSION_LABEL_COL,
    RAW_CHANNELS,
    MODEL_CHANNELS,
    PERIOD_COL,
    MOTION_SCORE_NONWEAR_MAX,
    MOTION_SCORE_HIGH_MIN,
    USE_PERIOD_WEIGHTS,
    GENERALIZED_CLASSIFICATION_PARTICIPANTS,
)

_N_MODEL_CHANNELS = len(MODEL_CHANNELS)

REQUIRED_COLUMNS = [
    WINDOW_ID_COL, ROW_IN_WINDOW_COL, GROUP_COL,
    CLASS_LABEL_COL, REGRESSION_LABEL_COL,
] + RAW_CHANNELS


@dataclass
class DataLoadSpec:
    """Controls file selection, window length, upsampling, and quality filters."""

    target_window_samples: int = WINDOW_SAMPLES_NEW
    file_set: str = "all"  # all | new_format | legacy
    include_legacy_upsampled: bool = False  # add legacy 112 files upsampled to target
    apply_nonwear_filter: bool = True
    apply_high_motion_filter: bool = False
    motion_nonwear_max: float | None = MOTION_SCORE_NONWEAR_MAX
    motion_high_min: float | None = MOTION_SCORE_HIGH_MIN

    def cache_slug(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


def default_load_spec(participant: str, lineage: str, generalized: bool = False) -> DataLoadSpec:
    if generalized and lineage == "all_continuous":
        return DataLoadSpec(
            target_window_samples=WINDOW_SAMPLES_LEGACY,
            file_set="legacy",
            apply_nonwear_filter=False,
        )
    p = PARTICIPANTS.get(participant, {})
    if participant in ("ring03_pid102", "ring04_pid101") and lineage == "all_continuous":
        fs = p.get("deploy_file_set", "legacy")
        return DataLoadSpec(
            target_window_samples=WINDOW_SAMPLES_LEGACY,
            file_set=fs,
            include_legacy_upsampled=False,
            apply_nonwear_filter=False,
        )
    return DataLoadSpec(
        target_window_samples=WINDOW_SAMPLES_LEGACY,
        file_set="all",
        include_legacy_upsampled=False,
        apply_nonwear_filter=False,
    )


def generalized_participant_spec(participant: str) -> DataLoadSpec:
    """Pooled generalized set: legacy 112-row windows only for every participant."""
    return DataLoadSpec(
        target_window_samples=WINDOW_SAMPLES_LEGACY,
        file_set="legacy",
        include_legacy_upsampled=False,
        apply_nonwear_filter=False,
    )


def _stack_window_features(win: pd.DataFrame) -> np.ndarray:
    base = win[RAW_CHANNELS].to_numpy(dtype=np.float32)
    led1, led2, led3 = base[:, 0], base[:, 1], base[:, 2]
    derived = np.stack([led3 - led1, led3 - led2], axis=1).astype(np.float32)
    return np.concatenate([base, derived], axis=1)


def _motion_score(win: pd.DataFrame) -> float:
    return float(win["AccX_mg"].std() + win["AccY_mg"].std() + win["AccZ_mg"].std())


def _upsample_window_features(feat: np.ndarray, target_len: int) -> np.ndarray:
    """Linear interpolation along time: (T, C) -> (target_len, C)."""
    t_src, c = feat.shape
    if t_src == target_len:
        return feat
    if t_src > target_len:
        raise ValueError(f"Cannot downsample {t_src} -> {target_len}")
    x_src = np.linspace(0.0, 1.0, t_src, dtype=np.float32)
    x_tgt = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    out = np.empty((target_len, c), dtype=np.float32)
    for j in range(c):
        out[:, j] = np.interp(x_tgt, x_src, feat[:, j])
    return out


@dataclass
class LineageDataset:
    X: np.ndarray
    y_label: np.ndarray
    y_stress: np.ndarray
    groups: np.ndarray
    source_file: np.ndarray
    period: np.ndarray | None = None
    n_windows_dropped: int = 0
    drop_reasons: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def _lineage_dir(participant: str, lineage: str) -> Path:
    p = PARTICIPANTS[participant]
    return p["dir"] / p["lineages"][lineage]


def _iter_data_files(lineage_dir: Path) -> list[Path]:
    csvs = {p.stem: p for p in lineage_dir.glob("*.csv")}
    xlsx = {p.stem: p for p in lineage_dir.glob("*.xlsx")}
    paths = []
    for stem in sorted(set(csvs) | set(xlsx)):
        paths.append(csvs.get(stem) or xlsx[stem])
    return paths


def _file_matches_set(path: Path, file_set: str) -> bool:
    if file_set == "all":
        return True
    is_new = "new_format" in path.name
    if file_set == "new_format":
        return is_new
    if file_set == "legacy":
        return not is_new
    raise ValueError(f"Unknown file_set: {file_set}")


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    return pd.read_excel(path, header=0)


def _native_window_samples(path: Path) -> int:
    return WINDOW_SAMPLES_NEW if "new_format" in path.name else WINDOW_SAMPLES_LEGACY


def _load_one_file(path: Path, spec: DataLoadSpec) -> LineageDataset:
    df = _read_table(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required column(s): {missing}")

    native_t = _native_window_samples(path)
    has_period = PERIOD_COL in df.columns and USE_PERIOD_WEIGHTS
    X_list, y_label_list, y_stress_list, group_list, period_list = [], [], [], [], []
    drops = {"wrong_row_count": 0, "ambiguous": 0, "nonwear": 0, "high_motion": 0}

    for window_id, win in df.groupby(WINDOW_ID_COL, sort=False):
        win = win.sort_values(ROW_IN_WINDOW_COL)

        if len(win) != native_t:
            drops["wrong_row_count"] += 1
            continue

        labels = win[CLASS_LABEL_COL].unique()
        stresses = win[REGRESSION_LABEL_COL].unique()
        groups_in_window = win[GROUP_COL].unique()
        if len(labels) != 1 or len(stresses) != 1 or len(groups_in_window) != 1:
            drops["ambiguous"] += 1
            continue

        if has_period:
            periods = win[PERIOD_COL].unique()
            if len(periods) != 1:
                drops["ambiguous"] += 1
                continue
            period_val = str(periods[0])
        else:
            period_val = None

        feat = _stack_window_features(win)
        if spec.apply_nonwear_filter and spec.motion_nonwear_max is not None:
            if _motion_score(win) < spec.motion_nonwear_max:
                drops["nonwear"] += 1
                continue
        if spec.apply_high_motion_filter and spec.motion_high_min is not None:
            if _motion_score(win) >= spec.motion_high_min:
                drops["high_motion"] += 1
                continue

        if native_t != spec.target_window_samples:
            feat = _upsample_window_features(feat, spec.target_window_samples)

        X_list.append(feat)
        y_label_list.append(labels[0])
        y_stress_list.append(float(stresses[0]))
        group_list.append(groups_in_window[0])
        if has_period:
            period_list.append(period_val)

    n = len(X_list)
    t = spec.target_window_samples
    period_arr = np.array(period_list, dtype=object) if has_period and period_list else None
    return LineageDataset(
        X=np.stack(X_list, axis=0) if n else np.empty((0, t, _N_MODEL_CHANNELS), dtype=np.float32),
        y_label=np.array(y_label_list, dtype=object),
        y_stress=np.array(y_stress_list, dtype=np.float32),
        groups=np.array(group_list, dtype=object),
        source_file=np.array([path.name] * n, dtype=object),
        period=period_arr,
        n_windows_dropped=int(sum(drops.values())),
        drop_reasons=drops,
        meta={"file": path.name, "native_window_samples": native_t},
    )


def _select_files(lineage_dir: Path, spec: DataLoadSpec) -> list[Path]:
    files = []
    for path in _iter_data_files(lineage_dir):
        if _file_matches_set(path, spec.file_set):
            files.append(path)
        elif spec.include_legacy_upsampled and _file_matches_set(path, "legacy"):
            files.append(path)
    if not files:
        raise FileNotFoundError(
            f"No data files matched file_set={spec.file_set!r} "
            f"(include_legacy_upsampled={spec.include_legacy_upsampled}) in {lineage_dir}"
        )
    return files


def _cache_path(participant: str, lineage: str, spec: DataLoadSpec) -> Path:
    return DATASET_DIR / participant / f"{lineage}_{spec.cache_slug()}.npz"


def load_lineage_dataset(
    participant: str,
    lineage: str,
    use_cache: bool = True,
    force_reload: bool = False,
    load_spec: DataLoadSpec | None = None,
) -> LineageDataset:
    spec = load_spec or default_load_spec(participant, lineage)
    cache_path = _cache_path(participant, lineage, spec)

    if use_cache and not force_reload and cache_path.exists():
        npz = np.load(cache_path, allow_pickle=True)
        stale = npz["X"].shape[2] != _N_MODEL_CHANNELS
        if not stale:
            period = npz["period"] if "period" in npz.files else None
            meta = {}
            if "meta_json" in npz.files:
                meta = json.loads(str(npz["meta_json"]))
            return LineageDataset(
                X=npz["X"], y_label=npz["y_label"], y_stress=npz["y_stress"],
                groups=npz["groups"], source_file=npz["source_file"],
                period=period,
                n_windows_dropped=int(npz["n_windows_dropped"]),
                meta=meta,
            )
        warnings.warn(f"Cache {cache_path} is stale -- rebuilding.")
        force_reload = True

    lineage_dir = _lineage_dir(participant, lineage)
    files = _select_files(lineage_dir, spec)
    parts = [_load_one_file(f, spec) for f in files]

    periods = [p.period for p in parts if p.period is not None]
    period_all = np.concatenate(periods) if periods and len(periods) == len(parts) else None
    drop_reasons = {}
    for p in parts:
        for k, v in p.drop_reasons.items():
            drop_reasons[k] = drop_reasons.get(k, 0) + v

    data = LineageDataset(
        X=np.concatenate([p.X for p in parts], axis=0),
        y_label=np.concatenate([p.y_label for p in parts]),
        y_stress=np.concatenate([p.y_stress for p in parts]),
        groups=np.concatenate([p.groups for p in parts]),
        source_file=np.concatenate([p.source_file for p in parts]),
        period=period_all,
        n_windows_dropped=sum(p.n_windows_dropped for p in parts),
        drop_reasons=drop_reasons,
        meta={
            "files": [f.name for f in files],
            "load_spec": asdict(spec),
        },
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
            meta_json=json.dumps(data.meta),
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
    per_participant_spec: dict[str, DataLoadSpec] | None = None,
) -> LineageDataset:
    participant_ids = participant_ids or list(GENERALIZED_CLASSIFICATION_PARTICIPANTS)
    unknown = [p for p in participant_ids if p not in PARTICIPANTS]
    if unknown:
        raise ValueError(f"Unknown participant id(s): {unknown}")

    parts = []
    for pid in participant_ids:
        if lineage not in PARTICIPANTS[pid]["lineages"]:
            warnings.warn(f"Participant {pid} has no lineage {lineage} -- skipped.")
            continue
        spec = (per_participant_spec or {}).get(pid) or generalized_participant_spec(pid)
        d = load_lineage_dataset(pid, lineage, use_cache=use_cache, force_reload=force_reload, load_spec=spec)
        if d.X.shape[0] == 0:
            warnings.warn(f"Participant {pid} has no windows for lineage {lineage} -- skipped.")
            continue
        groups = np.array([f"{pid}::{g}" for g in d.groups], dtype=object)
        source = np.array([f"{pid}/{s}" for s in d.source_file], dtype=object)
        parts.append(LineageDataset(
            X=d.X, y_label=d.y_label, y_stress=d.y_stress, groups=groups,
            source_file=source, period=d.period, n_windows_dropped=d.n_windows_dropped,
            drop_reasons=d.drop_reasons,
            meta={"participant": pid, **d.meta},
        ))

    if not parts:
        raise ValueError(f"No data loaded for lineage {lineage} across {participant_ids}")

    period_chunks = [p.period for p in parts if p.period is not None]
    period_all = np.concatenate(period_chunks) if period_chunks else None
    drop_reasons = {}
    for p in parts:
        for k, v in p.drop_reasons.items():
            drop_reasons[k] = drop_reasons.get(k, 0) + v

    return LineageDataset(
        X=np.concatenate([p.X for p in parts], axis=0),
        y_label=np.concatenate([p.y_label for p in parts]),
        y_stress=np.concatenate([p.y_stress for p in parts]),
        groups=np.concatenate([p.groups for p in parts]),
        source_file=np.concatenate([p.source_file for p in parts]),
        period=period_all,
        n_windows_dropped=sum(p.n_windows_dropped for p in parts),
        drop_reasons=drop_reasons,
        meta={"participants": participant_ids, "lineage": lineage, "generalized": True},
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and summarize a Pilot Phase C lineage.")
    parser.add_argument("--participant", default=None)
    parser.add_argument("--lineage", required=True)
    parser.add_argument("--force-reload", action="store_true")
    parser.add_argument("--generalized", action="store_true")
    parser.add_argument("--include-legacy-upsampled", action="store_true")
    args = parser.parse_args()

    if args.generalized:
        d = load_pooled_lineage_dataset(args.lineage, force_reload=args.force_reload)
    else:
        if not args.participant:
            parser.error("--participant is required unless --generalized")
        spec = default_load_spec(args.participant, args.lineage)
        if args.include_legacy_upsampled:
            spec = DataLoadSpec(
                target_window_samples=WINDOW_SAMPLES_NEW,
                file_set=spec.file_set,
                include_legacy_upsampled=True,
                apply_nonwear_filter=spec.apply_nonwear_filter,
            )
        d = load_lineage_dataset(args.participant, args.lineage, force_reload=args.force_reload, load_spec=spec)

    print(f"X={d.X.shape} groups={len(np.unique(d.groups))} dropped={d.n_windows_dropped}")
    print(f"drop_reasons={d.drop_reasons}")
    print(pd.Series(d.y_label).value_counts())
