"""
Raw data loader for the Pilot B cohort (4 new participants, no MATLAB feature
extraction available -- raw sensor CSVs + Qualtrics EMA exports only).

Differences from the original ``src/data/raw_preprocess.py`` pipeline:

  * File layout is messy: boot-log preambles, garbled binary bytes on some
    lines, mid-file device reboots (the elapsed-ms counter resets), and one
    participant (Aisha) has an extra ``humidityPct`` column that must be
    dropped so her channel schema matches everyone else's.
  * Device wall-clock dates are unreliable for some participants (do not
    match any survey date at all in one case).  We therefore trust the
    human-curated folder/filename pairing between a raw file and a survey
    day (``PARTICIPANT_DAY_MAP`` below) instead of the embedded device date,
    and align survey time to raw time using elapsed-seconds-since-first-
    event -- the same convention ``src/data/merge_labels.py`` already uses.
  * There are no pre-computed MATLAB blocks to match against: labels are
    assigned directly to each burst via the survey-interval lookup.
  * Each burst is windowed as two 5 s halves with the leading 0.5 s of each
    half discarded (~4.5 s / 112-sample usable windows), per the user's
    specification -- distinct from raw_preprocess.py's simpler "two
    consecutive un-trimmed 112-sample chunks" approximation.
  * Ring-wear: a burst-level accelerometer-flatline check discards any
    trailing run of bursts (to the end of that day) where the ring was
    evidently not being worn.

Public API
----------
    load_pilot_b_dataset()
        -> X (N, 112, 9) float32, y (N,) str, groups (N,) str, participants (N,) int,
           day_summary : list[dict]   (per participant-day diagnostics)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    DATA_DIR,
    RAW_SAMPLING_HZ,
    RAW_GAP_THRESHOLD_MS,
    RAW_WINDOW_SAMPLES,
    RAW_CHANNELS,
    PILOT_B_FLATLINE_FRACTION,
    PILOT_B_FLATLINE_MIN_BURSTS,
    PILOT_B_MIN_BURST_S,
)
from src.data.classify_survey import text_to_option, classify
from src.data.merge_labels import build_survey_intervals, assign_label

PILOT_B_EXTRACTED_DIR = DATA_DIR / "pliot_b" / "extracted"
SURVEY_MASTER_FILE = PILOT_B_EXTRACTED_DIR / "4_EMMA_503" / "EMMA" / "Survey_data.csv"

HALF_MS = 5000.0
EDGE_TRIM_MS = 500.0

AISHA_PARTICIPANT_ID = 504
SOPHIA_PARTICIPANT_ID = 505

# Channels prone to slow (minutes-scale) drift unrelated to the fast
# (seconds-scale) physiological signal -- skin temperature settling, LED
# baseline shifting as the ring warms/seats against skin, ambient humidity
# creep. A model given raw absolute levels of these can learn to key off
# "how far the baseline has drifted" as a cheap proxy for elapsed session
# time, which is spuriously predictive within any one participant-day (see
# detrend_slow_channels). Opt-in via the ``detrend`` flag on
# load_pilot_b_participant_dataset -- off by default so existing models are
# unaffected.
DETREND_CHANNELS = ["LED1", "LED2", "LED3", "Temp", "Humidity"]
DETREND_BASELINE_BURSTS = 10   # trailing window, in bursts, for the rolling baseline

# 10-channel variant for personalized models of participants whose device has
# a humidity sensor (currently Aisha and Sophia -- most participants' devices
# don't have one). Humidity is inserted next to Temp, matching the raw column
# order on those devices (LED1, LED2, LED3, Temp, Humidity, Acc*).
HUMIDITY_CHANNELS = [
    "LED1", "LED2", "LED3", "Temp", "Humidity", "AccX", "AccY", "AccZ",
    "LED3_LED1", "LED3_LED2",
]

SENSOR_COLS_BASE = ["timestamp_ms", "LED1", "LED2", "LED3", "Temp", "AccX", "AccY", "AccZ"]

# ---------------------------------------------------------------------------
# Participant / day-pairing table
#
# Raw device dates are not trustworthy (see plan notes).  Each entry pairs a
# raw file with the calendar date its survey responses were confirmed to be
# under in the master Survey_data.csv, using the folder/filename grouping the
# data was organized in as the source of truth for "which day is this."
#
# survey_override: normally survey rows are pulled from SURVEY_MASTER_FILE
# (a snapshot export bundled in Emma's zip). If a participant's raw data is
# updated after that snapshot was taken, the master file won't have their
# newest day's responses -- point survey_override at that day's own
# participant-specific survey xlsx instead. Aisha's day4 needs this: the
# master file predates her July 29 session.
# ---------------------------------------------------------------------------
PARTICIPANT_DAY_MAP = [
    # (participant_id, name, relative_csv_path, has_humidity, survey_date, survey_override)
    (501, "Shuail", "2_Shuail_501/1_suhail_501.csv",        False, "2026-06-01", None),
    (501, "Shuail", "2_Shuail_501/2_suhail_501_6_2.csv",     False, "2026-06-02", None),
    (501, "Shuail", "2_Shuail_501/3_suhail_501_6_3.csv",     False, "2026-06-03", None),

    (502, "Lehar",  "3_Lehar_502/Day1_june6/lehar1.csv",     False, "2026-06-17", None),
    (502, "Lehar",  "3_Lehar_502/Day2_June24/lehar2.csv",    False, "2026-06-24", None),
    (502, "Lehar",  "3_Lehar_502/Day3_July6th/lehar3.csv",   False, "2026-07-06", None),

    (503, "Emma",   "4_EMMA_503/EMMA/emma_d1.csv",           False, "2026-07-07", None),
    (503, "Emma",   "4_EMMA_503/EMMA/emma_d2.csv",           False, "2026-07-08", None),
    (503, "Emma",   "4_EMMA_503/EMMA/emma_d4.csv",           False, "2026-07-10", None),
    (503, "Emma",   "4_EMMA_503/EMMA/emma_d5.csv",           False, "2026-07-13", None),

    (504, "Aisha",  "5_Aisha_504/Aisha_day1.csv",            True,  "2026-07-23", None),
    (504, "Aisha",  "5_Aisha_504/Aisha_day2.csv",            True,  "2026-07-24", None),
    (504, "Aisha",  "5_Aisha_504/Aisha_day4.csv",            True,  "2026-07-29",
     "5_Aisha_504/Aisha_survey_day4.xlsx"),

    # Sophia's raw dates are reliable (unlike most other participants), but
    # her data was collected entirely after SURVEY_MASTER_FILE's snapshot
    # date (2026-07-28) -- the master has no rows for her at all, so every
    # day needs its own survey_override.
    (505, "Sophia", "6_Sophia_505/Sophia_day1.csv",          True,  "2026-07-27",
     "6_Sophia_505/Sophia_day1_survey.xlsx"),
    (505, "Sophia", "6_Sophia_505/Sophia_day2.csv",          True,  "2026-07-29",
     "6_Sophia_505/Sophia_day2_survey.xlsx"),
    (505, "Sophia", "6_Sophia_505/sophia_day3.csv",          True,  "2026-08-05",
     "6_Sophia_505/Sophia_day3_survey.xlsx"),
    (505, "Sophia", "6_Sophia_505/sophia_day4.csv",          True,  "2026-08-06",
     "6_Sophia_505/Sophia_day4_survey.xlsx"),
]


def participant_has_humidity(participant_id: int) -> bool:
    """Whether this participant's device recorded a humidity channel."""
    return any(
        row[0] == participant_id and row[3]
        for row in PARTICIPANT_DAY_MAP
    )


# ---------------------------------------------------------------------------
# Step 1: robust raw-file parsing
# ---------------------------------------------------------------------------

def load_pilot_b_file(path, has_humidity: bool) -> pd.DataFrame:
    """
    Parse one raw pilot-B CSV, tolerant of boot-log preambles, garbled bytes,
    trailer messages, and files that are missing their header row entirely.

    Every file (regardless of header presence) is read as headerless text,
    then every sensor column is numeric-coerced and rows that fail to
    coerce are dropped.  This single mechanism transparently removes boot
    logs, header text, trailer messages, and corrupted lines without any
    special-casing.
    """
    ncols = 10 if has_humidity else 9
    names = ["WallClock", "timestamp_ms", "LED1", "LED2", "LED3", "Temp"]
    if has_humidity:
        names.append("Humidity")
    names += ["AccX", "AccY", "AccZ"]

    raw = pd.read_csv(
        path, header=None, dtype=str, engine="python",
        on_bad_lines="skip", skip_blank_lines=True,
        encoding="latin-1",
    )
    raw = raw.iloc[:, :ncols].copy()
    raw.columns = names

    numeric_cols = [c for c in names if c != "WallClock"]
    for c in numeric_cols:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.dropna(subset=numeric_cols).reset_index(drop=True)

    # Humidity is kept in the dataframe when present (Aisha, Sophia); callers
    # decide whether to include it in a given model's channel list via the
    # `channels` argument to extract_windows(). The generalized cross-
    # participant model uses RAW_CHANNELS (no Humidity, for a uniform 9-ch
    # schema); personalized models for participants with a humidity sensor
    # use HUMIDITY_CHANNELS (10-ch).

    raw["LED3_LED1"] = raw["LED3"] - raw["LED1"]
    raw["LED3_LED2"] = raw["LED3"] - raw["LED2"]

    return raw


# ---------------------------------------------------------------------------
# Step 2: burst detection (handles both OFF-period gaps and mid-file reboots)
# ---------------------------------------------------------------------------

def find_bursts(df: pd.DataFrame) -> list[tuple[int, int]]:
    t = df["timestamp_ms"].values
    diffs = np.diff(t)
    boundary = np.where((diffs > RAW_GAP_THRESHOLD_MS) | (diffs < -RAW_GAP_THRESHOLD_MS))[0]

    starts = np.concatenate([[0], boundary + 1])
    ends = np.concatenate([boundary, [len(t) - 1]])

    bursts = []
    for s, e in zip(starts, ends):
        if e <= s:
            continue
        dur_s = (t[e] - t[s]) / 1000.0
        if dur_s >= PILOT_B_MIN_BURST_S:
            bursts.append((int(s), int(e)))
    return bursts


# ---------------------------------------------------------------------------
# Step 3: ring-wear -- crop a trailing accelerometer-flatline run
# ---------------------------------------------------------------------------

def acc_total_std(df: pd.DataFrame, s: int, e: int) -> float:
    seg = df.iloc[s:e + 1]
    return float(seg["AccX"].std() + seg["AccY"].std() + seg["AccZ"].std())


def crop_ring_off_tail(bursts: list[tuple[int, int]], burst_stds: list[float], day_baseline: float) -> int:
    """
    Return the number of trailing bursts to discard because the ring was
    evidently no longer being worn (accelerometer flatlined and stayed that
    way through the end of the day).

    ``day_baseline`` anchors to *this day's own* active-wear activity level
    (its 75th percentile of Acc_Total_STD) rather than a baseline pooled
    across all of a participant's days -- a participant whose other days
    happen to be mostly sedentary-but-worn would otherwise drag a pooled
    median down near the true "ring off" noise floor and mask a real
    end-of-day removal on a different day.
    """
    if day_baseline <= 0 or len(bursts) < PILOT_B_FLATLINE_MIN_BURSTS:
        return 0

    threshold = PILOT_B_FLATLINE_FRACTION * day_baseline

    # Walk backward from the end, extending the flatlined run while it holds.
    run_len = 0
    for std in reversed(burst_stds):
        if std < threshold:
            run_len += 1
        else:
            break

    if run_len >= PILOT_B_FLATLINE_MIN_BURSTS:
        return run_len
    return 0


# ---------------------------------------------------------------------------
# Step 3b: detrend slow-drift channels (opt-in)
# ---------------------------------------------------------------------------

def detrend_slow_channels(
    df: pd.DataFrame,
    bursts: list[tuple[int, int]],
    channels: list[str] = DETREND_CHANNELS,
    baseline_n_bursts: int = DETREND_BASELINE_BURSTS,
) -> pd.DataFrame:
    """
    Return a copy of ``df`` where each of ``channels`` has had a rolling,
    causal per-burst baseline subtracted, so the model sees deviation from
    recent local level rather than absolute (slowly drifting) level.

    For each burst, the baseline is the median of that channel's mean over
    the current burst and the preceding ``baseline_n_bursts - 1`` bursts
    (median, not mean, so a single unusually active burst doesn't skew the
    baseline). Causal (only past + current bursts) so this would also be
    valid to compute in an online/on-device setting, not just offline.

    Only ``bursts`` passed in are used to build the baseline -- pass the
    ring-wear-cropped burst list, not the raw pre-crop one, so a flatlined
    tail can't pull the baseline down.
    """
    df = df.copy()
    if not bursts:
        return df

    burst_means = {ch: [] for ch in channels}
    for s, e in bursts:
        seg = df.iloc[s:e + 1]
        for ch in channels:
            burst_means[ch].append(seg[ch].mean())

    baselines = {
        ch: pd.Series(burst_means[ch]).rolling(window=baseline_n_bursts, min_periods=1).median()
        for ch in channels
    }

    for bi, (s, e) in enumerate(bursts):
        for ch in channels:
            df.loc[s:e, ch] = df.loc[s:e, ch] - baselines[ch].iloc[bi]

    if "LED3" in channels or "LED1" in channels or "LED2" in channels:
        df["LED3_LED1"] = df["LED3"] - df["LED1"]
        df["LED3_LED2"] = df["LED3"] - df["LED2"]

    return df


# ---------------------------------------------------------------------------
# Step 4: windowing -- two 5 s halves, leading 0.5 s of each discarded
# ---------------------------------------------------------------------------

FULL_DUTY_CYCLE_MS = 8000.0   # bursts at/above this are a 10s ON cycle (2 halves);
                              # shorter bursts are a single ~5s half (e.g. Aisha's
                              # device, which duty-cycles at 5s ON, or a truncated
                              # partial burst from another participant)


def extract_windows(df: pd.DataFrame, s: int, e: int, channels: list[str] = RAW_CHANNELS) -> np.ndarray:
    seg = df.iloc[s:e + 1]
    t0 = seg["timestamp_ms"].iloc[0]
    t_rel = seg["timestamp_ms"].values - t0
    duration_ms = t_rel[-1]

    half_starts = (0.0, HALF_MS) if duration_ms >= FULL_DUTY_CYCLE_MS else (0.0,)

    windows = []
    for half_start in half_starts:
        lo = half_start + EDGE_TRIM_MS
        hi = half_start + HALF_MS
        mask = (t_rel >= lo) & (t_rel <= hi)
        sub = seg.loc[mask, channels]
        if len(sub) >= RAW_WINDOW_SAMPLES:
            windows.append(sub.iloc[:RAW_WINDOW_SAMPLES].values.astype(np.float32))

    if not windows:
        return np.empty((0, RAW_WINDOW_SAMPLES, len(channels)), dtype=np.float32)
    return np.stack(windows, axis=0)


# ---------------------------------------------------------------------------
# Step 5: survey labeling for one participant-day
# ---------------------------------------------------------------------------

def _load_survey_master() -> pd.DataFrame:
    return pd.read_csv(SURVEY_MASTER_FILE, header=0, skiprows=[1, 2], dtype=str)


def _classify_survey_rows(df: pd.DataFrame, label_column: str | None = None):
    """
    Given survey rows already filtered to one participant-day, return sorted
    (start, end, label) intervals.

    ``label_column``: if None (default), use the standard Opportunity/Pressure
    -> 4-class scheme (classify_survey.classify). If set to a column-name
    substring (e.g. "mental state"), skip that scheme entirely and use the
    matching column's raw value directly as the label instead -- one class
    per distinct value found. Rows where that column is blank are dropped
    (same as an unrecognized Opportunity/Pressure response would be).
    """
    if df.empty:
        return []

    df = df.copy()

    if label_column:
        col = next(c for c in df.columns if label_column.lower() in c.strip().lower())
        df["Label"] = df[col].apply(lambda v: v.strip() if isinstance(v, str) and v.strip() else None)
    else:
        opp_col = next(c for c in df.columns if "opportunity" in c.strip().lower())
        pres_col = next(c for c in df.columns if "pressure" in c.strip().lower())
        df["Opportunity_option"] = df[opp_col].apply(text_to_option)
        df["Pressure_option"] = df[pres_col].apply(text_to_option)
        df["Label"] = df.apply(
            lambda r: classify(r["Opportunity_option"], r["Pressure_option"]), axis=1
        )

    df = df.dropna(subset=["Label"])
    if df.empty:
        return []

    survey_df = df[["StartDate", "EndDate", "Label"]].sort_values("StartDate").reset_index(drop=True)
    return build_survey_intervals(survey_df)


def _day_survey_intervals(survey_master: pd.DataFrame, participant_id: int, survey_date: str, label_column: str | None = None):
    """Survey rows for one participant-day, filtered out of the bundled master snapshot."""
    pid_col = next(c for c in survey_master.columns if c.strip().lower() == "participant id")
    df = survey_master[survey_master[pid_col].astype(str).str.strip() == str(participant_id)].copy()
    df = df[df["StartDate"].astype(str).str.startswith(survey_date)].copy()
    return _classify_survey_rows(df, label_column=label_column)


def _day_survey_intervals_from_xlsx(xlsx_path, participant_id: int, label_column: str | None = None):
    """
    Survey rows for one participant-day, read directly from a per-day survey
    export instead of the (possibly stale) bundled master snapshot.
    """
    df = pd.read_excel(xlsx_path, header=0, skiprows=[1, 2], dtype=str)
    pid_col = next(c for c in df.columns if c.strip().lower() == "participant id")
    df = df[df[pid_col].astype(str).str.strip() == str(participant_id)].copy()
    return _classify_survey_rows(df, label_column=label_column)


# ---------------------------------------------------------------------------
# Step 6: full pipeline
# ---------------------------------------------------------------------------

def _parse_days(participant_filter: int | list[int] | None = None) -> list[dict]:
    if isinstance(participant_filter, int):
        participant_filter = {participant_filter}
    elif participant_filter is not None:
        participant_filter = set(participant_filter)

    parsed_days = []
    for participant_id, name, rel_path, has_humidity, survey_date, survey_override in PARTICIPANT_DAY_MAP:
        if participant_filter is not None and participant_id not in participant_filter:
            continue
        path = PILOT_B_EXTRACTED_DIR / rel_path
        df = load_pilot_b_file(path, has_humidity)
        bursts = find_bursts(df)
        burst_stds = [acc_total_std(df, s, e) for s, e in bursts]

        parsed_days.append(dict(
            participant_id=participant_id, name=name, rel_path=rel_path,
            survey_date=survey_date, survey_override=survey_override,
            df=df, bursts=bursts, burst_stds=burst_stds,
        ))
    return parsed_days


def _process_day(day: dict, survey_master: pd.DataFrame, channels: list[str], detrend: bool = False, label_column: str | None = None):
    """Window + label one participant-day. Returns (X_list, y_list, groups_list, summary)."""
    pid = day["participant_id"]
    df = day["df"]
    bursts = day["bursts"]
    burst_stds = day["burst_stds"]

    day_baseline = float(np.percentile(burst_stds, 75)) if burst_stds else 0.0
    n_crop = crop_ring_off_tail(bursts, burst_stds, day_baseline)
    kept_bursts = bursts[: len(bursts) - n_crop] if n_crop else bursts

    if detrend:
        df = detrend_slow_channels(df, kept_bursts)

    if day.get("survey_override"):
        intervals = _day_survey_intervals_from_xlsx(PILOT_B_EXTRACTED_DIR / day["survey_override"], pid, label_column=label_column)
    else:
        intervals = _day_survey_intervals(survey_master, pid, day["survey_date"], label_column=label_column)

    X_list, y_list, groups_list = [], [], []
    n_windows_labeled = 0
    n_windows_unknown = 0

    if intervals:
        t0 = df["timestamp_ms"].iloc[kept_bursts[0][0]] if kept_bursts else None

        for burst_idx, (s, e) in enumerate(kept_bursts):
            burst_start_rel_s = (df["timestamp_ms"].iloc[s] - t0) / 1000.0
            label = assign_label(burst_start_rel_s, intervals)

            windows = extract_windows(df, s, e, channels=channels)
            if len(windows) == 0:
                continue

            if label == "Unknown":
                n_windows_unknown += len(windows)
                continue

            group_id = f"{pid}_{day['rel_path']}_{burst_idx}"
            for w in windows:
                X_list.append(w)
                y_list.append(label)
                groups_list.append(group_id)
            n_windows_labeled += len(windows)

    summary = dict(
        participant_id=pid, name=day["name"], rel_path=day["rel_path"],
        survey_date=day["survey_date"],
        n_bursts_found=len(bursts), n_bursts_cropped_ringoff=n_crop,
        n_bursts_kept=len(kept_bursts),
        n_windows_labeled=n_windows_labeled, n_windows_unknown_dropped=n_windows_unknown,
        has_survey=bool(intervals),
    )
    return X_list, y_list, groups_list, summary


def load_pilot_b_dataset(participant_ids: list[int] | None = None, label_column: str | None = None):
    """
    Pooled multi-participant dataset, uniform 9-channel schema (no Humidity).

    ``participant_ids``: defaults to every participant in PARTICIPANT_DAY_MAP;
    pass an explicit list to restrict to a subset.

    ``label_column``: defaults to the standard Opportunity/Pressure 4-class
    scheme; pass a column-name substring (e.g. "mental state") to label
    directly from that survey column's raw values instead -- see
    _classify_survey_rows for details. One-off/experimental use only.
    """
    survey_master = _load_survey_master()
    parsed_days = _parse_days(participant_filter=participant_ids)

    all_X, all_y, all_groups, all_participants = [], [], [], []
    day_summary = []

    for day in parsed_days:
        X_list, y_list, groups_list, summary = _process_day(day, survey_master, RAW_CHANNELS, label_column=label_column)
        all_X.extend(X_list)
        all_y.extend(y_list)
        all_groups.extend(groups_list)
        all_participants.extend([day["participant_id"]] * len(X_list))
        day_summary.append(summary)

    X = np.stack(all_X, axis=0)
    y = np.array(all_y, dtype=object)
    groups = np.array(all_groups, dtype=object)
    participants = np.array(all_participants, dtype=int)

    return X, y, groups, participants, day_summary


def load_pilot_b_participant_dataset(
    participant_id: int, channels: list[str] | None = None, detrend: bool = False,
    label_column: str | None = None,
):
    """
    Load one participant's own data only, for a personalized model.

    ``channels`` defaults to the standard 9-channel RAW_CHANNELS; pass
    HUMIDITY_CHANNELS to include a 10th channel for participants whose
    device has a humidity sensor (Aisha, Sophia).

    ``detrend``: if True, subtract a rolling per-burst baseline from the
    slow-drift-prone channels (see detrend_slow_channels) before windowing.
    Off by default -- opt in per training run, doesn't change existing models.

    ``label_column``: defaults to the standard Opportunity/Pressure 4-class
    scheme; pass a column-name substring (e.g. "mental state") to label
    directly from that survey column's raw values instead. One-off/
    experimental use only -- see _classify_survey_rows.

    Returns
    -------
    X : (N, RAW_WINDOW_SAMPLES, len(channels)) float32
    y : (N,) str
    groups : (N,) str
    day_summary : list[dict]
    """
    channels = channels or RAW_CHANNELS
    survey_master = _load_survey_master()
    parsed_days = _parse_days(participant_filter=participant_id)

    all_X, all_y, all_groups = [], [], []
    day_summary = []

    for day in parsed_days:
        X_list, y_list, groups_list, summary = _process_day(day, survey_master, channels, detrend=detrend, label_column=label_column)
        all_X.extend(X_list)
        all_y.extend(y_list)
        all_groups.extend(groups_list)
        day_summary.append(summary)

    X = np.stack(all_X, axis=0)
    y = np.array(all_y, dtype=object)
    groups = np.array(all_groups, dtype=object)

    return X, y, groups, day_summary
