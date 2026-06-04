"""
Raw M3X sensor data loader and windowing pipeline for the 1D CNN.

The M3X ring records in a 10 s ON / ~21 s OFF duty cycle.  Each ON burst
contains ~250 samples at ~25 Hz.  We split every burst into
RAW_N_WINDOWS_PER_BURST non-overlapping windows of RAW_WINDOW_SAMPLES samples
(~ 4.5 s each), exactly mirroring the two sub-windows used by the feature-
based models (SVM / RF / XGBoost).

Label assignment
----------------
For each burst we find the nearest feature block by Block_Start_s in the
labeled CSV.  If the nearest block is within MATCH_TOLERANCE_S seconds, that
block's label is used for both windows.  Bursts with no match (>MATCH_TOLERANCE_S)
are dropped -- these correspond to bursts that MATLAB also discarded (e.g. the
first startup burst in session 2).

Public API
----------
    load_raw_cnn_dataset(s1_csv, s2_csv, labeled_csv)
        -> X  : np.ndarray  (N, RAW_WINDOW_SAMPLES, len(RAW_CHANNELS))
           y  : np.ndarray  (N,)  string labels
           groups : np.ndarray (N,)  Block_IDs for StratifiedGroupKFold
"""

import numpy as np
import pandas as pd

from config import (
    RAW_GAP_THRESHOLD_MS,
    RAW_MIN_BURST_S,
    RAW_WINDOW_SAMPLES,
    RAW_N_WINDOWS_PER_BURST,
    RAW_CHANNELS,
    LABELED_FEATURES_FILE,
    LABELED_FEATURES_MF_FILE,
    RAW_SESSION1_FILE,
    RAW_SESSION2_FILE,
    RAW_MF_SESSION1_FILE,
    RAW_MF_SESSION2_FILE,
)

MATCH_TOLERANCE_S = 2.0   # max allowed |burst_start - Block_Start_s| for a match

# ---------------------------------------------------------------------------
# Step 1: Load and clean a single raw CSV
# ---------------------------------------------------------------------------

def load_raw_session(csv_path) -> pd.DataFrame:
    """
    Load a raw M3X CSV, drop trailer rows, cast to float.

    The file has one header row followed by data rows.  The last 2-3 rows are
    plain-text status messages (END_OF_VALID_DATA, FLASH: ..., READ COMPLETE)
    that must be removed before numeric conversion.
    """
    df = pd.read_csv(
        csv_path,
        header=0,
        names=["Time_PPG", "LED1", "LED2", "LED3", "Temp", "AccX", "AccY", "AccZ"],
        skiprows=1,
        low_memory=False,
    )
    # Keep only rows where Time_PPG is numeric
    mask = pd.to_numeric(df["Time_PPG"], errors="coerce").notna()
    df = df[mask].copy()
    df = df.astype(float)
    df = df.reset_index(drop=True)

    # Derived channels requested by the model specification
    df["LED3_LED1"] = df["LED3"] - df["LED1"]
    df["LED3_LED2"] = df["LED3"] - df["LED2"]

    return df


# ---------------------------------------------------------------------------
# Step 2: Segment into duty-cycle ON bursts
# ---------------------------------------------------------------------------

def find_bursts(df: pd.DataFrame) -> list[tuple[int, int]]:
    """
    Return a list of (start_idx, end_idx) pairs, one per ON burst.

    A burst boundary is any consecutive sample pair whose timestamp gap exceeds
    RAW_GAP_THRESHOLD_MS milliseconds.  Bursts shorter than RAW_MIN_BURST_S
    seconds are discarded (typically the first startup burst and any partial
    burst at the end of a session).
    """
    t = df["Time_PPG"].values
    diffs = np.diff(t)
    gap_positions = np.where(diffs > RAW_GAP_THRESHOLD_MS)[0]

    starts = np.concatenate([[0], gap_positions + 1])
    ends   = np.concatenate([gap_positions, [len(t) - 1]])

    bursts = []
    for s, e in zip(starts, ends):
        dur_s = (t[e] - t[s]) / 1000.0
        if dur_s >= RAW_MIN_BURST_S:
            bursts.append((int(s), int(e)))

    return bursts


# ---------------------------------------------------------------------------
# Step 3: Build windowed numpy arrays from valid bursts
# ---------------------------------------------------------------------------

def _windows_from_burst(df: pd.DataFrame, start: int, end: int) -> np.ndarray:
    """
    Extract RAW_N_WINDOWS_PER_BURST non-overlapping windows from one burst.

    Returns array of shape (n_windows, RAW_WINDOW_SAMPLES, len(RAW_CHANNELS))
    where n_windows <= RAW_N_WINDOWS_PER_BURST.  Windows that don't have enough
    samples are dropped.
    """
    data = df.loc[start:end, RAW_CHANNELS].values  # (n_burst_samples, n_ch)
    windows = []
    for i in range(RAW_N_WINDOWS_PER_BURST):
        lo = i * RAW_WINDOW_SAMPLES
        hi = lo + RAW_WINDOW_SAMPLES
        if hi <= len(data):
            windows.append(data[lo:hi])
    return np.array(windows, dtype=np.float32) if windows else np.empty((0, RAW_WINDOW_SAMPLES, len(RAW_CHANNELS)), dtype=np.float32)


# ---------------------------------------------------------------------------
# Step 4: Match bursts to labeled feature blocks
# ---------------------------------------------------------------------------

def _compute_session_offset(
    burst_start_times_s: np.ndarray,
    feature_blocks: pd.DataFrame,
) -> float:
    """
    Determine the additive offset that maps raw relative burst times to the
    Block_Start_s coordinate system used in the labeled feature CSV.

    Iterates over the first few bursts until one finds a match within
    MATCH_TOLERANCE_S * 10.  Returns the offset so that
        burst_relative_s + offset ~ Block_Start_s
    """
    block_starts = feature_blocks["Block_Start_s"].values
    for t in burst_start_times_s[:10]:
        diffs = np.abs(block_starts - t)
        nearest = int(np.argmin(diffs))
        if diffs[nearest] <= MATCH_TOLERANCE_S * 10:
            # refine: the nearest block gives us the offset
            return float(block_starts[nearest] - t)
    # If nothing matched with 20 s tolerance, fall back to zero offset
    return 0.0


def match_bursts_to_labels(
    df: pd.DataFrame,
    bursts: list[tuple[int, int]],
    feature_blocks: pd.DataFrame,
) -> list[tuple[int, int, str, int]]:
    """
    For every valid burst, find the nearest labeled block and return a list of
    (burst_start_idx, burst_end_idx, label, block_id) tuples.

    Bursts that don't match any block within MATCH_TOLERANCE_S are dropped.
    """
    t = df["Time_PPG"].values
    first_t_ms = t[0]

    # Burst start times relative to first raw sample (seconds)
    burst_start_rel = np.array(
        [(t[s] - first_t_ms) / 1000.0 for s, _ in bursts],
        dtype=float,
    )

    block_starts = feature_blocks["Block_Start_s"].values
    offset = _compute_session_offset(burst_start_rel, feature_blocks)

    matched = []
    for i, (s, e) in enumerate(bursts):
        adjusted = burst_start_rel[i] + offset
        diffs    = np.abs(block_starts - adjusted)
        nearest  = int(np.argmin(diffs))
        if diffs[nearest] <= MATCH_TOLERANCE_S:
            label    = feature_blocks.iloc[nearest]["Label"]
            block_id = int(feature_blocks.iloc[nearest]["Block_ID"])
            matched.append((s, e, label, block_id))

    return matched


# ---------------------------------------------------------------------------
# Step 5: Assemble the full dataset from both sessions
# ---------------------------------------------------------------------------

def _session_feature_blocks(labeled_df: pd.DataFrame, session: int) -> pd.DataFrame:
    """
    Extract per-block (Block_Start_s, Label) for one session from the combined
    labeled CSV.  Session identity is inferred from the natural gap in Block_IDs
    that results from the offset applied when the two sessions were concatenated.
    """
    block_ids = sorted(labeled_df["Block_ID"].unique())
    # Find the largest jump in Block_IDs -- that is the session boundary
    id_arr = np.array(block_ids)
    jumps  = np.diff(id_arr)
    split  = int(np.argmax(jumps))          # index of the last session-1 ID
    s1_max = id_arr[split]

    if session == 1:
        mask = labeled_df["Block_ID"] <= s1_max
    else:
        mask = labeled_df["Block_ID"] > s1_max

    # One row per block (SubWindow_ID == 1 carries the representative metadata)
    blocks = (
        labeled_df[mask & (labeled_df["SubWindow_ID"] == 1)]
        [["Block_ID", "Block_Start_s", "Label"]]
        .sort_values("Block_Start_s")
        .reset_index(drop=True)
    )
    return blocks


def load_raw_cnn_dataset(
    s1_csv=None,
    s2_csv=None,
    labeled_csv=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full pipeline: raw CSVs -> labeled windowed arrays.

    Parameters
    ----------
    s1_csv, s2_csv  : paths to session raw CSVs  (default: from config)
    labeled_csv     : path to features_labeled_new.csv  (default: from config)

    Returns
    -------
    X      : (N, RAW_WINDOW_SAMPLES, len(RAW_CHANNELS))  float32
    y      : (N,)  str   stress labels
    groups : (N,)  int   Block_IDs for StratifiedGroupKFold
    """
    s1_csv     = s1_csv     or RAW_SESSION1_FILE
    s2_csv     = s2_csv     or RAW_SESSION2_FILE
    labeled_csv = labeled_csv or LABELED_FEATURES_FILE

    labeled_df = pd.read_csv(labeled_csv)

    all_X, all_y, all_groups = [], [], []

    for session, csv_path in [(1, s1_csv), (2, s2_csv)]:
        df     = load_raw_session(csv_path)
        bursts = find_bursts(df)
        blocks = _session_feature_blocks(labeled_df, session)

        # Drop Unknown labels (same as feature pipeline)
        blocks = blocks[blocks["Label"] != "Unknown"].reset_index(drop=True)

        matched = match_bursts_to_labels(df, bursts, blocks)

        n_matched = len(matched)
        n_bursts  = len(bursts)
        print(
            f"  Session {session}: {n_bursts} raw bursts -> "
            f"{n_matched} matched to labeled blocks"
        )

        for s, e, label, block_id in matched:
            windows = _windows_from_burst(df, s, e)
            if len(windows) == 0:
                continue
            for w in windows:
                all_X.append(w)
                all_y.append(label)
                all_groups.append(block_id)

    X      = np.stack(all_X,      axis=0)
    y      = np.array(all_y,      dtype=object)
    groups = np.array(all_groups, dtype=int)

    print(f"\nRaw CNN dataset: {X.shape[0]} windows  "
          f"({X.shape[1]} timesteps x {X.shape[2]} channels)")
    print("Label distribution:")
    for lbl, cnt in zip(*np.unique(y, return_counts=True)):
        print(f"  {lbl:<20}  {cnt:>5}  ({100*cnt/len(y):.1f}%)")

    return X, y, groups


# ---------------------------------------------------------------------------
# Motion-filtered variant: loads pre-segmented MATLAB Excel output
# ---------------------------------------------------------------------------

def _load_mf_session(xlsx_path) -> pd.DataFrame:
    """
    Load a motion-filtered raw Excel file produced by the MATLAB pipeline.

    Unlike the plain raw CSVs, these files already contain Block_ID,
    SubWindow_ID, and other segmentation metadata added by MATLAB.
    Time_PPG is in seconds (not milliseconds).  Derived channels are
    added here identically to load_raw_session().
    """
    xl = pd.ExcelFile(xlsx_path)
    df = xl.parse(xl.sheet_names[0])
    df = df[["Time_PPG", "LED1", "LED2", "LED3", "Temp",
             "AccX", "AccY", "AccZ",
             "Block_ID", "Block_Start_s", "SubWindow_ID"]].copy()
    df = df.astype({c: float for c in df.columns if c not in ("Block_ID", "SubWindow_ID")})
    df["Block_ID"]    = df["Block_ID"].astype(int)
    df["SubWindow_ID"] = df["SubWindow_ID"].astype(int)
    df["LED3_LED1"] = df["LED3"] - df["LED1"]
    df["LED3_LED2"] = df["LED3"] - df["LED2"]
    return df


def load_raw_mf_cnn_dataset(
    s1_xlsx=None,
    s2_xlsx=None,
    labeled_mf_csv=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Motion-filtered pipeline: pre-segmented MATLAB Excel files -> labeled windows.

    The motion-filtered Excel files contain per-row Block_ID and SubWindow_ID
    metadata.  Windows are extracted by grouping rows on these columns and
    taking the first RAW_WINDOW_SAMPLES rows from each group; groups with
    fewer than RAW_WINDOW_SAMPLES rows are dropped.

    Labels and globally-unique Block_IDs are taken from the combined
    features_labeled_mf.csv, matched by (Session_ID, Block_Start_s).

    Parameters
    ----------
    s1_xlsx, s2_xlsx   : paths to per-session motion-filtered xlsx files
    labeled_mf_csv     : path to features_labeled_mf.csv

    Returns
    -------
    X      : (N, RAW_WINDOW_SAMPLES, len(RAW_CHANNELS))  float32
    y      : (N,)  str   stress labels
    groups : (N,)  int   globally-unique Block_IDs for StratifiedGroupKFold
    """
    s1_xlsx        = s1_xlsx        or RAW_MF_SESSION1_FILE
    s2_xlsx        = s2_xlsx        or RAW_MF_SESSION2_FILE
    labeled_mf_csv = labeled_mf_csv or LABELED_FEATURES_MF_FILE

    labeled_df = pd.read_csv(labeled_mf_csv)

    SENSOR_COLS = ["LED1", "LED2", "LED3", "Temp",
                   "AccX", "AccY", "AccZ", "LED3_LED1", "LED3_LED2"]

    all_X, all_y, all_groups = [], [], []

    for session, xlsx_path in [(1, s1_xlsx), (2, s2_xlsx)]:
        df = _load_mf_session(xlsx_path)

        # Build a Block_Start_s -> (label, global_block_id) lookup from the
        # labeled CSV for this session.
        sess_lab = (
            labeled_df[labeled_df["Session_ID"] == session]
            [["Block_ID", "Block_Start_s", "Label"]]
            .drop_duplicates(subset="Block_ID")
        )
        sess_lab_starts = sess_lab["Block_Start_s"].values

        n_windows = 0
        n_dropped = 0
        n_unmatched = 0

        for (blk_id, sw_id), grp in df.groupby(["Block_ID", "SubWindow_ID"]):
            if len(grp) < RAW_WINDOW_SAMPLES:
                n_dropped += 1
                continue

            window = grp[SENSOR_COLS].values[:RAW_WINDOW_SAMPLES].astype(np.float32)

            # Match this block's start time to the labeled CSV
            blk_start = grp["Block_Start_s"].iloc[0]
            diffs = np.abs(sess_lab_starts - blk_start)
            nearest_idx = int(np.argmin(diffs))
            if diffs[nearest_idx] > 2.0:
                n_unmatched += 1
                continue

            row = sess_lab.iloc[nearest_idx]
            label    = row["Label"]
            global_id = int(row["Block_ID"])

            if label == "Unknown":
                continue

            all_X.append(window)
            all_y.append(label)
            all_groups.append(global_id)
            n_windows += 1

        print(
            f"  Session {session}: {n_windows} windows accepted, "
            f"{n_dropped} dropped (<{RAW_WINDOW_SAMPLES} samples), "
            f"{n_unmatched} unmatched"
        )

    X      = np.stack(all_X,      axis=0)
    y      = np.array(all_y,      dtype=object)
    groups = np.array(all_groups, dtype=int)

    print(f"\nMF CNN dataset: {X.shape[0]} windows  "
          f"({X.shape[1]} timesteps x {X.shape[2]} channels)")
    print("Label distribution:")
    for lbl, cnt in zip(*np.unique(y, return_counts=True)):
        print(f"  {lbl:<20}  {cnt:>5}  ({100*cnt/len(y):.1f}%)")

    return X, y, groups
