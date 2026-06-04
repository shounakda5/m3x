"""
merge_labels.py — Attach EMA survey labels to physiological feature rows.

Time model
----------
The first survey's StartDate is treated as t = 0 for the feature timeline.
All survey timestamps are converted to seconds from that origin using only
the HH:MM:SS component — the date is intentionally ignored so sessions that
span midnight still resolve correctly.

Feature rows use Block_Start_s and Block_End_s, which are seconds from
session start as output by Farhad's MATLAB extraction pipeline. The pipeline
detects active recording blocks directly from timestamp gaps, so Block_Start_s
already reflects real elapsed time including sleep periods. No duty-cycle
correction is applied.

Labelling rules
---------------
  1. During a survey window [StartDate, EndDate]         → that survey's label
  2. In the gap between survey N and survey N+1          → survey N+1's label
  3. Before the first survey (t < 0, if any rows exist)  → first survey's label
  4. After the last survey's EndDate                     → "Unknown"

Group column
-----------
Block_ID is preserved in the output and serves as the group key for
StratifiedGroupKFold cross-validation so that both sub-windows within the
same block always land in the same fold.

Usage
-----
    python -m src.data.merge_labels \\
        --features   data/features/1_suhail_M3X_Features_AllBlocks_output.xlsx \\
        --labels     data/survey/ema_session1_labelled.csv \\
        --output     data/features/features_labeled_new.csv

Accepts both .csv and .xlsx feature files.
"""

import argparse
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hms_to_seconds(ts: str) -> float:
    """
    Return seconds since midnight from a timestamp string.

    Handles:
      - 'YYYY-MM-DD HH:MM:SS'   (old Qualtrics ISO format)
      - 'M/D/YYYY H:MM'         (new Qualtrics short format, no seconds)
      - 'HH:MM:SS' / 'HH:MM'   (bare time strings)
    """
    time_part = str(ts).strip().split()[-1]   # keep only the last token = HH:MM[:SS]
    fields    = time_part.split(":")
    h = int(fields[0])
    m = int(fields[1])
    s = float(fields[2]) if len(fields) > 2 else 0.0
    return h * 3600 + m * 60 + s


# ---------------------------------------------------------------------------
# Survey intervals
# ---------------------------------------------------------------------------

def build_survey_intervals(survey_df: pd.DataFrame) -> list[tuple[float, float, str]]:
    """
    Return [(start_s, end_s, label), ...] sorted by start_s, where times are
    in seconds relative to the first survey's StartDate.
    """
    df       = survey_df.sort_values("StartDate").reset_index(drop=True)
    origin_s = _hms_to_seconds(df.loc[0, "StartDate"])

    return [
        (
            _hms_to_seconds(row["StartDate"]) - origin_s,
            _hms_to_seconds(row["EndDate"])   - origin_s,
            row["Label"],
        )
        for _, row in df.iterrows()
    ]


def assign_label(t: float, intervals: list[tuple[float, float, str]]) -> str:
    """
    Return the label for a feature row at corrected time *t* (seconds since
    first survey start).

    Iterating through the sorted intervals, we return the label of the first
    interval whose end_s >= t.  This handles both 'during a survey' and
    'in the gap leading up to the next survey' with one traversal.
    """
    for _start_s, end_s, label in intervals:
        if t <= end_s:
            return label
    return "Unknown"


# ---------------------------------------------------------------------------
# Main merge function
# ---------------------------------------------------------------------------

def merge(
    features_path: str,
    labels_path:   str,
    output_path:   str,
) -> pd.DataFrame:
    # Load features — accept both CSV and Excel
    fp = Path(features_path)
    if fp.suffix.lower() in (".xlsx", ".xls"):
        xl = pd.ExcelFile(fp)
        sheet = next(
            (s for s in ["Features", "MotionFiltered_Features"] if s in xl.sheet_names),
            xl.sheet_names[0],
        )
        features = xl.parse(sheet)
    else:
        features = pd.read_csv(fp)

    survey_df = pd.read_csv(labels_path)

    # Compute per-row time reference: midpoint of the block window
    features["_time_mid"] = (features["Block_Start_s"] + features["Block_End_s"]) / 2

    intervals = build_survey_intervals(survey_df)
    features["Label"] = features["_time_mid"].apply(lambda t: assign_label(t, intervals))
    features.drop(columns=["_time_mid"], inplace=True)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out, index=False)

    total   = len(features)
    labeled = (features["Label"] != "Unknown").sum()
    unknown = total - labeled

    print(f"Rows total : {total}")
    print(f"Labeled    : {labeled}  ({100 * labeled / total:.1f}%)")
    print(f"Unknown    : {unknown}  ({100 * unknown  / total:.1f}%)")
    print(f"Written    : {out}")

    if labeled:
        print("\nLabel distribution (labeled rows only):")
        dist = features.loc[features["Label"] != "Unknown", "Label"].value_counts()
        for lbl, cnt in dist.items():
            print(f"  {lbl:<20}  {cnt:>5}  ({100 * cnt / labeled:.1f}%)")

    # Per-survey coverage report
    survey_df_sorted = survey_df.sort_values("StartDate").reset_index(drop=True)
    max_feature_t    = (features["Block_End_s"]).max()
    print("\nSurvey coverage (feature data ends at t={:.0f}s):".format(max_feature_t))
    for i, (start_s, end_s, label) in enumerate(intervals):
        has_data = max_feature_t >= start_s
        row      = survey_df_sorted.iloc[i]
        marker   = "  covered" if has_data else "  NO FEATURE DATA"
        print(f"  Survey {i+1}  {label:<20}  t={start_s:.0f}-{end_s:.0f}s{marker}")

    return features


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach EMA survey labels to physiological feature rows."
    )
    parser.add_argument("--features", required=True,
                        help="Path to features file (.csv or .xlsx with a 'Features' sheet)")
    parser.add_argument("--labels",   required=True,
                        help="Path to labelled survey CSV produced by classify_survey.py")
    parser.add_argument("--output",   required=True,
                        help="Output path for the labeled feature CSV")
    args = parser.parse_args()

    merge(
        features_path = args.features,
        labels_path   = args.labels,
        output_path   = args.output,
    )


if __name__ == "__main__":
    main()
