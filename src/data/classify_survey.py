"""
Survey EMA Classifier
---------------------
Classifies each survey response row into one of four labels:
  - relax            : Both Opportunity and Pressure are LOW (1-3)
  - eustress         : Opportunity is HIGH (4-6), Pressure is LOW (1-3)
  - distress         : Pressure is HIGH (4-6), Opportunity is LOW (1-3)
  - eustress+distress: Both Opportunity and Pressure are HIGH (4-6)

Supports two EMA export formats:
  - Old format: Opportunity and Pressure cells contain free-text responses
    (e.g. "generally is a source of opportunity") which are mapped to 1-6.
  - New format: Opportunity and Pressure cells already contain numeric scores
    on a 1-6 (or 1-7) integer scale.

Usage:
    python classify_survey.py --input_dir <path_to_csv_folder> [--output_dir <path_to_output_folder>]

If --output_dir is not specified, results are saved alongside each input CSV
with a '_labelled' suffix.
"""

import os
import sys
import argparse
import pandas as pd

# ---------------------------------------------------------------------------
# Scale mapping: text → numeric option (1–6)
# ---------------------------------------------------------------------------
SCALE_MAP = {
    "very definitely is not a source of opportunity": 1,
    "very definitely is not a source of pressure":    1,
    "definitely is not a source of opportunity":      2,
    "definitely is not a source of pressure":         2,
    "generally is not a source of opportunity":       3,
    "generally is not a source of pressure":          3,
    "generally is a source of opportunity":           4,
    "generally is a source of pressure":              4,
    "definitely is a source of opportunity":          5,
    "definitely is a source of pressure":             5,
    "very definitely is a source of opportunity":     6,
    "very definitely is a source of pressure":        6,
}


def text_to_option(value) -> int | None:
    """
    Convert an Opportunity or Pressure cell to its numeric option (1-6).

    Accepts:
      - int / float already in range 1-6  → returned as int directly
      - free-text string matching SCALE_MAP → mapped to 1-6
      - NaN or unrecognised text           → None
    """
    if pd.isna(value):
        return None

    # Numeric already — trust it directly
    try:
        numeric = float(value)
        return int(round(numeric))
    except (TypeError, ValueError):
        pass

    # Free-text path
    normalized = str(value).strip().lower()
    for key, val in SCALE_MAP.items():
        if key in normalized:
            return val
    return None


def classify(opp_option: int | None, pres_option: int | None) -> str:
    """
    Apply the labelling rules:
        Opp LOW  (1-3) + Pres LOW  (1-3)  → relax
        Opp HIGH (4-6) + Pres LOW  (1-3)  → eustress
        Opp LOW  (1-3) + Pres HIGH (4-6)  → distress
        Opp HIGH (4-6) + Pres HIGH (4-6)  → eustress+distress
        Any missing value                  → Unknown
    """
    if opp_option is None or pres_option is None:
        return "Unknown"

    opp_high  = opp_option  >= 4
    pres_high = pres_option >= 4

    if not opp_high and not pres_high:
        return "relax"
    elif opp_high and not pres_high:
        return "eustress"
    elif not opp_high and pres_high:
        return "distress"
    else:
        return "eustress+distress"


def process_file(csv_path: str, output_dir: str | None) -> pd.DataFrame:
    """Read one CSV, label every row, write output, return the labelled DataFrame."""
    # Qualtrics exports have 3 header rows; row 0 = real headers, rows 1-2 = metadata
    df_raw = pd.read_csv(csv_path, header=0, skiprows=[1, 2], dtype=str)

    # Identify the correct column names (robust to slight naming differences)
    col_map = {}
    for col in df_raw.columns:
        cl = col.strip().lower()
        if "startdate" in cl or cl == "startdate" or cl == "start date":
            col_map["StartDate"] = col
        elif "enddate" in cl or cl == "enddate" or cl == "end date":
            col_map["EndDate"] = col
        elif "opportunity" in cl:
            col_map["Opportunity"] = col
        elif "pressure" in cl:
            col_map["Pressure"] = col

    missing = [k for k in ("StartDate", "EndDate", "Opportunity", "Pressure") if k not in col_map]
    if missing:
        print(f"  [WARN] {os.path.basename(csv_path)}: could not find columns {missing}. Skipping.")
        return pd.DataFrame()

    # Extract only the relevant columns
    result = df_raw[[col_map["StartDate"], col_map["EndDate"],
                     col_map["Opportunity"], col_map["Pressure"]]].copy()
    result.columns = ["StartDate", "EndDate", "Opportunity_raw", "Pressure_raw"]

    # Encode options and classify
    result["Opportunity_option"] = result["Opportunity_raw"].apply(text_to_option)
    result["Pressure_option"]    = result["Pressure_raw"].apply(text_to_option)
    result["Label"]              = result.apply(
        lambda r: classify(r["Opportunity_option"], r["Pressure_option"]), axis=1
    )

    # Final output columns
    out = result[["StartDate", "EndDate", "Opportunity_raw", "Pressure_raw",
                  "Opportunity_option", "Pressure_option", "Label"]].copy()
    out.rename(columns={
        "Opportunity_raw":    "Opportunity",
        "Pressure_raw":       "Pressure",
        "Opportunity_option": "Opportunity_score",
        "Pressure_option":    "Pressure_score",
    }, inplace=True)

    # Determine output path
    base_name = os.path.splitext(os.path.basename(csv_path))[0] + "_labelled.csv"
    out_path  = os.path.join(output_dir if output_dir else os.path.dirname(csv_path), base_name)
    out.to_csv(out_path, index=False)

    return out


def summarise(label_series: pd.Series, filename: str):
    counts = label_series.value_counts()
    total  = len(label_series)
    print(f"\n  Label distribution for {filename}:")
    for label, count in counts.items():
        print(f"    {label:<20}  {count:>4}  ({100*count/total:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Classify EMA survey CSVs into stress labels.")
    parser.add_argument("--input_dir",  required=True,  help="Directory containing input CSV files")
    parser.add_argument("--output_dir", required=False, help="Directory to write labelled CSVs (default: same as input)")
    args = parser.parse_args()

    input_dir  = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else None

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    csv_files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(".csv"))
    if not csv_files:
        print(f"No CSV files found in '{input_dir}'. Exiting.")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV file(s) in '{input_dir}'.\n")
    all_labels = []

    for fname in csv_files:
        fpath = os.path.join(input_dir, fname)
        print(f"Processing: {fname}")
        labelled = process_file(fpath, output_dir)
        if labelled.empty:
            continue
        summarise(labelled["Label"], fname)
        all_labels.append(labelled["Label"])

    # Aggregate summary across all files
    if all_labels:
        combined = pd.concat(all_labels, ignore_index=True)
        total    = len(combined)
        print(f"\n{'='*50}")
        print(f"AGGREGATE across all {len(csv_files)} file(s)  —  {total} rows total:")
        for label, count in combined.value_counts().items():
            print(f"  {label:<20}  {count:>4}  ({100*count/total:.1f}%)")
        print(f"{'='*50}")

    print("\nDone. Labelled CSVs written.")


if __name__ == "__main__":
    main()
