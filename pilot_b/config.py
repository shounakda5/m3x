"""
Phase-specific configuration for the Pilot B cohort (5 participants: Shuail,
Lehar, Emma, Aisha, Sophia) -- raw sensor + Qualtrics survey data only, no
MATLAB feature extraction.

Constants shared with the original two-session pipeline (raw-windowing
geometry: RAW_SAMPLING_HZ, RAW_WINDOW_SAMPLES, RAW_CHANNELS,
RAW_GAP_THRESHOLD_MS, etc.) stay in the root ``config.py`` -- they describe
the M3X ring's physical duty cycle, not anything Pilot-B-specific. Only what
is unique to this cohort lives here.
"""

from pathlib import Path

PILOT_B_ROOT      = Path(__file__).parent
PILOT_B_DATA_DIR  = PILOT_B_ROOT / "data"
PILOT_B_RAW_DIR   = PILOT_B_DATA_DIR / "raw"
PILOT_B_EXTRACTED_DIR = PILOT_B_RAW_DIR / "extracted"
PILOT_B_RESULTS_DIR   = PILOT_B_ROOT / "results"

# ---------------------------------------------------------------------------
# Pilot B cohort constants
# ---------------------------------------------------------------------------
PILOT_B_FLATLINE_FRACTION      = 0.10   # burst Acc_Total_STD below this fraction
                                         # of the participant's baseline median
                                         # counts as "ring not worn"
PILOT_B_FLATLINE_MIN_BURSTS    = 2      # min consecutive trailing flatlined
                                         # bursts required to crop the tail
PILOT_B_MIN_BURST_S            = 4.0    # Aisha's device duty-cycles at ~5s ON
                                         # (everyone else is ~10s ON) -- use a
                                         # lower floor than RAW_MIN_BURST_S so
                                         # her single-half bursts aren't dropped
PILOT_B_AISHA_HOLDOUT_FRACTION = 0.20   # fraction of Aisha's bursts held out
                                         # entirely from training
PILOT_B_RANDOM_STATE           = 42
