"""
Phase-specific configuration for Pilot Phase C.

Unlike the original pipeline (data/) and the Pilot B cohort (pilot_b/), this
phase's incoming data arrives **already labeled** -- there is no survey/
classify_survey/merge_labels step here. The pipeline is just:

    incoming batch(es) -> validated, combined dataset snapshot -> train -> export

Everything specific to this phase (paths, class taxonomy, hyperparameters)
lives here rather than in the root config.py, so this phase can evolve
independently as new data arrives week over week.
"""

from pathlib import Path

PHASE_ROOT    = Path(__file__).parent
DATA_DIR      = PHASE_ROOT / "data"
INCOMING_DIR  = DATA_DIR / "incoming"
DATASET_DIR   = DATA_DIR / "dataset"
RESULTS_DIR   = PHASE_ROOT / "results"

# ---------------------------------------------------------------------------
# Label taxonomy
#
# Pre-labeled data means the labels on disk are the source of truth -- fill
# this in to match whatever classes actually show up in the first real batch
# (see README.md "Adding a new data batch"). Left as a placeholder until
# then so nothing here silently assumes the wrong classes.
# ---------------------------------------------------------------------------
CLASSES = []  # TODO: set once the first batch's label scheme is known

# ---------------------------------------------------------------------------
# Cross-validation / training
# ---------------------------------------------------------------------------
N_FOLDS       = 5
RANDOM_STATE  = 42
