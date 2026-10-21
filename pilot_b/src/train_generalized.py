"""
train_generalized.py -- Train a generalized 1D CNN on the Pilot B cohort
(5 participants: Shuail/501, Lehar/502, Emma/503, Aisha/504, Sophia/505).

Pilot B has no MATLAB feature extraction -- only raw sensor CSVs and Qualtrics
EMA exports -- so the raw 1D CNN pipeline (same architecture as
results/cnn_raw_baseline, the best model from the original two sessions) is
the only applicable model.

Two validation threads are produced from a single trained model:

  1. General cross-participant validation -- 5-fold StratifiedGroupKFold
     over a training-eligible pool that has been uniformly sampled to an
     equal total burst budget per participant (each participant's natural
     class mix is preserved within that budget, so no single person
     dominates by data volume).
  2. Aisha-specific validation -- a slice of Aisha's data is held out
     *before* balancing/training and never seen by the model; the final
     model is scored on it separately.

Usage
-----
    python pilot_b/src/train_generalized.py
    python pilot_b/src/train_generalized.py --output_dir pilot_b/results/cnn_generalized

Outputs (in --output_dir)
-------------------------
    dataset_summary.csv            per participant-day parsing/labeling diagnostics
    fold_results.csv                general-validation per-fold accuracy/F1
    general_validation_report.txt   pooled classification report + confusion matrix
    aisha_holdout_report.txt         Aisha-only held-out classification report
    cnn_1d_model.pkl / .tflite       final model (trained on the full balanced pool)
    cnn_1d_scaler.npz                per-channel mean/std used by the final model
    PILOT_B_SUMMARY.md               consolidated human-readable report
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import CLASSES, N_FOLDS
from pilot_b.config import (
    PILOT_B_RESULTS_DIR,
    PILOT_B_AISHA_HOLDOUT_FRACTION,
    PILOT_B_RANDOM_STATE,
)
from pilot_b.src.loader import load_pilot_b_dataset
from src.models.cnn_model import CNNClassifier
from train_cnn import (
    run_raw_cross_val,
    _fit_channel_scaler,
    _apply_channel_scaler,
    _print_summary,
    _save_fold_results,
    _export_tflite,
)

AISHA_PID = 504
PARTICIPANT_NAMES = {501: "Shuail", 502: "Lehar", 503: "Emma", 504: "Aisha", 505: "Sophia"}


# ---------------------------------------------------------------------------
# Group-level table helper
# ---------------------------------------------------------------------------

def _group_table(y: np.ndarray, groups: np.ndarray, participants: np.ndarray) -> pd.DataFrame:
    """One row per unique burst-group with its label and participant."""
    df = pd.DataFrame({"group": groups, "label": y, "participant": participants})
    return df.drop_duplicates(subset="group").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 1: hold out a slice of Aisha's data, untouched by training
# ---------------------------------------------------------------------------

def split_aisha_holdout(y, groups, participants, rng: np.random.RandomState):
    gt = _group_table(y, groups, participants)
    aisha = gt[gt["participant"] == AISHA_PID]

    holdout_groups = set()
    for label, sub in aisha.groupby("label"):
        n_hold = max(1, round(len(sub) * PILOT_B_AISHA_HOLDOUT_FRACTION))
        n_hold = min(n_hold, len(sub))
        chosen = rng.choice(sub["group"].values, size=n_hold, replace=False)
        holdout_groups.update(chosen)

    is_holdout = np.array([g in holdout_groups for g in groups])
    return is_holdout, holdout_groups


# ---------------------------------------------------------------------------
# Step 2: uniform sampling -- equal total bursts per participant
#
# Caps every participant's *total* burst budget to the smallest participant's
# total, but lets each participant's own natural class mix fill that budget
# (simple random sample of their bursts, not stratified per class).
#
# An earlier version capped per (participant, class) cell to the scarcest
# participant in that class -- e.g. Aisha's 305 available eustress bursts
# got capped to Emma's 5, because the floor is taken across participants
# per class. That discarded 98% of Aisha's real eustress data and was the
# single biggest driver of her poor held-out transfer. Capping by
# participant total instead keeps that real (non-duplicated) data; the
# trade-off is that cross-participant class balance is no longer exact.
# ---------------------------------------------------------------------------

def uniform_sample_by_participant_total(y, groups, participants, rng: np.random.RandomState):
    gt = _group_table(y, groups, participants)
    totals = gt.groupby("participant").size()
    n = int(totals.min())

    selected_groups = set()
    for participant in totals.index:
        candidates = gt[gt["participant"] == participant]["group"].values
        chosen = rng.choice(candidates, size=n, replace=False)
        selected_groups.update(chosen)

    mask = np.array([g in selected_groups for g in groups])

    sampled = gt[gt["group"].isin(selected_groups)]
    sampling_report = (
        sampled.groupby(["participant", "label"]).size().rename("sampled_groups").reset_index()
    )
    available = gt.groupby(["participant", "label"]).size().rename("available_groups").reset_index()
    sampling_report = available.merge(sampling_report, on=["participant", "label"], how="left")
    sampling_report["sampled_groups"] = sampling_report["sampled_groups"].fillna(0).astype(int)
    sampling_report["participant_name"] = sampling_report["participant"].map(PARTICIPANT_NAMES)
    sampling_report = sampling_report[["participant", "participant_name", "label", "available_groups", "sampled_groups"]]

    return mask, sampling_report


# ---------------------------------------------------------------------------
# Step 3: reporting helpers
# ---------------------------------------------------------------------------

def _classification_report_str(y_true, y_pred) -> str:
    lines = []
    lines.append(f"Accuracy    : {accuracy_score(y_true, y_pred):.4f}")
    lines.append(f"F1 Macro    : {f1_score(y_true, y_pred, average='macro', labels=CLASSES, zero_division=0):.4f}")
    lines.append(f"F1 Weighted : {f1_score(y_true, y_pred, average='weighted', labels=CLASSES, zero_division=0):.4f}")
    lines.append("")
    lines.append(classification_report(y_true, y_pred, labels=CLASSES, zero_division=0))
    lines.append(f"Confusion matrix (rows=true, cols=predicted)  classes={CLASSES}")
    lines.append(str(confusion_matrix(y_true, y_pred, labels=CLASSES)))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Train generalized 1D CNN on Pilot B raw data.")
    parser.add_argument("--output_dir", default=str(PILOT_B_RESULTS_DIR / "cnn_generalized"))
    parser.add_argument("--n_folds", type=int, default=N_FOLDS)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(PILOT_B_RANDOM_STATE)

    # ------------------------------------------------------------------
    # 1. Load full dataset
    # ------------------------------------------------------------------
    print("Loading Pilot B raw sensor data ...")
    X, y, groups, participants, day_summary = load_pilot_b_dataset()
    pd.DataFrame(day_summary).to_csv(out / "dataset_summary.csv", index=False)
    print(f"Full dataset: {X.shape[0]} windows across {len(np.unique(groups))} bursts, "
          f"{len(np.unique(participants))} participants")

    # ------------------------------------------------------------------
    # 2. Hold out a slice of Aisha's data (never trained on)
    # ------------------------------------------------------------------
    is_holdout, holdout_groups = split_aisha_holdout(y, groups, participants, rng)
    X_aisha_holdout, y_aisha_holdout = X[is_holdout], y[is_holdout]
    print(f"\nAisha holdout: {is_holdout.sum()} windows across {len(holdout_groups)} bursts "
          f"(held out permanently from training)")

    pool_mask = ~is_holdout
    X_pool, y_pool = X[pool_mask], y[pool_mask]
    groups_pool, participants_pool = groups[pool_mask], participants[pool_mask]

    # ------------------------------------------------------------------
    # 3. Uniform sampling -- equal total bursts per participant
    # ------------------------------------------------------------------
    balance_mask, sampling_report = uniform_sample_by_participant_total(
        y_pool, groups_pool, participants_pool, rng
    )
    sampling_report.to_csv(out / "sampling_report.csv", index=False)

    X_train_pool = X_pool[balance_mask]
    y_train_pool = y_pool[balance_mask]
    groups_train_pool = groups_pool[balance_mask]

    print(f"\nTraining-eligible pool after per-participant balancing (equal total bursts, "
          f"natural class mix preserved): "
          f"{X_train_pool.shape[0]} windows across {len(np.unique(groups_train_pool))} bursts "
          f"(down from {X_pool.shape[0]} windows / {len(np.unique(groups_pool))} bursts pre-balancing)")
    print("\nClass distribution in balanced training pool:")
    lbl, cnt = np.unique(y_train_pool, return_counts=True)
    for l, c in zip(lbl, cnt):
        print(f"  {l:<20}  {c:>5}")

    assert not (set(groups_train_pool) & holdout_groups), \
        "Aisha holdout groups leaked into the training pool"

    # ------------------------------------------------------------------
    # 4. Thread 1 -- general cross-participant validation (grouped CV)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 62}\nThread 1: General cross-participant validation "
          f"({args.n_folds}-fold StratifiedGroupKFold)\n{'=' * 62}")
    fold_results = run_raw_cross_val(X_train_pool, y_train_pool, groups_train_pool, n_folds=args.n_folds)
    cv_summary = _print_summary(fold_results)
    _save_fold_results(fold_results, out / "fold_results.csv")

    all_y_test = np.concatenate([r["y_test"] for r in fold_results])
    all_y_pred = np.concatenate([r["y_pred"] for r in fold_results])
    (out / "general_validation_report.txt").write_text(
        _classification_report_str(all_y_test, all_y_pred), encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # 5. Retrain final model on the full balanced pool
    # ------------------------------------------------------------------
    print("\nRetraining final model on the full balanced pool ...")
    mean, std = _fit_channel_scaler(X_train_pool)
    X_train_scaled = _apply_channel_scaler(X_train_pool, mean, std)

    final_model = CNNClassifier()
    final_model.fit(X_train_scaled, y_train_pool)

    joblib.dump(final_model._model, out / "cnn_1d_model.pkl")
    np.savez(out / "cnn_1d_scaler.npz", mean=mean, std=std)
    _export_tflite(final_model._model, out / "cnn_1d_model.tflite")

    # ------------------------------------------------------------------
    # 6. Thread 2 -- Aisha-specific held-out validation
    # ------------------------------------------------------------------
    print(f"\n{'=' * 62}\nThread 2: Aisha-specific held-out validation\n{'=' * 62}")
    X_aisha_scaled = _apply_channel_scaler(X_aisha_holdout, mean, std)
    y_aisha_pred = final_model.predict(X_aisha_scaled)

    aisha_report = _classification_report_str(y_aisha_holdout, y_aisha_pred)
    print(aisha_report)
    (out / "aisha_holdout_report.txt").write_text(aisha_report, encoding="utf-8")

    aisha_acc = accuracy_score(y_aisha_holdout, y_aisha_pred)
    aisha_f1m = f1_score(y_aisha_holdout, y_aisha_pred, average="macro", labels=CLASSES, zero_division=0)

    # ------------------------------------------------------------------
    # 7. Consolidated summary
    # ------------------------------------------------------------------
    _write_summary_md(
        out, day_summary, sampling_report,
        X, X_pool, X_train_pool, X_aisha_holdout,
        cv_summary, aisha_acc, aisha_f1m,
    )
    print(f"\nAll results written to {out}/")


def _write_summary_md(out, day_summary, sampling_report,
                       X_full, X_pool, X_train_pool, X_aisha_holdout,
                       cv_summary, aisha_acc, aisha_f1m):
    lines = []
    lines.append("# Pilot B -- Generalized CNN: Consolidated Results\n")

    lines.append("## Dataset composition (per participant-day)\n")
    lines.append(pd.DataFrame(day_summary).to_markdown(index=False))
    lines.append("")

    lines.append("## Dataset funnel\n")
    lines.append(f"- Full labeled dataset: **{X_full.shape[0]}** windows")
    lines.append(f"- After removing Aisha's held-out slice: **{X_pool.shape[0]}** windows")
    lines.append(f"- After per-participant balancing, equal total budget (final training pool): **{X_train_pool.shape[0]}** windows")
    lines.append(f"- Aisha held-out validation set: **{X_aisha_holdout.shape[0]}** windows\n")

    lines.append("## Per-participant sampling detail (equal total budget, natural class mix preserved)\n")
    lines.append(sampling_report.to_markdown(index=False))
    lines.append("")

    lines.append("## Thread 1 -- General cross-participant validation (5-fold grouped CV)\n")
    lines.append(f"- Accuracy    : {cv_summary['accuracy_mean']:.4f} +/- {cv_summary['accuracy_std']:.4f}")
    lines.append(f"- F1 Macro    : {cv_summary['f1_macro_mean']:.4f} +/- {cv_summary['f1_macro_std']:.4f}")
    lines.append(f"- F1 Weighted : {cv_summary['f1_weighted_mean']:.4f} +/- {cv_summary['f1_weighted_std']:.4f}")
    lines.append("\nSee `general_validation_report.txt` for the full per-class report and confusion matrix.\n")

    lines.append("## Thread 2 -- Aisha-specific held-out validation\n")
    lines.append(f"- Accuracy : {aisha_acc:.4f}")
    lines.append(f"- F1 Macro : {aisha_f1m:.4f}")
    lines.append("\nSee `aisha_holdout_report.txt` for the full per-class report and confusion matrix.\n")

    lines.append("## Reference -- original 2-session CNN baseline\n")
    lines.append("- Accuracy 80.6% +/- 2.4%, F1 Macro 0.801 +/- 0.023 (`results/cnn_raw_baseline`)\n")

    (out / "PILOT_B_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
