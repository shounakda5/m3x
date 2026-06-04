"""
train.py — Train SVM, Random Forest, and XGBoost classifiers on the
           engineered feature dataset with stratified-group k-fold CV.

           For the 1D CNN on raw sensor data use train_cnn.py instead.

Usage
-----
    python train.py --data data/features/features_labeled_new.csv
    python train.py --data data/features/features_labeled_new.csv --models svm rf
    python train.py --data data/features/features_labeled_new.csv --models svm rf --output_dir results/baseline

Options
-------
    --data        Path to the labeled feature CSV (required)
    --models      Space-separated subset to train: svm rf xgb (default: all three)
    --n_folds     Number of CV folds (default: value in config.py)
    --output_dir  Directory for result CSVs (default: results/)
"""

import argparse
import sys
from pathlib import Path

import joblib

from config import N_FOLDS, RESULTS_DIR
from src.data.preprocess import load_labeled_dataset, extract_Xy, clean_features, fit_scaler
from src.models.svm_model  import SVMClassifier
from src.models.rf_model   import RFClassifier
from src.models.xgb_model  import XGBClassifier
from src.evaluation.cross_val import run_cross_val, aggregate_results
from src.evaluation.metrics import (
    print_model_summary,
    print_comparison,
    save_fold_results,
    save_feature_importances,
)

_ALL_MODELS = {
    "svm": SVMClassifier,
    "rf":  RFClassifier,
    "xgb": XGBClassifier,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train stress classifiers with stratified-group k-fold CV."
    )
    parser.add_argument("--data",       required=True,
                        help="Path to labeled feature CSV (must contain a 'Label' column)")
    parser.add_argument("--models",     nargs="+", default=list(_ALL_MODELS.keys()),
                        choices=list(_ALL_MODELS.keys()),
                        help="Models to train (default: all). Example: --models svm rf")
    parser.add_argument("--n_folds",    type=int, default=N_FOLDS,
                        help=f"Number of CV folds (default: {N_FOLDS})")
    parser.add_argument("--output_dir", default=str(RESULTS_DIR),
                        help="Directory to write result CSVs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load & validate
    # ------------------------------------------------------------------
    print(f"Loading data from : {args.data}")
    df = load_labeled_dataset(args.data)

    unknown_mask = df["Label"] == "Unknown"
    if unknown_mask.any():
        print(f"Dropping {unknown_mask.sum()} rows with 'Unknown' label.")
        df = df[~unknown_mask].reset_index(drop=True)

    X, y, groups, feature_cols = extract_Xy(df)
    X = clean_features(X)

    print(f"\nDataset summary")
    print(f"  Rows         : {len(df)}")
    print(f"  Features     : {len(feature_cols)}")
    print(f"  Classes      : {sorted(y.unique())}")
    print(f"  Groups (10s windows) : {groups.nunique() if groups is not None else 'N/A'}")
    print(f"\nClass distribution:")
    for lbl, cnt in y.value_counts().items():
        print(f"  {lbl:<12}  {cnt:>5}  ({100 * cnt / len(y):.1f}%)")

    if groups is None:
        print(
            "\n[WARN] Window_10s column not found — falling back to non-grouped folds.\n"
            "       Sub-window leakage between train and test is possible."
        )

    print(f"\nRunning {args.n_folds}-fold StratifiedGroupKFold cross-validation\n")

    # ------------------------------------------------------------------
    # 2. Train and evaluate each model
    # ------------------------------------------------------------------
    models      = [_ALL_MODELS[name]() for name in args.models]
    all_results = {}

    for model in models:
        print("-" * 40)
        print(f"Model : {model.name}")
        fold_results            = run_cross_val(model, X, y, groups, n_folds=args.n_folds)
        agg                     = aggregate_results(fold_results)
        all_results[model.name] = agg
        print_model_summary(model.name, agg)

        # Retrain on full dataset → serialize model + scaler
        X_full = X.copy()
        scaler = None
        if model.requires_scaling():
            scaler, X_full = fit_scaler(X_full)
        model.fit(X_full, y)

        model_name_lower = model.name.lower()
        if hasattr(model, "_model"):
            joblib.dump(model._model, output_dir / f"{model_name_lower}_model.pkl")
        if scaler is not None:
            joblib.dump(scaler, output_dir / f"{model_name_lower}_scaler.pkl")

        # RF: also save feature importances
        if hasattr(model, "feature_importances"):
            importances = model.feature_importances(feature_cols)
            save_feature_importances(
                importances,
                str(output_dir / f"{model.name}_feature_importances.csv"),
            )

    # ------------------------------------------------------------------
    # 3. Side-by-side comparison
    # ------------------------------------------------------------------
    print_comparison(all_results)

    # ------------------------------------------------------------------
    # 4. Persist results
    # ------------------------------------------------------------------
    save_fold_results(all_results, str(output_dir / "fold_results.csv"))
    print("\nDone.")


if __name__ == "__main__":
    main()
