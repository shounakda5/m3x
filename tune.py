"""
tune.py — Hyperparameter search for SVM, Random Forest, XGBoost, and 1D CNN.

Uses RandomizedSearchCV with the same StratifiedGroupKFold strategy as
train.py so the group constraint is respected during the search.  Scoring
is f1_macro to account for class imbalance.

XGBoost is tuned via RandomizedSearchCV using integer-encoded labels (required
by XGBClassifier's multiclass objective).  Sample weights are NOT applied
during the search because RandomizedSearchCV cannot route fold-specific weight
arrays; use the final train.py run (which applies balanced sample weights
inside XGBClassifier.fit) for the definitive weighted evaluation.

CNN is tuned via a manual random search because Keras models cannot be
directly wrapped in RandomizedSearchCV without additional scaffolding.

Usage
-----
    python tune.py --data data/features/features_labeled.csv

The best parameters are printed and written to results/best_params.txt.
Paste them into config.py to make them permanent, then re-run train.py.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import loguniform, randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier as XGBBase

from config import N_FOLDS, RESULTS_DIR
from src.data.preprocess import (
    load_labeled_dataset,
    extract_Xy,
    clean_features,
    fit_scaler,
    apply_scaler,
)


SVM_SEARCH_SPACE = {
    "C":      loguniform(1e-2, 1e3),
    "gamma":  loguniform(1e-4, 1e1),
    "kernel": ["rbf", "linear"],
}

RF_SEARCH_SPACE = {
    "n_estimators":      randint(100, 600),
    "max_depth":         [None, 5, 10, 15, 20, 30],
    "min_samples_split": randint(2, 20),
    "min_samples_leaf":  randint(1, 10),
    "max_features":      ["sqrt", "log2", None],
}

XGB_SEARCH_SPACE = {
    "n_estimators":      randint(100, 600),
    "max_depth":         [3, 4, 5, 6, 7, 8],
    "learning_rate":     loguniform(0.01, 0.3),
    "subsample":         [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree":  [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight":  randint(1, 10),
    "gamma":             [0, 0.05, 0.1, 0.2, 0.5, 1.0],
    "reg_alpha":         loguniform(1e-4, 1.0),
    "reg_lambda":        loguniform(0.1, 10.0),
}

# Discrete choices for CNN random search
CNN_SEARCH_SPACE = {
    "filters_1":        [16, 32, 64],
    "filters_2":        [32, 64, 128],
    "kernel_size":      [3, 5],
    "dense_1":          [64, 128, 256],
    "dense_2":          [32, 64, 128],
    "dense_3":          [16, 32, 64],
    "conv_dropout":     [0.2, 0.3, 0.4],
    "dense_dropout":    [0.3, 0.4, 0.5],
    "learning_rate":    [1e-3, 5e-4, 2e-4],
    "batch_size":       [16, 32, 64],
}

N_ITER     = 60
CNN_N_ITER = 15   # fewer CNN trials — each trains a full neural network
CV_SCORE   = "f1_macro"


def search(estimator, space, X, y, groups, label, n_iter=N_ITER):
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS)
    rs = RandomizedSearchCV(
        estimator           = estimator,
        param_distributions = space,
        n_iter              = n_iter,
        cv                  = sgkf,
        scoring             = CV_SCORE,
        random_state        = 42,
        n_jobs              = -1,
        verbose             = 1,
        refit               = False,
    )
    rs.fit(X, y, groups=groups)

    best   = rs.best_params_
    score  = rs.best_score_
    print(f"\n{label} best {CV_SCORE}: {score:.4f}")
    print(f"{label} best params:")
    for k, v in sorted(best.items()):
        print(f"  {k}: {v}")
    return best, score


def search_cnn(X, y, groups, n_iter=CNN_N_ITER):
    """Manual random search for the 1D CNN using StratifiedGroupKFold."""
    from src.models.cnn_model import CNNClassifier

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS)
    rng  = np.random.RandomState(42)

    best_score  = -np.inf
    best_params = None

    print(f"\n  Running {n_iter} random CNN trials (this may take several minutes)...")

    for trial in range(1, n_iter + 1):
        params = {k: rng.choice(v).item() for k, v in CNN_SEARCH_SPACE.items()}

        fold_scores = []
        for train_idx, test_idx in sgkf.split(X, y, groups):
            X_train = X.iloc[train_idx].copy()
            X_test  = X.iloc[test_idx].copy()
            y_train = y.iloc[train_idx]
            y_test  = y.iloc[test_idx]

            scaler, X_train_sc = fit_scaler(X_train)
            X_test_sc = apply_scaler(scaler, X_test)

            model = CNNClassifier(**params)
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)
            fold_scores.append(
                f1_score(y_test, y_pred, average="macro", zero_division=0)
            )

        mean_score = float(np.mean(fold_scores))
        print(
            f"    Trial {trial:>2}/{n_iter}  f1_macro={mean_score:.4f}"
            f"  filters=({params['filters_1']},{params['filters_2']})"
            f"  dense=({params['dense_1']},{params['dense_2']},{params['dense_3']})"
            f"  lr={params['learning_rate']}"
        )

        if mean_score > best_score:
            best_score  = mean_score
            best_params = dict(params)

    print(f"\nCNN_1D best {CV_SCORE}: {best_score:.4f}")
    print("CNN_1D best params:")
    for k, v in sorted(best_params.items()):
        print(f"  {k}: {v}")
    return best_params, best_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",      required=True)
    parser.add_argument("--n-iter",    type=int, default=N_ITER)
    parser.add_argument("--cnn-iter",  type=int, default=CNN_N_ITER,
                        help=f"Number of CNN random-search trials (default: {CNN_N_ITER})")
    parser.add_argument("--skip-cnn",  action="store_true",
                        help="Skip the CNN hyperparameter search")
    args = parser.parse_args()

    n_iter     = args.n_iter
    cnn_n_iter = args.cnn_iter

    df = load_labeled_dataset(args.data)
    df = df[df["Label"] != "Unknown"].reset_index(drop=True)
    X, y, groups, _ = extract_Xy(df)
    X = clean_features(X)

    # SVM requires scaled features
    _, X_scaled = fit_scaler(X)

    print("=" * 60)
    print(f"Searching {n_iter} iterations each, scoring={CV_SCORE}")
    print("=" * 60)

    print("\n--- SVM ---")
    svm_best, svm_score = search(
        SVC(class_weight="balanced", probability=True, random_state=42),
        SVM_SEARCH_SPACE, X_scaled, y, groups, "SVM",
        n_iter=n_iter,
    )

    print("\n--- Random Forest ---")
    rf_best, rf_score = search(
        RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
        RF_SEARCH_SPACE, X, y, groups, "RandomForest",
        n_iter=n_iter,
    )

    # XGBoost requires integer-encoded labels for its multiclass objective.
    y_enc_xgb = LabelEncoder().fit_transform(y)
    print("\n--- XGBoost ---")
    xgb_best, xgb_score = search(
        XGBBase(
            eval_metric="mlogloss",
            verbosity=0,
            random_state=42,
            n_jobs=-1,
        ),
        XGB_SEARCH_SPACE, X, y_enc_xgb, groups, "XGBoost",
        n_iter=n_iter,
    )

    cnn_best  = None
    cnn_score = None
    if not args.skip_cnn:
        print("\n--- 1D CNN ---")
        cnn_best, cnn_score = search_cnn(X, y, groups, n_iter=cnn_n_iter)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out = Path(RESULTS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    report = out / "best_params.txt"
    with open(report, "w") as f:
        f.write(f"Scoring metric : {CV_SCORE}\n")
        f.write(f"N iterations   : {n_iter}\n\n")

        f.write(f"SVM   best {CV_SCORE}: {svm_score:.4f}\n")
        for k, v in sorted(svm_best.items()):
            f.write(f"  {k}: {v}\n")

        f.write(f"\nRF    best {CV_SCORE}: {rf_score:.4f}\n")
        for k, v in sorted(rf_best.items()):
            f.write(f"  {k}: {v}\n")

        f.write(f"\nXGBoost  best {CV_SCORE}: {xgb_score:.4f}\n")
        f.write("  (scored on integer-encoded labels; sample weights applied in train.py)\n")
        for k, v in sorted(xgb_best.items()):
            f.write(f"  {k}: {v}\n")

        if cnn_best is not None:
            f.write(f"\nCNN_1D  best {CV_SCORE}: {cnn_score:.4f}\n")
            f.write(f"  N trials: {cnn_n_iter}\n")
            for k, v in sorted(cnn_best.items()):
                f.write(f"  {k}: {v}\n")

    print(f"\nResults written to {report}")
    print("\nPaste these into config.py SVM_PARAMS / RF_PARAMS / CNN_PARAMS and re-run train.py")


if __name__ == "__main__":
    main()
