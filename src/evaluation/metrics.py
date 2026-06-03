"""Reporting and comparison utilities for cross-validation results."""

import numpy as np
import pandas as pd
from pathlib import Path

from config import CLASSES


def print_model_summary(model_name: str, agg: dict) -> None:
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  {model_name}")
    print(sep)
    print(f"  Accuracy    : {agg['accuracy_mean']:.4f}  +/-  {agg['accuracy_std']:.4f}")
    print(f"  F1 Macro    : {agg['f1_macro_mean']:.4f}  +/-  {agg['f1_macro_std']:.4f}")
    print(f"  F1 Weighted : {agg['f1_weighted_mean']:.4f}  +/-  {agg['f1_weighted_std']:.4f}")
    print(f"\nPer-class report (predictions pooled across all folds):")
    print(agg["classification_report"])
    print(f"Confusion matrix  (rows = true label, cols = predicted label)")
    print(f"Classes: {CLASSES}")
    print(agg["confusion_matrix"])


def print_comparison(results: dict) -> None:
    """Print a side-by-side summary table for all evaluated models."""
    print(f"\n{'=' * 62}")
    print("  MODEL COMPARISON")
    print(f"{'=' * 62}")
    header = f"{'Model':<20}  {'Accuracy':>14}  {'F1 Macro':>14}  {'F1 Weighted':>14}"
    print(header)
    print("-" * len(header))
    for name, agg in results.items():
        print(
            f"{name:<20}"
            f"  {agg['accuracy_mean']:.4f} +/- {agg['accuracy_std']:.3f}"
            f"  {agg['f1_macro_mean']:.4f} +/- {agg['f1_macro_std']:.3f}"
            f"  {agg['f1_weighted_mean']:.4f} +/- {agg['f1_weighted_std']:.3f}"
        )
    print()


def save_fold_results(results: dict, output_path: str) -> None:
    """Write per-fold metrics for every model to a CSV for offline analysis."""
    rows = []
    for model_name, agg in results.items():
        for r in agg["fold_results"]:
            rows.append({
                "model":        model_name,
                "fold":         r["fold"],
                "accuracy":     r["accuracy"],
                "f1_macro":     r["f1_macro"],
                "f1_weighted":  r["f1_weighted"],
            })
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Per-fold results saved : {output_path}")


def save_feature_importances(importances: pd.Series, output_path: str) -> None:
    """Write RF feature importances to CSV, sorted descending."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    importances.to_csv(output_path, header=["importance"])
    print(f"Feature importances saved : {output_path}")
