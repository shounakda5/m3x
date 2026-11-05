"""Label modes for Pilot Phase C classification."""

from __future__ import annotations

import numpy as np

from PilotPhaseC.config import (
    BINARY_BAD_FOUR_CLASS,
    BINARY_GOOD_FOUR_CLASS,
    CLASSES,
    CLASSES_BINARY,
)


def training_class_list(label_mode: str) -> list[str]:
    if label_mode == "four_class":
        return list(CLASSES)
    if label_mode == "binary_good_bad":
        return list(CLASSES_BINARY)
    raise ValueError(f"Unknown label_mode: {label_mode!r}")


def map_window_labels(y_four: np.ndarray, label_mode: str) -> np.ndarray:
    y_four = np.asarray(y_four, dtype=object)
    if label_mode == "four_class":
        unknown = set(y_four) - set(CLASSES)
        if unknown:
            raise ValueError(f"Unknown 4-class labels: {unknown}")
        return y_four
    if label_mode == "binary_good_bad":
        out = []
        for lab in y_four:
            if lab in BINARY_GOOD_FOUR_CLASS:
                out.append("good")
            elif lab in BINARY_BAD_FOUR_CLASS:
                out.append("bad")
            else:
                raise ValueError(f"Cannot map label {lab!r} to binary good/bad")
        return np.array(out, dtype=object)
    raise ValueError(f"Unknown label_mode: {label_mode!r}")


def four_class_support(y_four: np.ndarray) -> dict[str, int]:
    return {c: int(np.sum(y_four == c)) for c in CLASSES}
