"""
Tests for src/data/classify_survey.py

Covers:
  - text_to_option: old text format, new numeric format, NaN, edge values
  - classify: all four label outcomes, Unknown for missing inputs
  - label string values match the plan's taxonomy
  - process_file: end-to-end with a synthetic numeric-format EMA CSV
"""

import os
import tempfile

import pandas as pd
import pytest

from src.data.classify_survey import classify, text_to_option, process_file


# ---------------------------------------------------------------------------
# text_to_option
# ---------------------------------------------------------------------------

class TestTextToOption:
    def test_numeric_int(self):
        assert text_to_option(4) == 4

    def test_numeric_float(self):
        assert text_to_option(3.0) == 3

    def test_numeric_string(self):
        # New EMA exports numeric values but they may arrive as strings from read_csv
        assert text_to_option("5") == 5

    def test_text_low_opportunity(self):
        assert text_to_option("generally is not a source of opportunity") == 3

    def test_text_high_pressure(self):
        assert text_to_option("definitely is a source of pressure") == 5

    def test_nan_returns_none(self):
        assert text_to_option(float("nan")) is None

    def test_pd_na_returns_none(self):
        assert text_to_option(pd.NA) is None

    def test_unrecognised_text_returns_none(self):
        assert text_to_option("some random text") is None

    def test_boundary_low(self):
        assert text_to_option(1) == 1

    def test_boundary_high(self):
        assert text_to_option(6) == 6


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------

class TestClassify:
    def test_relax(self):
        # Both low
        assert classify(1, 1) == "relax"
        assert classify(3, 3) == "relax"

    def test_eustress(self):
        # High opportunity, low pressure
        assert classify(4, 1) == "eustress"
        assert classify(6, 3) == "eustress"

    def test_distress(self):
        # Low opportunity, high pressure
        assert classify(1, 4) == "distress"
        assert classify(3, 6) == "distress"

    def test_eustress_plus_distress(self):
        # Both high
        assert classify(4, 4) == "eustress+distress"
        assert classify(6, 6) == "eustress+distress"

    def test_unknown_when_opp_none(self):
        assert classify(None, 4) == "Unknown"

    def test_unknown_when_pres_none(self):
        assert classify(4, None) == "Unknown"

    def test_unknown_when_both_none(self):
        assert classify(None, None) == "Unknown"

    def test_boundary_3_is_low(self):
        # Score 3 is LOW → opp=3, pres=3 → relax
        assert classify(3, 3) == "relax"

    def test_boundary_4_is_high(self):
        # Score 4 is HIGH → opp=4, pres=4 → eustress+distress
        assert classify(4, 4) == "eustress+distress"


# ---------------------------------------------------------------------------
# Label string values match plan taxonomy
# ---------------------------------------------------------------------------

class TestLabelValues:
    EXPECTED_LABELS = {"relax", "eustress", "distress", "eustress+distress", "Unknown"}

    def test_all_classify_outputs_in_expected_set(self):
        combinations = [
            (1, 1), (3, 3),   # relax
            (4, 1), (6, 3),   # eustress
            (1, 4), (3, 6),   # distress
            (4, 4), (6, 6),   # eustress+distress
            (None, 1), (1, None),  # Unknown
        ]
        for opp, pres in combinations:
            result = classify(opp, pres)
            assert result in self.EXPECTED_LABELS, (
                f"classify({opp}, {pres}) returned '{result}', not in {self.EXPECTED_LABELS}"
            )

    def test_no_old_label_names(self):
        old_labels = {"Relaxed", "Mixed", "Eustress", "Distress"}
        for opp in range(1, 7):
            for pres in range(1, 7):
                result = classify(opp, pres)
                assert result not in old_labels, (
                    f"classify({opp}, {pres}) returned old label '{result}'"
                )


# ---------------------------------------------------------------------------
# process_file — end-to-end with synthetic numeric EMA CSV
# ---------------------------------------------------------------------------

class TestProcessFile:
    def _make_numeric_ema_csv(self, path: str):
        """Write a minimal Qualtrics-style CSV with numeric Opp/Pressure scores."""
        lines = [
            # Row 0: column headers
            "StartDate,EndDate,Status,Opportunity,Pressure",
            # Row 1: Qualtrics metadata (skipped by process_file)
            "Start Date,End Date,Status,Opportunity,Pressure",
            # Row 2: import ID row (skipped)
            "{},{},{},{},{}",
            # Row 3+: actual data
            "6/2/2026 14:30,6/2/2026 14:31,0,4,3",   # eustress
            "6/2/2026 15:04,6/2/2026 15:05,0,2,5",   # distress
            "6/2/2026 15:37,6/2/2026 15:38,0,2,2",   # relax
            "6/2/2026 16:06,6/2/2026 16:07,0,5,4",   # eustress+distress
        ]
        with open(path, "w") as f:
            f.write("\n".join(lines))

    def _make_text_ema_csv(self, path: str):
        """Write a minimal Qualtrics-style CSV with text Opp/Pressure responses."""
        lines = [
            "StartDate,EndDate,Status,Opportunity,Pressure",
            "Start Date,End Date,Status,Opportunity,Pressure",
            "{},{},{},{},{}",
            "2026-06-01 13:30:59,2026-06-01 13:31:37,0,generally is a source of opportunity,generally is not a source of pressure",
            "2026-06-01 14:52:21,2026-06-01 14:52:39,0,generally is not a source of opportunity,definitely is a source of pressure",
        ]
        with open(path, "w") as f:
            f.write("\n".join(lines))

    def test_numeric_format_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "ema_numeric.csv")
            self._make_numeric_ema_csv(csv_path)
            result = process_file(csv_path, tmpdir)

            assert not result.empty
            assert list(result["Label"]) == ["eustress", "distress", "relax", "eustress+distress"]

    def test_numeric_format_scores_are_integers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "ema_numeric.csv")
            self._make_numeric_ema_csv(csv_path)
            result = process_file(csv_path, tmpdir)

            assert result["Opportunity_score"].tolist() == [4, 2, 2, 5]
            assert result["Pressure_score"].tolist()    == [3, 5, 2, 4]

    def test_text_format_still_works(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "ema_text.csv")
            self._make_text_ema_csv(csv_path)
            result = process_file(csv_path, tmpdir)

            assert not result.empty
            assert result["Label"].iloc[0] == "eustress"   # opp=4, pres=3
            assert result["Label"].iloc[1] == "distress"   # opp=3, pres=5

    def test_output_csv_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "ema_numeric.csv")
            self._make_numeric_ema_csv(csv_path)
            process_file(csv_path, tmpdir)

            expected_out = os.path.join(tmpdir, "ema_numeric_labelled.csv")
            assert os.path.exists(expected_out)

    def test_output_has_required_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "ema_numeric.csv")
            self._make_numeric_ema_csv(csv_path)
            result = process_file(csv_path, tmpdir)

            for col in ("StartDate", "EndDate", "Opportunity", "Pressure",
                        "Opportunity_score", "Pressure_score", "Label"):
                assert col in result.columns, f"Missing column: {col}"
