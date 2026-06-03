"""
Tests for src/data/merge_labels.py

Covers:
  - _hms_to_seconds: old ISO format, new M/D/YYYY H:MM format, bare HH:MM
  - build_survey_intervals: correct relative times and label assignment
  - assign_label: during window, between windows, before first, after last
  - merge (integration): correct column names, Label column populated,
    Block_ID preserved in output, both CSV and xlsx feature inputs
"""

import os
import tempfile

import pandas as pd
import pytest

from src.data.merge_labels import (
    _hms_to_seconds,
    assign_label,
    build_survey_intervals,
    merge,
)


# ---------------------------------------------------------------------------
# _hms_to_seconds
# ---------------------------------------------------------------------------

class TestHmsToSeconds:
    def test_iso_format_with_seconds(self):
        # "YYYY-MM-DD HH:MM:SS"
        assert _hms_to_seconds("2026-06-01 13:30:59") == pytest.approx(13 * 3600 + 30 * 60 + 59)

    def test_new_format_no_seconds(self):
        # "M/D/YYYY H:MM"
        assert _hms_to_seconds("6/2/2026 14:30") == pytest.approx(14 * 3600 + 30 * 60)

    def test_bare_hhmm(self):
        assert _hms_to_seconds("09:05") == pytest.approx(9 * 3600 + 5 * 60)

    def test_bare_hhmmss(self):
        assert _hms_to_seconds("09:05:30") == pytest.approx(9 * 3600 + 5 * 60 + 30)

    def test_midnight(self):
        assert _hms_to_seconds("00:00:00") == pytest.approx(0.0)

    def test_end_of_day(self):
        assert _hms_to_seconds("23:59:59") == pytest.approx(23 * 3600 + 59 * 60 + 59)

    def test_fractional_seconds(self):
        # Some Qualtrics exports include sub-second precision
        assert _hms_to_seconds("2026-06-01 13:30:59.5") == pytest.approx(13 * 3600 + 30 * 60 + 59.5)


# ---------------------------------------------------------------------------
# build_survey_intervals
# ---------------------------------------------------------------------------

class TestBuildSurveyIntervals:
    def _make_survey_df(self, rows):
        return pd.DataFrame(rows, columns=["StartDate", "EndDate", "Label"])

    def test_origin_at_zero(self):
        df = self._make_survey_df([
            ("2026-06-01 14:00:00", "2026-06-01 14:00:30", "relax"),
        ])
        intervals = build_survey_intervals(df)
        start_s, end_s, label = intervals[0]
        assert start_s == pytest.approx(0.0)
        assert end_s   == pytest.approx(30.0)
        assert label   == "relax"

    def test_second_interval_relative(self):
        df = self._make_survey_df([
            ("2026-06-01 14:00:00", "2026-06-01 14:00:30", "relax"),
            ("2026-06-01 14:30:00", "2026-06-01 14:30:20", "eustress"),
        ])
        intervals = build_survey_intervals(df)
        assert intervals[1][0] == pytest.approx(30 * 60)       # 30 min after origin
        assert intervals[1][1] == pytest.approx(30 * 60 + 20)

    def test_new_format_dates(self):
        df = self._make_survey_df([
            ("6/2/2026 14:30", "6/2/2026 14:30", "eustress"),
            ("6/2/2026 15:04", "6/2/2026 15:04", "distress"),
        ])
        intervals = build_survey_intervals(df)
        assert intervals[0][0] == pytest.approx(0.0)
        assert intervals[1][0] == pytest.approx(34 * 60)   # 34 minutes later

    def test_sorted_by_start(self):
        # Intentionally out of order
        df = self._make_survey_df([
            ("6/2/2026 15:04", "6/2/2026 15:04", "distress"),
            ("6/2/2026 14:30", "6/2/2026 14:30", "eustress"),
        ])
        intervals = build_survey_intervals(df)
        assert intervals[0][2] == "eustress"   # earlier survey comes first
        assert intervals[1][2] == "distress"


# ---------------------------------------------------------------------------
# assign_label
# ---------------------------------------------------------------------------

class TestAssignLabel:
    @pytest.fixture
    def intervals(self):
        # Three surveys with 30s gaps between them
        # Survey 1: t=0-30   → relax
        # Survey 2: t=60-90  → eustress
        # Survey 3: t=120-150 → distress
        return [
            (0.0,   30.0,  "relax"),
            (60.0,  90.0,  "eustress"),
            (120.0, 150.0, "distress"),
        ]

    def test_during_first_survey(self, intervals):
        assert assign_label(15.0, intervals) == "relax"

    def test_at_end_of_first_survey(self, intervals):
        assert assign_label(30.0, intervals) == "relax"

    def test_gap_between_survey1_and_2(self, intervals):
        # t=45s is after survey1 ends (30s) but before survey2 ends (90s) → eustress
        assert assign_label(45.0, intervals) == "eustress"

    def test_during_second_survey(self, intervals):
        assert assign_label(75.0, intervals) == "eustress"

    def test_gap_between_survey2_and_3(self, intervals):
        # t=100s → before survey3 ends (150s) → distress
        assert assign_label(100.0, intervals) == "distress"

    def test_at_end_of_last_survey(self, intervals):
        assert assign_label(150.0, intervals) == "distress"

    def test_after_last_survey(self, intervals):
        assert assign_label(151.0, intervals) == "Unknown"

    def test_before_first_survey(self, intervals):
        # Negative time (sensor started before first survey) → first survey's label
        assert assign_label(-5.0, intervals) == "relax"


# ---------------------------------------------------------------------------
# merge — integration test
# ---------------------------------------------------------------------------

class TestMerge:
    def _make_labelled_survey_csv(self, path: str):
        """Write a minimal labelled survey CSV (output of classify_survey.py)."""
        rows = [
            ("6/2/2026 14:30", "6/2/2026 14:30", "4", "3", "4", "3", "eustress"),
            ("6/2/2026 15:04", "6/2/2026 15:04", "2", "5", "2", "5", "distress"),
            ("6/2/2026 15:37", "6/2/2026 15:37", "2", "2", "2", "2", "relax"),
        ]
        df = pd.DataFrame(rows, columns=[
            "StartDate", "EndDate", "Opportunity", "Pressure",
            "Opportunity_score", "Pressure_score", "Label",
        ])
        df.to_csv(path, index=False)

    def _make_feature_csv(self, path: str):
        """
        Write a minimal feature CSV in the new schema.
        Origin time ~= first survey at 14:30 → Block_Start_s ~ 0.
        """
        rows = [
            # Block 1, subwindows 1+2 — near session start
            (1, 1, 31.2,  41.0,  9.8,  24.4),
            (1, 2, 31.2,  41.0,  9.8,  24.4),
            # Block 2 — ~30 min in, still covered by survey 1
            (2, 1, 1830.0, 1840.0, 10.0, 24.3),
            (2, 2, 1830.0, 1840.0, 10.0, 24.3),
            # Block 3 — ~90 min in, past all surveys → Unknown
            (3, 1, 9000.0, 9010.0, 10.0, 24.5),
            (3, 2, 9000.0, 9010.0, 10.0, 24.5),
        ]
        df = pd.DataFrame(rows, columns=[
            "Block_ID", "SubWindow_ID",
            "Block_Start_s", "Block_End_s", "Block_Duration_s",
            "mean_temperature",
        ])
        df.to_csv(path, index=False)

    def test_label_column_added(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            feat_path   = os.path.join(tmpdir, "features.csv")
            survey_path = os.path.join(tmpdir, "survey_labelled.csv")
            out_path    = os.path.join(tmpdir, "features_labeled.csv")

            self._make_feature_csv(feat_path)
            self._make_labelled_survey_csv(survey_path)

            result = merge(feat_path, survey_path, out_path)
            assert "Label" in result.columns

    def test_block_id_preserved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            feat_path   = os.path.join(tmpdir, "features.csv")
            survey_path = os.path.join(tmpdir, "survey_labelled.csv")
            out_path    = os.path.join(tmpdir, "features_labeled.csv")

            self._make_feature_csv(feat_path)
            self._make_labelled_survey_csv(survey_path)

            result = merge(feat_path, survey_path, out_path)
            assert "Block_ID" in result.columns
            assert set(result["Block_ID"].unique()) == {1, 2, 3}

    def test_time_mid_not_in_output(self):
        # The internal _time_mid scratch column must be dropped before writing
        with tempfile.TemporaryDirectory() as tmpdir:
            feat_path   = os.path.join(tmpdir, "features.csv")
            survey_path = os.path.join(tmpdir, "survey_labelled.csv")
            out_path    = os.path.join(tmpdir, "features_labeled.csv")

            self._make_feature_csv(feat_path)
            self._make_labelled_survey_csv(survey_path)

            result = merge(feat_path, survey_path, out_path)
            assert "_time_mid" not in result.columns

    def test_blocks_near_origin_get_second_label(self):
        # New EMA has StartDate == EndDate (instantaneous surveys), so Survey 1's
        # window is 0 s wide and its label only applies at exactly t=0.
        # Block 1 midpoint (t≈36s) falls in the gap after Survey 1 ends (t=0)
        # and before Survey 2 ends (t=2040s), so it gets Survey 2's label
        # per the "gap → next survey's label" rule.
        with tempfile.TemporaryDirectory() as tmpdir:
            feat_path   = os.path.join(tmpdir, "features.csv")
            survey_path = os.path.join(tmpdir, "survey_labelled.csv")
            out_path    = os.path.join(tmpdir, "features_labeled.csv")

            self._make_feature_csv(feat_path)
            self._make_labelled_survey_csv(survey_path)

            result = merge(feat_path, survey_path, out_path)
            block1 = result[result["Block_ID"] == 1]["Label"].unique()
            assert len(block1) == 1
            assert block1[0] == "distress"

    def test_blocks_past_all_surveys_are_unknown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            feat_path   = os.path.join(tmpdir, "features.csv")
            survey_path = os.path.join(tmpdir, "survey_labelled.csv")
            out_path    = os.path.join(tmpdir, "features_labeled.csv")

            self._make_feature_csv(feat_path)
            self._make_labelled_survey_csv(survey_path)

            result = merge(feat_path, survey_path, out_path)
            # Block 3 at t≈9005s = ~150 min >> last survey at 67 min → Unknown
            block3_labels = result[result["Block_ID"] == 3]["Label"].unique()
            assert list(block3_labels) == ["Unknown"]

    def test_output_csv_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            feat_path   = os.path.join(tmpdir, "features.csv")
            survey_path = os.path.join(tmpdir, "survey_labelled.csv")
            out_path    = os.path.join(tmpdir, "features_labeled.csv")

            self._make_feature_csv(feat_path)
            self._make_labelled_survey_csv(survey_path)
            merge(feat_path, survey_path, out_path)

            assert os.path.exists(out_path)
            saved = pd.read_csv(out_path)
            assert "Label" in saved.columns

    def test_xlsx_feature_file_accepted(self):
        pytest.importorskip("openpyxl")
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write an xlsx with a 'Features' sheet
            feat_xlsx   = os.path.join(tmpdir, "features.xlsx")
            survey_path = os.path.join(tmpdir, "survey_labelled.csv")
            out_path    = os.path.join(tmpdir, "features_labeled.csv")

            rows = [
                (1, 1, 31.2, 41.0, 9.8, 24.4),
                (1, 2, 31.2, 41.0, 9.8, 24.4),
            ]
            df_feat = pd.DataFrame(rows, columns=[
                "Block_ID", "SubWindow_ID",
                "Block_Start_s", "Block_End_s", "Block_Duration_s",
                "mean_temperature",
            ])
            with pd.ExcelWriter(feat_xlsx, engine="openpyxl") as writer:
                df_feat.to_excel(writer, sheet_name="Features", index=False)

            self._make_labelled_survey_csv(survey_path)
            result = merge(feat_xlsx, survey_path, out_path)
            assert "Label" in result.columns
            assert "Block_ID" in result.columns
