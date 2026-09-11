"""Unit tests for the rule-based query parser.

These tests use a stub DwellSummary-like dataclass instead of the real
one from pipeline.dwell_time. That keeps the test file importable
without supervision/ultralytics/opencv, which the heavy CV pipeline
pulls in. The query module itself is duck-typed on the summary shape.

Run with:
    pytest tests/test_query_parser.py -v
"""

from dataclasses import dataclass

import pytest

from query.nl_interface import ParsedQuery, QueryParser, QueryResponder, QueryType


@dataclass
class _StubSummary:
    """Duck-typed stand-in for pipeline.dwell_time.DwellSummary."""

    n_records: int
    n_pedestrians: int
    n_vehicles: int
    mean_dwell_seconds: float
    median_dwell_seconds: float
    min_dwell_seconds: float
    max_dwell_seconds: float
    still_present: int


class TestQueryParser:
    """Tests for keyword-based query parsing."""

    def setup_method(self):
        self.parser = QueryParser()

    def test_peak_hour_query(self):
        q = self.parser.parse("What was the peak hour this morning?")
        assert q.query_type == QueryType.PEAK_HOUR
        assert q.time_window == "this morning"

    def test_total_visits_query(self):
        q = self.parser.parse("How many people came in today?")
        assert q.query_type == QueryType.TOTAL_VISITS
        assert q.time_window == "today"

    def test_busyness_query(self):
        q = self.parser.parse("Is it busy right now?")
        assert q.query_type == QueryType.BUSYNESS_NOW

    def test_dwell_query(self):
        q = self.parser.parse("How long did people stay?")
        assert q.query_type == QueryType.DWELL_TIME

    def test_compare_days_query(self):
        q = self.parser.parse("Compare Monday vs Friday traffic")
        assert q.query_type == QueryType.COMPARE_DAYS
        # The parser iterates DAY_NAMES dict order. Both orders are valid;
        # just assert both days are present, in some order.
        assert set(q.comparison) == {"monday", "friday"}

    def test_unknown_query(self):
        q = self.parser.parse("What is the meaning of life?")
        assert q.query_type == QueryType.UNKNOWN

    def test_class_extraction_pedestrian(self):
        q = self.parser.parse("How many pedestrians came in?")
        assert q.target_class == "pedestrians"

    def test_class_extraction_vehicle(self):
        q = self.parser.parse("How many cars came in?")
        assert q.target_class == "vehicles"


class TestQueryResponder:
    """Tests for the plain-English response formatter."""

    def setup_method(self):
        self.responder = QueryResponder()

    def test_unknown_response(self):
        summary = _StubSummary(
            n_records=0,
            n_pedestrians=0,
            n_vehicles=0,
            mean_dwell_seconds=0.0,
            median_dwell_seconds=0.0,
            min_dwell_seconds=0.0,
            max_dwell_seconds=0.0,
            still_present=0,
        )
        q = ParsedQuery(query_type=QueryType.UNKNOWN)
        response = self.responder.answer(q, summary)
        assert "don't understand" in response.lower()

    def test_total_visits_response(self):
        summary = _StubSummary(
            n_records=50,
            n_pedestrians=42,
            n_vehicles=8,
            mean_dwell_seconds=180.0,
            median_dwell_seconds=120.0,
            min_dwell_seconds=5.0,
            max_dwell_seconds=600.0,
            still_present=3,
        )
        q = ParsedQuery(query_type=QueryType.TOTAL_VISITS)
        response = self.responder.answer(q, summary)
        assert "50" in response
        assert "42" in response
        assert "8" in response


class TestPrivacyConfig:
    """Tests for the privacy configuration defaults."""

    def test_default_is_strict(self):
        from pipeline.privacy import PrivacyConfig

        cfg = PrivacyConfig()
        assert cfg.on_device_only is True
        assert cfg.no_per_person_persistence is True
        assert cfg.session_scoped_ids is True
        assert cfg.refuse_face_recognition is True

    def test_env_can_loosen(self):
        import os

        from pipeline.privacy import load_default_privacy_config

        os.environ["CAMPUS_ACCESS_ON_DEVICE_ONLY"] = "false"
        try:
            cfg = load_default_privacy_config()
            assert cfg.on_device_only is False
        finally:
            del os.environ["CAMPUS_ACCESS_ON_DEVICE_ONLY"]

    def test_env_default_remains_strict(self):
        import os

        from pipeline.privacy import load_default_privacy_config

        # Make sure no env vars are set.
        for var in [
            "CAMPUS_ACCESS_ON_DEVICE_ONLY",
            "CAMPUS_ACCESS_NO_PERSON_PERSISTENCE",
            "CAMPUS_ACCESS_SESSION_SCOPED_IDS",
            "CAMPUS_ACCESS_REFUSE_FACE_RECOGNITION",
        ]:
            os.environ.pop(var, None)
        cfg = load_default_privacy_config()
        assert cfg.on_device_only is True


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
