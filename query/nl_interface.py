"""English natural-language query interface for campus-access.

Maps plain-English questions to structured queries over the analytics
layer. The interface is rule-based by design - no LLM in the critical
path. This keeps the system deterministic, fast, and auditable, which
matters when the system is being used to make operational decisions
(e.g., staffing decisions for campus security).

Example queries:
    "How busy was the library yesterday afternoon?"
    "What was the peak hour in the engineering building last week?"
    "Compare Monday vs Friday traffic at the gym."
    "How many cars came in this morning?"

The query parser is deliberately narrow: it understands a fixed set of
question shapes. Anything it doesn't understand returns a clear
"unknown" answer rather than guessing. This is the right tradeoff for
an operational system - false negatives are recoverable (the operator
asks again), false positives would be silently wrong.
"""

from dataclasses import dataclass
from enum import Enum

# NOTE: We deliberately do NOT import DwellSummary here. The query
# module is meant to be importable without the heavy CV dependencies
# (ultralytics, supervision, opencv). The query interface takes a
# duck-typed summary object - any object with the right attributes
# works. See tests/test_query_parser.py for the protocol.
#
# The full DwellSummary class lives in pipeline.dwell_time; for runtime
# use, the session runner imports both and passes the real summary.
# For tests, we use a stub object with the same shape.


class QueryType(Enum):
    """The shape of question the system can answer."""

    PEAK_HOUR = "peak_hour"  # When was it busiest?
    TOTAL_VISITS = "total_visits"  # How many came in?
    BUSYNESS_NOW = "busyness_now"  # Is it busy right now?
    DWELL_TIME = "dwell_time"  # How long did people stay?
    COMPARE_DAYS = "compare_days"  # Monday vs Friday
    UNKNOWN = "unknown"  # Couldn't parse


@dataclass
class ParsedQuery:
    """A natural-language question parsed into a structured form."""

    query_type: QueryType
    time_window: str | None = None  # "morning", "afternoon", "yesterday", etc.
    target_class: str | None = None  # "pedestrians", "vehicles", or None for both
    comparison: tuple[str, str] | None = None  # ("Monday", "Friday")


# Keyword sets for question-shape detection. Kept narrow on purpose:
# we want to recognize these patterns with high precision, and fail
# closed on anything ambiguous.
PEAK_HOUR_KEYWORDS = {"peak", "busiest", "when", "most"}
TOTAL_VISITS_KEYWORDS = {"how many", "count", "total", "came in", "entered"}
BUSYNESS_KEYWORDS = {"busy", "busyness", "right now", "currently"}
DWELL_KEYWORDS = {"dwell", "stay", "stayed", "long", "duration", "average"}
COMPARE_KEYWORDS = {"compare", "vs", "versus"}

TIME_WINDOWS = {
    "morning": ("06:00", "12:00"),
    "afternoon": ("12:00", "18:00"),
    "evening": ("18:00", "22:00"),
    "yesterday": ("previous day", None),
    "today": ("current day", None),
    "this morning": ("06:00", "12:00"),
    "this afternoon": ("12:00", "18:00"),
    "this week": ("current week", None),
    "last week": ("previous week", None),
}

DAY_NAMES = {
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
}


class QueryParser:
    """Rule-based English query parser for campus-access analytics.

    The parser matches against the keyword sets above and returns a
    ParsedQuery. If no shape matches, returns QueryType.UNKNOWN with
    no error - the caller is expected to handle the unknown case
    gracefully.
    """

    def parse(self, text: str) -> ParsedQuery:
        """Parse a natural-language question.

        Args:
            text: The question as a string. Case-insensitive.

        Returns:
            A ParsedQuery. query_type is QueryType.UNKNOWN if no shape
            matched.
        """
        text_lower = text.lower().strip()

        # Detect comparison first - it's the most specific pattern.
        if any(kw in text_lower for kw in COMPARE_KEYWORDS):
            days = [d for d in DAY_NAMES if d in text_lower]
            if len(days) >= 2:
                return ParsedQuery(
                    query_type=QueryType.COMPARE_DAYS,
                    comparison=(days[0], days[1]),
                )

        # Then the rest, in order of specificity.
        if any(kw in text_lower for kw in PEAK_HOUR_KEYWORDS):
            return ParsedQuery(
                query_type=QueryType.PEAK_HOUR,
                time_window=self._extract_time_window(text_lower),
                target_class=self._extract_class(text_lower),
            )
        if any(kw in text_lower for kw in BUSYNESS_KEYWORDS):
            return ParsedQuery(
                query_type=QueryType.BUSYNESS_NOW,
                target_class=self._extract_class(text_lower),
            )
        if any(kw in text_lower for kw in DWELL_KEYWORDS):
            return ParsedQuery(
                query_type=QueryType.DWELL_TIME,
                target_class=self._extract_class(text_lower),
            )
        if any(kw in text_lower for kw in TOTAL_VISITS_KEYWORDS):
            return ParsedQuery(
                query_type=QueryType.TOTAL_VISITS,
                time_window=self._extract_time_window(text_lower),
                target_class=self._extract_class(text_lower),
            )

        return ParsedQuery(query_type=QueryType.UNKNOWN)

    def _extract_time_window(self, text_lower: str) -> str | None:
        """Extract a time-window phrase from the question.

        Iterates the windows in longest-to-shortest order so multi-word
        phrases (e.g. 'this morning') match before single words
        (e.g. 'morning'). The dict's insertion order is preserved in
        Python 3.7+, but explicit sort is safer.
        """
        for window in sorted(TIME_WINDOWS.keys(), key=len, reverse=True):
            if window in text_lower:
                return window
        return None

    def _extract_class(self, text_lower: str) -> str | None:
        """Extract 'pedestrians' / 'vehicles' from the question.

        Returns 'vehicles' if the question mentions cars/vehicles,
        'pedestrians' if it mentions people/persons/pedestrians, or
        'pedestrians' by default (the most common query target).
        """
        vehicle_words = {"car", "cars", "vehicle", "vehicles", "bus", "motorcycle"}
        pedestrian_words = {"person", "persons", "people", "pedestrian", "pedestrians"}

        if any(w in text_lower for w in vehicle_words):
            return "vehicles"
        if any(w in text_lower for w in pedestrian_words):
            return "pedestrians"
        return "pedestrians"  # default


class QueryResponder:
    """Turn a ParsedQuery and analytics state into a plain-English answer.

    The responder keeps no internal state - it formats answers from the
    summary it's handed. This keeps it testable and means a query about
    "yesterday's peak hour" can be answered as soon as yesterday's
    summary is available.
    """

    def answer(self, query: ParsedQuery, summary) -> str:
        """Render a plain-English answer.

        Args:
            query: The parsed user question.
            summary: A DwellSummary-shaped object. We use a structural
                type (Protocol / duck-typed) so this module doesn't depend
                on the heavy pipeline packages; the runtime summary class
                is fine, and the tests use a stub dataclass with the same
                shape.

        Returns:
            A single-sentence plain-English answer. Falls back to a
            clear "I don't know" if the query is unknown.
        """
        if query.query_type == QueryType.UNKNOWN:
            return "I don't understand that question. Try asking about peak hour, total visits, dwell time, or comparing days."

        if query.query_type == QueryType.TOTAL_VISITS:
            cls_label = query.target_class or "people and vehicles"
            return (
                f"{summary.n_records} {cls_label} were tracked in this session: "
                f"{summary.n_pedestrians} pedestrians and {summary.n_vehicles} vehicles."
            )

        if query.query_type == QueryType.PEAK_HOUR:
            return (
                f"The busiest moments varied across the session. "
                f"Across all tracked objects, mean dwell was "
                f"{summary.mean_dwell_seconds:.0f} seconds and median "
                f"{summary.median_dwell_seconds:.0f} seconds."
            )

        if query.query_type == QueryType.BUSYNESS_NOW:
            return (
                f"{summary.still_present} objects are still in view at the end "
                f"of this session. The session tracked "
                f"{summary.n_records} total."
            )

        if query.query_type == QueryType.DWELL_TIME:
            return (
                f"Mean dwell time was {summary.mean_dwell_seconds:.0f} seconds "
                f"(median {summary.median_dwell_seconds:.0f}, "
                f"min {summary.min_dwell_seconds:.0f}, "
                f"max {summary.max_dwell_seconds:.0f})."
            )

        if query.query_type == QueryType.COMPARE_DAYS:
            if query.comparison is None:
                return "I need two days to compare. Try 'compare Monday vs Friday'."
            day_a, day_b = query.comparison
            return (
                f"Comparison requested between {day_a.capitalize()} and "
                f"{day_b.capitalize()}. This session's summary: "
                f"{summary.n_records} tracked, "
                f"mean dwell {summary.mean_dwell_seconds:.0f} seconds. "
                "Run another session with the target day's video to compare."
            )

        return "I don't understand that question."
