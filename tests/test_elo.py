"""Tests for the draw-aware Elo baseline."""

from datetime import UTC, datetime, timedelta

import pytest

from forecast.modeling.elo import EloBaseline


def make_matches():
    start = datetime(2020, 8, 1, tzinfo=UTC)
    scores = [(2, 0), (0, 0), (0, 1), (3, 1), (1, 1), (0, 2), (1, 0), (0, 0), (0, 1)]
    return [
        {
            "id": str(index),
            "kickoff_utc": start + timedelta(days=index),
            "result_available_at": start + timedelta(days=index, hours=3),
            "season": "2020-21" if index < 6 else "2021-22",
            "home_team_id": "A" if index % 2 == 0 else "B",
            "away_team_id": "B" if index % 2 == 0 else "A",
            "home_goals": score[0],
            "away_goals": score[1],
        }
        for index, score in enumerate(scores)
    ]


def test_elo_learns_home_draw_away_probabilities() -> None:
    model = EloBaseline().fit(
        make_matches(),
        through=datetime(2021, 1, 1, tzinfo=UTC),
    )
    probabilities = model.predict_proba("A", "B", season="2021-22")

    assert len(probabilities) == 3
    assert sum(probabilities) == pytest.approx(1.0)
    assert all(0 <= probability <= 1 for probability in probabilities)
    assert len(set(round(value, 6) for value in probabilities)) > 1


def test_elo_requires_past_outcome_data() -> None:
    with pytest.raises(ValueError, match="requires at least two observed outcomes"):
        EloBaseline().fit(
            make_matches()[:1],
            through=datetime(2021, 1, 1, tzinfo=UTC),
        )
