from datetime import UTC, datetime

import pandas as pd
import pytest

from forecast.modeling.training import train_walk_forward


def test_training_refuses_labels_observed_after_test_cutoff(tmp_path) -> None:
    seasons = [season for season in ("2020-21", "2021-22", "2022-23", "2023-24") for _ in range(3)]
    kickoffs = [datetime(2020 + index // 3, 10, 1, 15, tzinfo=UTC) for index in range(len(seasons))]

    with pytest.raises(ValueError, match="no leakage-safe walk-forward folds"):
        train_walk_forward(
            pd.DataFrame({"numeric_feature": range(len(seasons))}),
            [index % 3 for index in range(len(seasons))],
            leagues=["PL"] * len(seasons),
            seasons=seasons,
            model_dir=tmp_path / "models",
            kickoffs=kickoffs,
            label_available_at=[datetime(2025, 1, 1, tzinfo=UTC)] * len(seasons),
        )
