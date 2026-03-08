"""ELO rating system for F1 drivers."""

import pandas as pd

from config.settings import ELO_INITIAL_RATING, ELO_K_FACTOR
from src.utils.logger import logger


class EloRating:
    """Compute ELO ratings for drivers based on head-to-head race results."""

    def __init__(self, k_factor: int = ELO_K_FACTOR, initial: int = ELO_INITIAL_RATING):
        self.k_factor = k_factor
        self.initial = initial
        self.ratings: dict[int, float] = {}

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        """Expected score for player A against player B."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))

    def _get_rating(self, driver_id: int) -> float:
        """Get current rating for a driver, initializing if needed."""
        if driver_id not in self.ratings:
            self.ratings[driver_id] = float(self.initial)
        return self.ratings[driver_id]

    def update_race(self, finishing_order: list[int]) -> None:
        """Update ratings based on a race finishing order.

        Args:
            finishing_order: List of driver_ids in finishing order (1st place first).
        """
        n = len(finishing_order)
        if n < 2:
            return

        new_ratings = {}
        for i, driver_a in enumerate(finishing_order):
            rating_a = self._get_rating(driver_a)
            total_delta = 0.0

            for j, driver_b in enumerate(finishing_order):
                if i == j:
                    continue
                rating_b = self._get_rating(driver_b)
                expected = self._expected_score(rating_a, rating_b)
                actual = 1.0 if i < j else 0.0  # driver_a beat driver_b
                total_delta += self.k_factor * (actual - expected)

            # Scale by number of opponents to keep deltas reasonable
            new_ratings[driver_a] = rating_a + total_delta / (n - 1)

        self.ratings.update(new_ratings)

    def compute_all_ratings(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Compute ELO ratings from a full results DataFrame.

        Args:
            results_df: DataFrame with columns: season, round, driver_id, position.
                        Must be sorted by season, round.

        Returns:
            DataFrame with columns: season, round, driver_id, elo_rating
        """
        self.ratings.clear()
        records = []

        grouped = results_df.groupby(["season", "round"])
        for (season, round_num), group in grouped:
            # Record pre-race ELO for each driver
            for _, row in group.iterrows():
                driver_id = row["driver_id"]
                records.append({
                    "season": season,
                    "round": round_num,
                    "driver_id": driver_id,
                    "elo_rating": self._get_rating(driver_id),
                })

            # Update ratings based on finishing order
            valid = group.dropna(subset=["position"]).sort_values("position")
            if len(valid) >= 2:
                self.update_race(valid["driver_id"].tolist())

        return pd.DataFrame(records)
