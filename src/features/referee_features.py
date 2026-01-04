"""
Referee Feature Builder

Generates betting-relevant features from referee assignments and historical statistics.
Tracks crew tendencies for totals, pace, and home/away bias.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, List

import pandas as pd
from loguru import logger

from src.data.referee_data import RefereeDataClient
from src.utils.constants import BETS_DB_PATH


class RefereeFeatureBuilder:
    """
    Build referee-related features for NBA predictions.

    Features generated:
    - ref_crew_total_bias: Points above/below league average
    - ref_crew_pace_factor: Pace multiplier (1.0 = average)
    - ref_crew_over_rate: Historical over hit rate
    - ref_crew_home_bias: Home win rate with this crew
    """

    def __init__(self, db_path: str = None):
        """
        Initialize referee feature builder.

        Args:
            db_path: Path to database (defaults to BETS_DB_PATH)
        """
        self.client = RefereeDataClient(db_path=db_path or BETS_DB_PATH)
        self._cache = {}
        self._cache_time = None
        self._cache_duration = timedelta(hours=6)

    def get_game_features(self, game_id: str) -> Dict[str, float]:
        """
        Get referee features for a specific game.

        Args:
            game_id: NBA game ID

        Returns:
            Dictionary of referee features
        """
        # Get referee assignments for this game
        assignments = self.client.get_referee_assignments(game_id=game_id)

        if assignments.empty:
            logger.debug(f"No referee assignments found for game {game_id}")
            return self._empty_features()

        # Get unique referees for this game
        ref_names = assignments["ref_name"].unique().tolist()

        # Get statistics for each referee
        crew_stats = []
        for ref_name in ref_names:
            stats = self.client.get_referee_stats(ref_name=ref_name)
            if not stats.empty:
                # Get most recent season stats
                latest = stats.sort_values("season", ascending=False).iloc[0]
                crew_stats.append(latest.to_dict())

        if not crew_stats:
            logger.debug(f"No referee stats available for game {game_id}")
            return self._empty_features()

        # Aggregate crew statistics
        return self._aggregate_crew_features(crew_stats)

    def get_game_features_by_teams(
        self,
        home_team: str,
        away_team: str,
        game_date: str = None
    ) -> Dict[str, float]:
        """
        Get referee features for a game by team matchup.

        This is a fallback when game_id is not available yet.
        Returns empty features since we need actual referee assignments.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            game_date: Game date (YYYY-MM-DD)

        Returns:
            Dictionary of referee features
        """
        # Without game_id, we can't get referee assignments yet
        logger.debug(f"Referee assignments not available without game_id for {home_team} vs {away_team}")
        return self._empty_features()

    def _aggregate_crew_features(self, crew_stats: List[Dict]) -> Dict[str, float]:
        """
        Aggregate statistics for a referee crew.

        Args:
            crew_stats: List of referee stat dictionaries

        Returns:
            Aggregated features
        """
        # Calculate crew averages
        total_bias_values = [s.get("avg_total_points", 0) or 0 for s in crew_stats]
        pace_values = [s.get("pace_factor", 1.0) or 1.0 for s in crew_stats]
        over_rates = [s.get("over_rate", 0.5) or 0.5 for s in crew_stats]
        home_bias_values = [s.get("home_win_rate", 0.5) or 0.5 for s in crew_stats]

        # Average across crew (simple mean for now)
        ref_crew_total_bias = sum(total_bias_values) / len(total_bias_values) if total_bias_values else 0.0
        ref_crew_pace_factor = sum(pace_values) / len(pace_values) if pace_values else 1.0
        ref_crew_over_rate = sum(over_rates) / len(over_rates) if over_rates else 0.5
        ref_crew_home_bias = sum(home_bias_values) / len(home_bias_values) if home_bias_values else 0.5

        return {
            "ref_crew_total_bias": round(ref_crew_total_bias, 2),
            "ref_crew_pace_factor": round(ref_crew_pace_factor, 3),
            "ref_crew_over_rate": round(ref_crew_over_rate, 3),
            "ref_crew_home_bias": round(ref_crew_home_bias, 3),
            "ref_crew_size": len(crew_stats),
        }

    def _empty_features(self) -> Dict[str, float]:
        """Return neutral/empty referee features."""
        return {
            "ref_crew_total_bias": 0.0,
            "ref_crew_pace_factor": 1.0,  # Neutral
            "ref_crew_over_rate": 0.5,  # Neutral
            "ref_crew_home_bias": 0.5,  # Neutral
            "ref_crew_size": 0,
        }

    def get_features_for_games(
        self,
        games_df: pd.DataFrame,
        game_id_col: str = "game_id"
    ) -> pd.DataFrame:
        """
        Generate referee features for multiple games.

        Args:
            games_df: DataFrame with game information
            game_id_col: Column name for game ID

        Returns:
            DataFrame with referee features added
        """
        features_list = []

        for _, game in games_df.iterrows():
            game_id = game.get(game_id_col)
            if game_id:
                features = self.get_game_features(game_id)
            else:
                features = self._empty_features()

            features_list.append(features)

        features_df = pd.DataFrame(features_list)
        return pd.concat([games_df.reset_index(drop=True), features_df], axis=1)

    def get_crew_summary(self, game_id: str) -> Dict:
        """
        Get a summary of the referee crew for a game.

        Args:
            game_id: NBA game ID

        Returns:
            Dictionary with crew information
        """
        assignments = self.client.get_referee_assignments(game_id=game_id)

        if assignments.empty:
            return {
                "game_id": game_id,
                "crew": [],
                "crew_size": 0,
                "has_stats": False,
            }

        ref_names = assignments["ref_name"].unique().tolist()

        crew_info = []
        for ref_name in ref_names:
            stats = self.client.get_referee_stats(ref_name=ref_name)
            has_stats = not stats.empty

            crew_info.append({
                "name": ref_name,
                "has_stats": has_stats,
            })

        return {
            "game_id": game_id,
            "crew": crew_info,
            "crew_size": len(crew_info),
            "has_stats": any(c["has_stats"] for c in crew_info),
        }


if __name__ == "__main__":
    # Test the referee feature builder
    builder = RefereeFeatureBuilder()

    # Example: Get features for a hypothetical game
    print("Referee Feature Builder Test")
    print("=" * 60)

    # Test empty features (no game_id)
    features = builder.get_game_features_by_teams("LAL", "BOS")
    print("\\nFeatures without game_id (should be neutral):")
    for k, v in features.items():
        print(f"  {k}: {v}")

    print("\\n" + "=" * 60)
    print("To get actual features, need game_id from today's games")
