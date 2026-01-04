"""
Confirmed Lineup Feature Builder

Generates features from confirmed starting lineups.
Integrates with existing player impact calculations to estimate lineup strength.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict

import pandas as pd
from loguru import logger

from src.data.lineup_scrapers import ESPNLineupClient
from src.data.espn_injuries import PlayerImpactCalculator
from src.utils.constants import BETS_DB_PATH


class ConfirmedLineupFeatureBuilder:
    """
    Build lineup-related features from confirmed starters.

    Features generated:
    - confirmed_home_impact: Sum of confirmed starter impacts
    - confirmed_away_impact: Sum of confirmed starter impacts
    - lineup_impact_diff: Home - Away impact difference
    - home_lineup_uncertainty: % of questionable/GTD starters
    - away_lineup_uncertainty: % of questionable/GTD starters
    """

    def __init__(self, db_path: str = None):
        """
        Initialize lineup feature builder.

        Args:
            db_path: Path to database (defaults to BETS_DB_PATH)
        """
        self.lineup_client = ESPNLineupClient(db_path=db_path or BETS_DB_PATH)
        self.impact_calc = PlayerImpactCalculator()
        self._cache = {}
        self._cache_time = None
        self._cache_duration = timedelta(minutes=30)

    def get_game_features(
        self,
        game_id: str = None,
        home_team: str = None,
        away_team: str = None,
        game_date: str = None
    ) -> Dict[str, float]:
        """
        Get lineup features for a game.

        Args:
            game_id: Game ID (preferred)
            home_team: Home team abbreviation (fallback)
            away_team: Away team abbreviation (fallback)
            game_date: Game date in YYYY-MM-DD (fallback)

        Returns:
            Dictionary of lineup features
        """
        # Try to get lineups by game_id first
        if game_id:
            lineups = self.lineup_client.get_lineups(game_id=game_id, starters_only=True)
        elif game_date and (home_team or away_team):
            # Fallback: get by date and filter by team
            lineups = self.lineup_client.get_lineups(game_date=game_date, starters_only=True)
            if home_team and away_team and not lineups.empty:
                lineups = lineups[lineups["team_abbrev"].isin([home_team, away_team])]
        else:
            logger.debug("Insufficient parameters to fetch lineups")
            return self._empty_features()

        if lineups.empty:
            logger.debug("No confirmed lineups available")
            return self._empty_features()

        # Calculate impact for each team
        home_impact, home_uncertainty = self._calculate_team_lineup_impact(
            lineups, home_team
        )
        away_impact, away_uncertainty = self._calculate_team_lineup_impact(
            lineups, away_team
        )

        return {
            "confirmed_home_impact": round(home_impact, 2),
            "confirmed_away_impact": round(away_impact, 2),
            "lineup_impact_diff": round(home_impact - away_impact, 2),
            "home_lineup_uncertainty": round(home_uncertainty, 3),
            "away_lineup_uncertainty": round(away_uncertainty, 3),
        }

    def _calculate_team_lineup_impact(
        self,
        lineups: pd.DataFrame,
        team_abbrev: str
    ) -> tuple:
        """
        Calculate total impact and uncertainty for a team's lineup.

        Args:
            lineups: DataFrame with lineup data
            team_abbrev: Team abbreviation

        Returns:
            Tuple of (total_impact, uncertainty_rate)
        """
        if team_abbrev is None:
            return 0.0, 0.0

        team_lineups = lineups[lineups["team_abbrev"] == team_abbrev]

        if team_lineups.empty:
            return 0.0, 0.0

        # Pre-load stats for efficiency
        stats_df = self.impact_calc.stats_cache.refresh()

        total_impact = 0.0
        uncertain_count = 0
        total_starters = len(team_lineups)

        for _, player in team_lineups.iterrows():
            player_name = player["player_name"]
            status = player.get("status", "active")

            # Get player stats
            player_stats = self._lookup_player_stats(player_name, team_abbrev, stats_df)

            if player_stats:
                # Calculate impact
                impact = self.impact_calc.calculate_player_impact(player_stats)
                total_impact += impact

            # Track uncertainty (questionable, GTD, etc.)
            if status.lower() in ["questionable", "doubtful", "game time decision", "gtd"]:
                uncertain_count += 1

        # Calculate uncertainty rate
        uncertainty = uncertain_count / total_starters if total_starters > 0 else 0.0

        return total_impact, uncertainty

    def _lookup_player_stats(
        self,
        player_name: str,
        team: str,
        stats_df: pd.DataFrame
    ) -> Optional[dict]:
        """
        Look up player stats from pre-loaded DataFrame.

        Args:
            player_name: Player name
            team: Team abbreviation
            stats_df: Pre-loaded stats DataFrame

        Returns:
            Player stats dict or None
        """
        if stats_df is None or stats_df.empty:
            return None

        name_lower = player_name.lower().strip()

        # Try exact match
        match = stats_df[stats_df["player_name"].str.lower() == name_lower]

        # Try partial match if no exact match
        if match.empty:
            last_name = name_lower.split()[-1] if name_lower else ""
            if last_name:
                match = stats_df[stats_df["player_name"].str.lower().str.contains(last_name, na=False)]

        if not match.empty:
            return match.iloc[0].to_dict()

        return None

    def _empty_features(self) -> Dict[str, float]:
        """Return empty/neutral lineup features."""
        return {
            "confirmed_home_impact": 0.0,
            "confirmed_away_impact": 0.0,
            "lineup_impact_diff": 0.0,
            "home_lineup_uncertainty": 0.0,
            "away_lineup_uncertainty": 0.0,
        }

    def get_features_for_games(
        self,
        games_df: pd.DataFrame,
        game_id_col: str = "game_id",
        home_team_col: str = "home_team",
        away_team_col: str = "away_team",
        game_date_col: str = "game_date"
    ) -> pd.DataFrame:
        """
        Generate lineup features for multiple games.

        Args:
            games_df: DataFrame with game information
            game_id_col: Column name for game ID
            home_team_col: Column name for home team
            away_team_col: Column name for away team
            game_date_col: Column name for game date

        Returns:
            DataFrame with lineup features added
        """
        features_list = []

        for _, game in games_df.iterrows():
            features = self.get_game_features(
                game_id=game.get(game_id_col),
                home_team=game.get(home_team_col),
                away_team=game.get(away_team_col),
                game_date=game.get(game_date_col)
            )
            features_list.append(features)

        features_df = pd.DataFrame(features_list)
        return pd.concat([games_df.reset_index(drop=True), features_df], axis=1)


if __name__ == "__main__":
    # Test the lineup feature builder
    builder = ConfirmedLineupFeatureBuilder()

    print("Confirmed Lineup Feature Builder Test")
    print("=" * 60)

    # Test empty features
    features = builder.get_game_features(home_team="LAL", away_team="BOS")
    print("\nFeatures without confirmed lineups (should be zeros):")
    for k, v in features.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("To get actual features, lineups must be collected first")
