"""
Lineup Features for Game Predictions

Combines player impact data with lineup/injury information to create
game-level features based on expected player availability.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from .player_impact import PlayerImpactModel


@dataclass
class LineupImpact:
    """Impact assessment for a team's lineup."""
    team_abbrev: str
    total_impact: float  # Sum of expected starters' impacts
    missing_impact: float  # Impact of injured players
    available_impact: float  # Impact of healthy players
    n_players: int  # Number of players with known impacts
    n_injured: int  # Number of injured players


class LineupFeatureBuilder:
    """
    Builds lineup-based features using player impact data.

    Uses the PlayerImpactModel to assess team strength based on
    expected available players (accounting for injuries).
    """

    # Number of players to consider as starters
    STARTER_COUNT = 8  # Top 8 by minutes (starters + key bench)

    def __init__(
        self,
        impact_model: Optional[PlayerImpactModel] = None,
        impact_model_path: Optional[str] = None,
    ):
        """
        Initialize lineup feature builder.

        Args:
            impact_model: Pre-fitted PlayerImpactModel
            impact_model_path: Path to load PlayerImpactModel from
        """
        if impact_model is not None:
            self.impact_model = impact_model
        elif impact_model_path and Path(impact_model_path).exists():
            self.impact_model = PlayerImpactModel()
            self.impact_model.load(impact_model_path)
        else:
            self.impact_model = None
            logger.warning("No player impact model provided")

        # Cache for team rosters (team_id -> list of player_ids sorted by impact)
        self._roster_cache: Dict[int, List[int]] = {}

    def set_impact_model(self, model: PlayerImpactModel) -> None:
        """Set or update the player impact model."""
        self.impact_model = model
        self._roster_cache.clear()

    def _build_roster_cache(self) -> None:
        """Build cache of top players per team."""
        if self.impact_model is None or not self.impact_model.is_fitted:
            return

        impacts_df = self.impact_model.get_impacts_df()

        if impacts_df.empty:
            return

        # Group by team and get top players by impact
        for team_id in impacts_df['team_id'].unique():
            if team_id == 0:  # Skip unknown team
                continue

            team_players = impacts_df[impacts_df['team_id'] == team_id]
            team_players = team_players.sort_values('minutes', ascending=False)

            # Store top players by minutes (they play the most)
            top_players = team_players.head(self.STARTER_COUNT)['player_id'].tolist()
            self._roster_cache[team_id] = top_players

        logger.info(f"Built roster cache for {len(self._roster_cache)} teams")

    def get_team_impact(
        self,
        team_id: int,
        injured_player_ids: Optional[List[int]] = None,
    ) -> LineupImpact:
        """
        Calculate team lineup impact accounting for injuries.

        Args:
            team_id: NBA team ID
            injured_player_ids: List of injured player IDs

        Returns:
            LineupImpact with breakdown of available/missing impact
        """
        if self.impact_model is None or not self.impact_model.is_fitted:
            return LineupImpact(
                team_abbrev='UNK',
                total_impact=0.0,
                missing_impact=0.0,
                available_impact=0.0,
                n_players=0,
                n_injured=0,
            )

        injured_set = set(injured_player_ids) if injured_player_ids else set()

        # Build roster cache if needed
        if not self._roster_cache:
            self._build_roster_cache()

        # Get team's top players
        roster = self._roster_cache.get(team_id, [])

        if not roster:
            # Try to get from impacts directly
            impacts_df = self.impact_model.get_impacts_df()
            team_players = impacts_df[impacts_df['team_id'] == team_id]
            if not team_players.empty:
                team_players = team_players.sort_values('minutes', ascending=False)
                roster = team_players.head(self.STARTER_COUNT)['player_id'].tolist()

        # Calculate impacts
        total_impact = 0.0
        missing_impact = 0.0
        available_impact = 0.0
        n_injured = 0

        team_abbrev = 'UNK'

        for player_id in roster:
            info = self.impact_model.get_player_info(player_id)
            if info is None:
                continue

            team_abbrev = info.team_abbrev
            total_impact += info.impact

            if player_id in injured_set:
                missing_impact += info.impact
                n_injured += 1
            else:
                available_impact += info.impact

        return LineupImpact(
            team_abbrev=team_abbrev,
            total_impact=total_impact,
            missing_impact=missing_impact,
            available_impact=available_impact,
            n_players=len(roster),
            n_injured=n_injured,
        )

    def build_game_features(
        self,
        home_team_id: int,
        away_team_id: int,
        home_injured_ids: Optional[List[int]] = None,
        away_injured_ids: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """
        Build lineup-based features for a game.

        Args:
            home_team_id: Home team NBA ID
            away_team_id: Away team NBA ID
            home_injured_ids: List of injured home player IDs
            away_injured_ids: List of injured away player IDs

        Returns:
            Dict of features
        """
        home_impact = self.get_team_impact(home_team_id, home_injured_ids)
        away_impact = self.get_team_impact(away_team_id, away_injured_ids)

        features = {
            # Raw impacts
            'home_lineup_impact': home_impact.available_impact,
            'away_lineup_impact': away_impact.available_impact,
            'home_missing_impact': home_impact.missing_impact,
            'away_missing_impact': away_impact.missing_impact,

            # Differentials
            'lineup_impact_diff': home_impact.available_impact - away_impact.available_impact,
            'missing_impact_diff': home_impact.missing_impact - away_impact.missing_impact,

            # Total team strength
            'home_total_impact': home_impact.total_impact,
            'away_total_impact': away_impact.total_impact,

            # Injury burden
            'home_injury_pct': (
                home_impact.missing_impact / home_impact.total_impact
                if home_impact.total_impact != 0 else 0
            ),
            'away_injury_pct': (
                away_impact.missing_impact / away_impact.total_impact
                if away_impact.total_impact != 0 else 0
            ),

            # Counts
            'home_n_injured': home_impact.n_injured,
            'away_n_injured': away_impact.n_injured,
        }

        return features

    def build_features_for_games(
        self,
        games_df: pd.DataFrame,
        injury_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Build lineup features for multiple games.

        Args:
            games_df: DataFrame with game info (must have home_team_id, away_team_id)
            injury_df: Optional DataFrame with injury info
                      (game_id, team_id, player_id, is_out)

        Returns:
            DataFrame with lineup features for each game
        """
        if self.impact_model is None or not self.impact_model.is_fitted:
            logger.warning("No impact model available, returning empty features")
            return pd.DataFrame()

        results = []

        for _, game in games_df.iterrows():
            game_id = game.get('game_id')
            home_team_id = game.get('home_team_id')
            away_team_id = game.get('away_team_id')

            # Get injuries for this game
            home_injured = []
            away_injured = []

            if injury_df is not None and game_id in injury_df['game_id'].values:
                game_injuries = injury_df[injury_df['game_id'] == game_id]

                home_inj = game_injuries[
                    (game_injuries['team_id'] == home_team_id) &
                    (game_injuries['is_out'] == True)
                ]
                away_inj = game_injuries[
                    (game_injuries['team_id'] == away_team_id) &
                    (game_injuries['is_out'] == True)
                ]

                home_injured = home_inj['player_id'].tolist()
                away_injured = away_inj['player_id'].tolist()

            # Build features
            features = self.build_game_features(
                home_team_id, away_team_id,
                home_injured, away_injured
            )
            features['game_id'] = game_id

            results.append(features)

        return pd.DataFrame(results)


def get_expected_lineup_impact(
    team_id: int,
    injured_player_ids: Optional[List[int]] = None,
    impact_model: Optional[PlayerImpactModel] = None,
    impact_model_path: str = "data/cache/player_impact/player_impact_model.parquet",
) -> float:
    """
    Get expected lineup impact for a team, accounting for injuries.

    Convenience function for quick impact calculation.

    Args:
        team_id: NBA team ID
        injured_player_ids: List of injured player IDs
        impact_model: Pre-loaded PlayerImpactModel
        impact_model_path: Path to load model from if not provided

    Returns:
        Net impact of available players (sum of RAPM values)
    """
    builder = LineupFeatureBuilder(
        impact_model=impact_model,
        impact_model_path=impact_model_path,
    )

    lineup = builder.get_team_impact(team_id, injured_player_ids)
    return lineup.available_impact


class LineupChemistryTracker:
    """
    Tracks how often player combinations have played together.

    Lineups that have logged more minutes together may perform
    better than expected based on individual impacts.
    """

    def __init__(self, cache_dir: str = "data/cache/lineup_chemistry"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # (player_a, player_b) -> minutes together
        self._pair_minutes: Dict[Tuple[int, int], float] = {}

    def add_stint(
        self,
        player_ids: List[int],
        minutes: float,
    ) -> None:
        """
        Record a stint where players played together.

        Args:
            player_ids: List of player IDs on court together
            minutes: Duration of stint in minutes
        """
        # Update pair minutes for all combinations
        for i, p1 in enumerate(player_ids):
            for p2 in player_ids[i+1:]:
                pair = tuple(sorted([p1, p2]))
                self._pair_minutes[pair] = self._pair_minutes.get(pair, 0) + minutes

    def get_chemistry_score(
        self,
        player_ids: List[int],
    ) -> float:
        """
        Calculate chemistry score for a group of players.

        Higher score = players have played together more.

        Args:
            player_ids: List of player IDs

        Returns:
            Average minutes together per pair (normalized)
        """
        if len(player_ids) < 2:
            return 0.0

        total_minutes = 0.0
        n_pairs = 0

        for i, p1 in enumerate(player_ids):
            for p2 in player_ids[i+1:]:
                pair = tuple(sorted([p1, p2]))
                total_minutes += self._pair_minutes.get(pair, 0)
                n_pairs += 1

        if n_pairs == 0:
            return 0.0

        avg_minutes = total_minutes / n_pairs

        # Normalize to 0-1 scale (100 minutes together = 1.0)
        return min(1.0, avg_minutes / 100)

    def save(self, path: Optional[str] = None) -> str:
        """Save chemistry data to disk."""
        path = path or str(self.cache_dir / "lineup_chemistry.parquet")

        records = [
            {'player_a': pair[0], 'player_b': pair[1], 'minutes': mins}
            for pair, mins in self._pair_minutes.items()
        ]

        df = pd.DataFrame(records)
        df.to_parquet(path)

        logger.info(f"Saved {len(records)} player pairs to {path}")
        return path

    def load(self, path: Optional[str] = None) -> "LineupChemistryTracker":
        """Load chemistry data from disk."""
        path = path or str(self.cache_dir / "lineup_chemistry.parquet")

        if not Path(path).exists():
            logger.warning(f"No chemistry data at {path}")
            return self

        df = pd.read_parquet(path)

        self._pair_minutes = {
            (row['player_a'], row['player_b']): row['minutes']
            for _, row in df.iterrows()
        }

        logger.info(f"Loaded {len(self._pair_minutes)} player pairs")
        return self


if __name__ == "__main__":
    # Quick test
    print("Testing LineupFeatureBuilder...")

    # Create a mock impact model for testing
    model = PlayerImpactModel()

    # Test without fitted model
    builder = LineupFeatureBuilder(impact_model=model)

    features = builder.build_game_features(
        home_team_id=1610612747,  # Lakers
        away_team_id=1610612738,  # Celtics
    )

    print(f"\nGame features (no model):")
    for k, v in features.items():
        print(f"  {k}: {v}")
