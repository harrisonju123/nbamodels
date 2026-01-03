"""
Game Feature Engineering

Combines team features into game-level prediction features.
Integrates player impact and lineup features for more granular signals.
"""

import os
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from .team_features import TeamFeatureBuilder
from .elo import EloRatingSystem

# Optional imports for player-level features
try:
    from .lineup_features import LineupFeatureBuilder
    from .player_impact import PlayerImpactModel
    HAS_LINEUP_FEATURES = True
except ImportError:
    HAS_LINEUP_FEATURES = False
    LineupFeatureBuilder = None
    PlayerImpactModel = None

# Optional import for matchup features
try:
    from .matchup_features import MatchupFeatureBuilder
    HAS_MATCHUP_FEATURES = True
except ImportError:
    HAS_MATCHUP_FEATURES = False
    MatchupFeatureBuilder = None


class GameFeatureBuilder:
    """Builds game-level features for prediction."""

    # Default path for player impact model
    DEFAULT_IMPACT_PATH = "data/cache/player_impact/player_impact_model.parquet"

    def __init__(
        self,
        team_builder: Optional[TeamFeatureBuilder] = None,
        use_elo: bool = True,
        use_lineup_features: bool = None,  # Auto-detect if None
        use_matchup_features: bool = True,
        player_impact_path: Optional[str] = None,
    ):
        self.team_builder = team_builder or TeamFeatureBuilder()
        self.use_elo = use_elo
        self.elo_system = EloRatingSystem() if use_elo else None

        # Determine player impact model path
        impact_path = player_impact_path or self.DEFAULT_IMPACT_PATH

        # Auto-detect lineup features if not explicitly set
        # Enable if model exists and HAS_LINEUP_FEATURES is True
        if use_lineup_features is None:
            use_lineup_features = (
                HAS_LINEUP_FEATURES and
                os.path.exists(impact_path)
            )
            if use_lineup_features:
                logger.info(f"Auto-enabled lineup features (found {impact_path})")

        # Initialize lineup feature builder if requested
        self.use_lineup_features = use_lineup_features and HAS_LINEUP_FEATURES
        self.lineup_builder = None

        if self.use_lineup_features:
            try:
                self.lineup_builder = LineupFeatureBuilder(impact_model_path=impact_path)
                logger.info("Lineup feature builder initialized")
            except Exception as e:
                logger.warning(f"Could not initialize lineup features: {e}")
                self.use_lineup_features = False

        # Initialize matchup feature builder if requested
        self.use_matchup_features = use_matchup_features and HAS_MATCHUP_FEATURES
        self.matchup_builder = None

        if self.use_matchup_features:
            try:
                self.matchup_builder = MatchupFeatureBuilder()
                logger.info("Matchup feature builder initialized")
            except Exception as e:
                logger.warning(f"Could not initialize matchup features: {e}")
                self.use_matchup_features = False

    def build_game_features(
        self,
        games_df: pd.DataFrame,
        odds_df: Optional[pd.DataFrame] = None,
        player_features_df: Optional[pd.DataFrame] = None,
        injury_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Build game-level features for prediction.

        Args:
            games_df: DataFrame with game results
            odds_df: Optional DataFrame with betting odds
            player_features_df: Optional DataFrame with player-aggregated features
            injury_df: Optional DataFrame with injury info (game_id, team_id, player_id, is_out)

        Returns:
            DataFrame with game features and target
        """
        # Build team features
        team_features = self.team_builder.build_all_features(games_df)

        # Separate home and away team features
        home_features = team_features[team_features["is_home"] == 1].copy()
        away_features = team_features[team_features["is_home"] == 0].copy()

        # Rename columns for merging
        home_cols = {col: f"home_{col}" for col in home_features.columns
                     if col not in ["game_id", "date", "season"]}
        away_cols = {col: f"away_{col}" for col in away_features.columns
                     if col not in ["game_id", "date", "season"]}

        home_features = home_features.rename(columns=home_cols)
        away_features = away_features.rename(columns=away_cols)

        # Merge home and away features
        game_features = pd.merge(
            home_features,
            away_features[["game_id"] + list(away_cols.values())],
            on="game_id",
            suffixes=("", "_drop")
        )

        # Calculate differential features
        game_features = self._add_differential_features(game_features)

        # Add target variables
        game_features = self._add_targets(game_features, games_df)

        # Add odds features if available
        if odds_df is not None and not odds_df.empty:
            game_features = self._add_odds_features(game_features, odds_df)

        # Add player features if available
        if player_features_df is not None and not player_features_df.empty:
            game_features = self._add_player_features(game_features, player_features_df)

        # Add lineup features (player impact based)
        if self.use_lineup_features and self.lineup_builder is not None:
            game_features = self._add_lineup_features(game_features, games_df, injury_df)

        # Add matchup features (H2H, division, rivalry)
        if self.use_matchup_features and self.matchup_builder is not None:
            game_features = self._add_matchup_features(game_features, games_df)

        # Add Elo features
        if self.use_elo:
            game_features = self._add_elo_features(game_features, games_df)

        # Add schedule features (B2B, rest)
        game_features = self._add_schedule_features(game_features, games_df)

        # Clean up columns
        game_features = self._clean_columns(game_features)

        return game_features

    def _add_differential_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add differential features (home - away)."""
        # Find rolling stat columns
        windows = self.team_builder.windows

        for window in windows:
            suffix = f"_{window}g"

            # Rating differentials
            for stat in ["pts_for", "pts_against", "net_rating", "pace", "win_rate", "win_streak"]:
                home_col = f"home_{stat}{suffix}"
                away_col = f"away_{stat}{suffix}"
                if home_col in df.columns and away_col in df.columns:
                    df[f"diff_{stat}{suffix}"] = df[home_col] - df[away_col]

        # Rest differential
        if "home_rest_days" in df.columns and "away_rest_days" in df.columns:
            df["rest_diff"] = df["home_rest_days"] - df["away_rest_days"]

        # Travel differential
        if "home_travel_distance" in df.columns and "away_travel_distance" in df.columns:
            df["travel_diff"] = df["home_travel_distance"] - df["away_travel_distance"]

        # Season record differential (playoff implications)
        if "home_season_win_pct" in df.columns and "away_season_win_pct" in df.columns:
            df["season_win_pct_diff"] = df["home_season_win_pct"] - df["away_season_win_pct"]

        return df

    def _add_targets(
        self,
        game_features: pd.DataFrame,
        games_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add target variables for prediction."""
        # Drop any existing columns that conflict with target names
        for col in ["home_win", "point_diff", "total_points"]:
            if col in game_features.columns:
                game_features = game_features.drop(columns=[col])

        # Merge with original game data for outcomes
        targets = games_df[["game_id", "home_score", "away_score"]].copy()
        targets["home_win"] = (targets["home_score"] > targets["away_score"]).astype(int)
        targets["point_diff"] = targets["home_score"] - targets["away_score"]
        targets["total_points"] = targets["home_score"] + targets["away_score"]

        game_features = pd.merge(
            game_features,
            targets[["game_id", "home_win", "point_diff", "total_points"]],
            on="game_id",
            how="left"
        )

        return game_features

    def _add_odds_features(
        self,
        game_features: pd.DataFrame,
        odds_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add betting odds as features."""
        # Aggregate odds by game (use average across books)
        odds_agg = (
            odds_df
            .groupby(["game_id", "market", "team"])
            .agg({
                "odds": "mean",
                "implied_prob": "mean",
                "line": "first",
            })
            .reset_index()
        )

        # Pivot to get odds columns
        for market in ["moneyline", "spread", "total"]:
            market_odds = odds_agg[odds_agg["market"] == market].copy()

            if market == "moneyline":
                # Home and away moneyline odds
                home_ml = market_odds[market_odds["team"] == "home"][["game_id", "odds", "implied_prob"]]
                home_ml.columns = ["game_id", "home_ml_odds", "home_ml_implied"]

                away_ml = market_odds[market_odds["team"] == "away"][["game_id", "odds", "implied_prob"]]
                away_ml.columns = ["game_id", "away_ml_odds", "away_ml_implied"]

                game_features = game_features.merge(home_ml, on="game_id", how="left")
                game_features = game_features.merge(away_ml, on="game_id", how="left")

            elif market == "spread":
                # Home spread
                home_spread = market_odds[market_odds["team"] == "home"][["game_id", "line", "odds"]]
                home_spread.columns = ["game_id", "spread", "spread_odds"]

                game_features = game_features.merge(home_spread, on="game_id", how="left")

            elif market == "total":
                # Over/under total
                over_total = market_odds[market_odds["team"] == "over"][["game_id", "line", "odds"]]
                over_total.columns = ["game_id", "total_line", "over_odds"]

                game_features = game_features.merge(over_total, on="game_id", how="left")

        return game_features

    def _add_player_features(
        self,
        game_features: pd.DataFrame,
        player_features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge player-aggregated features with game features."""
        # Player features should have columns like:
        # game_id, home_player_*, away_player_*, diff_player_*

        # Only keep columns that don't already exist
        existing_cols = set(game_features.columns)
        new_cols = ["game_id"] + [
            c for c in player_features_df.columns
            if c not in existing_cols and c != "game_id"
        ]

        if len(new_cols) <= 1:  # Only game_id
            logger.warning("No new player features to add")
            return game_features

        player_subset = player_features_df[new_cols].copy()

        game_features = game_features.merge(
            player_subset,
            on="game_id",
            how="left"
        )

        logger.info(f"Added {len(new_cols) - 1} player features")
        return game_features

    def _add_lineup_features(
        self,
        game_features: pd.DataFrame,
        games_df: pd.DataFrame,
        injury_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Add lineup-based features using player impact model.

        These features capture:
        - Total team strength based on expected lineups
        - Impact of injured players
        - Lineup impact differential between teams
        """
        if self.lineup_builder is None:
            return game_features

        # Build lineup features for each game
        lineup_features_list = []

        for _, game in games_df.iterrows():
            game_id = game.get('game_id')
            home_team_id = game.get('home_team_id')
            away_team_id = game.get('away_team_id')

            # Get injuries for this game
            home_injured = []
            away_injured = []

            if injury_df is not None and 'game_id' in injury_df.columns:
                if game_id in injury_df['game_id'].values:
                    game_injuries = injury_df[
                        (injury_df['game_id'] == game_id) &
                        (injury_df['is_out'] == True)
                    ]

                    home_inj = game_injuries[game_injuries['team_id'] == home_team_id]
                    away_inj = game_injuries[game_injuries['team_id'] == away_team_id]

                    home_injured = home_inj['player_id'].tolist() if 'player_id' in home_inj.columns else []
                    away_injured = away_inj['player_id'].tolist() if 'player_id' in away_inj.columns else []

            # Build features using the lineup builder
            features = self.lineup_builder.build_game_features(
                home_team_id, away_team_id,
                home_injured, away_injured
            )
            features['game_id'] = game_id
            lineup_features_list.append(features)

        if not lineup_features_list:
            return game_features

        lineup_df = pd.DataFrame(lineup_features_list)

        # Merge with game features
        game_features = game_features.merge(lineup_df, on='game_id', how='left')

        # Fill any NaN lineup features with 0
        lineup_cols = [c for c in lineup_df.columns if c != 'game_id']
        for col in lineup_cols:
            if col in game_features.columns:
                game_features[col] = game_features[col].fillna(0)

        n_with_features = (game_features['lineup_impact_diff'].abs() > 0).sum()
        logger.info(f"Added lineup features for {n_with_features} games")

        return game_features

    def _add_matchup_features(
        self,
        game_features: pd.DataFrame,
        games_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add matchup-specific features using MatchupFeatureBuilder.

        These features capture:
        - Head-to-head historical performance
        - Division/conference/rivalry indicators
        - Style matchup interactions
        """
        if self.matchup_builder is None:
            return game_features

        try:
            # Build matchup features for all games
            matchup_features = self.matchup_builder.build_all_matchup_features(games_df)

            # Merge with game features
            game_features = game_features.merge(matchup_features, on="game_id", how="left")

            # Fill NaN values for matchup features with defaults
            matchup_cols = [c for c in matchup_features.columns if c != "game_id"]
            for col in matchup_cols:
                if col in game_features.columns:
                    if col in ["h2h_home_win_rate"]:
                        game_features[col] = game_features[col].fillna(0.5)
                    elif col in ["h2h_total_points"]:
                        game_features[col] = game_features[col].fillna(215.0)
                    else:
                        game_features[col] = game_features[col].fillna(0)

            logger.info(f"Added {len(matchup_cols)} matchup features")

        except Exception as e:
            logger.warning(f"Could not add matchup features: {e}")

        return game_features

    def _add_elo_features(
        self,
        game_features: pd.DataFrame,
        games_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add Elo rating features to game data."""
        # Calculate Elo for all games (includes pre-game ratings)
        games_with_elo = self.elo_system.calculate_historical_elo(games_df)

        # Extract Elo columns we need
        elo_cols = ["game_id", "home_elo", "away_elo", "elo_diff", "elo_prob"]
        elo_features = games_with_elo[elo_cols].copy()

        # Merge with game features
        game_features = game_features.merge(elo_features, on="game_id", how="left")

        logger.info(f"Added Elo features (diff range: {elo_features['elo_diff'].min():.0f} to {elo_features['elo_diff'].max():.0f})")
        return game_features

    def _add_schedule_features(
        self,
        game_features: pd.DataFrame,
        games_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add schedule-related features (back-to-back, rest days)."""
        # Columns we want from games_df
        schedule_cols = ["game_id"]

        # B2B columns
        if "home_b2b" in games_df.columns:
            schedule_cols.append("home_b2b")
        if "away_b2b" in games_df.columns:
            schedule_cols.append("away_b2b")
        if "b2b_advantage" in games_df.columns:
            schedule_cols.append("b2b_advantage")

        # Rest columns (if not already present)
        if "home_rest" in games_df.columns and "home_rest_days" not in game_features.columns:
            schedule_cols.append("home_rest")
        if "away_rest" in games_df.columns and "away_rest_days" not in game_features.columns:
            schedule_cols.append("away_rest")
        if "rest_advantage" in games_df.columns:
            schedule_cols.append("rest_advantage")

        # Only merge if we have schedule columns to add
        if len(schedule_cols) > 1:
            schedule_features = games_df[schedule_cols].copy()

            # Convert boolean B2B to int (fill NA first)
            for col in ["home_b2b", "away_b2b"]:
                if col in schedule_features.columns:
                    schedule_features[col] = schedule_features[col].fillna(False).astype(int)

            game_features = game_features.merge(schedule_features, on="game_id", how="left")

            # Fill any NaN B2B values with 0 (not on B2B)
            for col in ["home_b2b", "away_b2b", "b2b_advantage"]:
                if col in game_features.columns:
                    game_features[col] = game_features[col].fillna(0).astype(int)

            logger.info(f"Added schedule features: {[c for c in schedule_cols if c != 'game_id']}")

        return game_features

    def _clean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove unnecessary columns and reorder."""
        # Drop duplicate/unnecessary columns
        drop_patterns = ["_drop", "opponent", "points_for", "points_against"]

        for pattern in drop_patterns:
            cols_to_drop = [c for c in df.columns if pattern in c]
            df = df.drop(columns=cols_to_drop, errors="ignore")

        # Reorder columns
        id_cols = ["game_id", "date", "season", "home_team", "away_team"]
        target_cols = ["home_win", "point_diff", "total_points"]
        odds_cols = [c for c in df.columns if any(x in c for x in ["ml_", "spread", "total_line", "over_odds"])]

        feature_cols = [c for c in df.columns if c not in id_cols + target_cols + odds_cols]

        ordered_cols = id_cols + feature_cols + odds_cols + target_cols
        ordered_cols = [c for c in ordered_cols if c in df.columns]

        return df[ordered_cols]

    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """Get list of feature columns for modeling."""
        # Exact column names to exclude (targets and identifiers)
        exclude_exact = {
            "game_id", "date", "season", "home_team", "away_team",
            "home_win", "point_diff", "total_points",
            "home_score", "away_score",
            # Raw per-game stats that would leak info
            "home_is_home", "away_is_home",
            "home_point_diff", "away_point_diff",
            "home_total_points", "away_total_points",
            "home_points_for", "away_points_for",
            "home_points_against", "away_points_against",
            "home_win", "away_win",
        }

        # Patterns to exclude (substring match)
        exclude_patterns = ["opponent", "team_x", "team_y"]

        feature_cols = [
            col for col in df.columns
            if col not in exclude_exact
            and not any(pattern in col.lower() for pattern in exclude_patterns)
        ]

        return feature_cols

    def prepare_train_test_split(
        self,
        df: pd.DataFrame,
        train_seasons: list,
        val_seasons: list,
        test_seasons: list,
    ) -> tuple:
        """
        Split data by season for proper temporal validation.

        Args:
            df: DataFrame with game features
            train_seasons: Seasons for training (e.g., [2015, 2016, ..., 2021])
            val_seasons: Seasons for validation (e.g., [2022])
            test_seasons: Seasons for testing (e.g., [2023])

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        train_df = df[df["season"].isin(train_seasons)].copy()
        val_df = df[df["season"].isin(val_seasons)].copy()
        test_df = df[df["season"].isin(test_seasons)].copy()

        logger.info(f"Train: {len(train_df)} games ({train_seasons[0]}-{train_seasons[-1]})")
        logger.info(f"Val: {len(val_df)} games ({val_seasons})")
        logger.info(f"Test: {len(test_df)} games ({test_seasons})")

        return train_df, val_df, test_df


def build_modeling_dataset(
    games_path: str = "data/raw/games.parquet",
    odds_path: str = "data/raw/odds.parquet",
    player_features_path: str = "data/features/player_features.parquet",
    output_path: str = "data/features/game_features.parquet",
) -> pd.DataFrame:
    """
    Build complete modeling dataset from raw data.

    Args:
        games_path: Path to games data
        odds_path: Path to odds data (optional)
        player_features_path: Path to player features (optional)
        output_path: Path to save features

    Returns:
        DataFrame with all features
    """
    # Load data
    games_df = pd.read_parquet(games_path)
    logger.info(f"Loaded {len(games_df)} games")

    odds_df = None
    if os.path.exists(odds_path):
        odds_df = pd.read_parquet(odds_path)
        logger.info(f"Loaded {len(odds_df)} odds records")

    player_features_df = None
    if os.path.exists(player_features_path):
        player_features_df = pd.read_parquet(player_features_path)
        logger.info(f"Loaded {len(player_features_df)} player feature records")

    # Build features
    builder = GameFeatureBuilder()
    features = builder.build_game_features(games_df, odds_df, player_features_df)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features.to_parquet(output_path)
    logger.info(f"Saved {len(features)} game features to {output_path}")

    return features


if __name__ == "__main__":
    # Example usage
    import pandas as pd

    # Create sample data
    games = pd.DataFrame({
        "game_id": list(range(1, 21)),
        "date": pd.date_range("2024-01-01", periods=20, freq="D"),
        "season": [2023] * 20,
        "home_team": ["LAL", "BOS", "LAL", "DEN", "LAL"] * 4,
        "away_team": ["BOS", "LAL", "DEN", "LAL", "GSW"] * 4,
        "home_score": [110, 105, 115, 120, 108] * 4,
        "away_score": [105, 108, 110, 105, 112] * 4,
    })

    builder = GameFeatureBuilder()
    features = builder.build_game_features(games)

    print("Game features shape:", features.shape)
    print("\nFeature columns:")
    print(builder.get_feature_columns(features))
    print("\nSample:")
    print(features.head())
