"""
Player Feature Engineering

Fetches player-level statistics using nba_api and aggregates them to team-level
features for improved game predictions.
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# nba_api endpoints
try:
    from nba_api.stats.endpoints import (
        leaguegamefinder,
        boxscoretraditionalv2,
        playergamelog,
        commonteamroster,
    )
    from nba_api.stats.static import teams
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False
    logger.warning("nba_api not installed. Run: pip install nba_api")


# Team abbreviation mappings (nba_api uses different abbreviations sometimes)
TEAM_ABBREV_MAP = {
    "PHX": "PHO",  # Phoenix
    "BKN": "BRK",  # Brooklyn
    "CHA": "CHO",  # Charlotte (older games)
    "NOP": "NOH",  # New Orleans (older games)
}

# Reverse mapping
TEAM_ABBREV_REVERSE = {v: k for k, v in TEAM_ABBREV_MAP.items()}


class PlayerStatsClient:
    """Fetches player statistics from nba_api."""

    def __init__(self, cache_dir: str = "data/cache/player_stats"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_player_box_scores(
        self,
        game_id: str,
        delay: float = 0.6,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch player box scores for a specific game.

        Args:
            game_id: NBA game ID (e.g., "0022400001")
            delay: Delay between API calls to avoid rate limiting

        Returns:
            DataFrame with player stats or None if error
        """
        if not NBA_API_AVAILABLE:
            logger.error("nba_api not available")
            return None

        try:
            time.sleep(delay)  # Rate limiting

            box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
            players_df = box.player_stats.get_data_frame()

            if players_df.empty:
                return None

            # Standardize column names
            players_df.columns = players_df.columns.str.lower()

            return players_df

        except Exception as e:
            logger.debug(f"Error fetching box score for {game_id}: {e}")
            return None

    def get_season_player_stats(
        self,
        season: int,
        season_type: str = "Regular Season",
        delay: float = 0.6,
    ) -> pd.DataFrame:
        """
        Fetch all player game logs for a season.

        Args:
            season: Season year (e.g., 2023 for 2023-24 season)
            season_type: "Regular Season" or "Playoffs"
            delay: Delay between API calls

        Returns:
            DataFrame with player game logs
        """
        cache_path = os.path.join(
            self.cache_dir,
            f"player_logs_{season}_{season_type.replace(' ', '_')}.parquet"
        )

        # Check cache
        if os.path.exists(cache_path):
            logger.info(f"Loading cached player logs for {season}")
            return pd.read_parquet(cache_path)

        if not NBA_API_AVAILABLE:
            logger.error("nba_api not available")
            return pd.DataFrame()

        logger.info(f"Fetching player logs for {season} season...")

        # Get all teams
        all_teams = teams.get_teams()
        all_player_logs = []

        for team in all_teams:
            team_id = team["id"]
            team_abbrev = team["abbreviation"]

            try:
                time.sleep(delay)

                # Get team roster
                roster = commonteamroster.CommonTeamRoster(
                    team_id=team_id,
                    season=f"{season}-{str(season+1)[-2:]}"
                )
                roster_df = roster.common_team_roster.get_data_frame()

                # Get game logs for each player
                for _, player in roster_df.iterrows():
                    player_id = player["PLAYER_ID"]

                    try:
                        time.sleep(delay)

                        logs = playergamelog.PlayerGameLog(
                            player_id=player_id,
                            season=f"{season}-{str(season+1)[-2:]}",
                            season_type_all_star=season_type,
                        )
                        log_df = logs.player_game_log.get_data_frame()

                        if not log_df.empty:
                            log_df["TEAM_ABBREVIATION"] = team_abbrev
                            all_player_logs.append(log_df)

                    except Exception as e:
                        logger.debug(f"Error fetching logs for player {player_id}: {e}")
                        continue

                logger.info(f"  Fetched {team_abbrev} player logs")

            except Exception as e:
                logger.warning(f"Error fetching {team_abbrev} roster: {e}")
                continue

        if not all_player_logs:
            logger.warning("No player logs fetched")
            return pd.DataFrame()

        # Combine all logs
        df = pd.concat(all_player_logs, ignore_index=True)
        df.columns = df.columns.str.lower()

        # Save to cache
        df.to_parquet(cache_path)
        logger.info(f"Cached {len(df)} player game logs for {season}")

        return df

    def get_player_logs_from_games(
        self,
        game_ids: List[str],
        delay: float = 0.6,
        batch_size: int = 50,
    ) -> pd.DataFrame:
        """
        Fetch player box scores for a list of games.

        More efficient than getting full season logs when you only need
        specific games.

        Args:
            game_ids: List of NBA game IDs
            delay: Delay between API calls
            batch_size: Save progress every N games

        Returns:
            DataFrame with player stats for all games
        """
        cache_path = os.path.join(self.cache_dir, "player_box_scores.parquet")

        # Load existing cache
        existing_df = None
        existing_game_ids = set()
        if os.path.exists(cache_path):
            existing_df = pd.read_parquet(cache_path)
            existing_game_ids = set(existing_df["game_id"].unique())
            logger.info(f"Loaded {len(existing_game_ids)} cached games")

        # Find games that need fetching
        new_game_ids = [g for g in game_ids if g not in existing_game_ids]

        if not new_game_ids:
            logger.info("All games already cached")
            return existing_df if existing_df is not None else pd.DataFrame()

        logger.info(f"Fetching {len(new_game_ids)} new games...")

        all_box_scores = []
        for i, game_id in enumerate(new_game_ids):
            box = self.get_player_box_scores(game_id, delay=delay)

            if box is not None:
                all_box_scores.append(box)

            if (i + 1) % batch_size == 0:
                logger.info(f"  Processed {i + 1}/{len(new_game_ids)} games")

        if not all_box_scores:
            return existing_df if existing_df is not None else pd.DataFrame()

        # Combine new data
        new_df = pd.concat(all_box_scores, ignore_index=True)

        # Merge with existing
        if existing_df is not None:
            df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            df = new_df

        # Save updated cache
        df.to_parquet(cache_path)
        logger.info(f"Cached {len(df)} total player box scores")

        return df


class PlayerFeatureBuilder:
    """Builds player-level features and aggregates to team level."""

    # Rolling windows (in games)
    WINDOWS = [3, 5, 10]

    # Key player stats to track
    PLAYER_STATS = [
        "pts",      # Points
        "reb",      # Rebounds
        "ast",      # Assists
        "stl",      # Steals
        "blk",      # Blocks
        "tov",      # Turnovers
        "min",      # Minutes
        "fgm",      # Field goals made
        "fga",      # Field goals attempted
        "fg3m",     # 3-pointers made
        "fg3a",     # 3-pointers attempted
        "ftm",      # Free throws made
        "fta",      # Free throws attempted
        "plus_minus",  # Plus/minus
    ]

    def __init__(
        self,
        windows: List[int] = None,
        stats_client: PlayerStatsClient = None,
    ):
        self.windows = windows or self.WINDOWS
        self.stats_client = stats_client or PlayerStatsClient()

    def build_player_rolling_stats(
        self,
        player_logs: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate rolling statistics for each player.

        Args:
            player_logs: DataFrame with player game logs

        Returns:
            DataFrame with player rolling stats
        """
        df = player_logs.copy()

        # Ensure required columns exist
        required_cols = ["player_id", "game_date"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.warning(f"Missing columns: {missing}")
            return df

        # Sort by player and date
        df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)

        # Calculate rolling stats for each window
        for window in self.windows:
            df = self._add_player_rolling(df, window)

        return df

    def _add_player_rolling(
        self,
        df: pd.DataFrame,
        window: int,
    ) -> pd.DataFrame:
        """Add rolling stats for a given window size."""
        suffix = f"_roll{window}"
        grouped = df.groupby("player_id")

        for stat in self.PLAYER_STATS:
            if stat not in df.columns:
                continue

            # Use shift(1) to avoid data leakage
            df[f"{stat}{suffix}"] = (
                grouped[stat]
                .apply(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                .reset_index(level=0, drop=True)
            )

        # Calculate advanced stats
        if "fga" in df.columns and "fgm" in df.columns:
            fg_made = df.groupby("player_id")["fgm"].apply(
                lambda x: x.shift(1).rolling(window, min_periods=1).sum()
            ).reset_index(level=0, drop=True)
            fg_att = df.groupby("player_id")["fga"].apply(
                lambda x: x.shift(1).rolling(window, min_periods=1).sum()
            ).reset_index(level=0, drop=True)
            df[f"fg_pct{suffix}"] = fg_made / fg_att.replace(0, np.nan)

        if "fg3a" in df.columns and "fg3m" in df.columns:
            fg3_made = df.groupby("player_id")["fg3m"].apply(
                lambda x: x.shift(1).rolling(window, min_periods=1).sum()
            ).reset_index(level=0, drop=True)
            fg3_att = df.groupby("player_id")["fg3a"].apply(
                lambda x: x.shift(1).rolling(window, min_periods=1).sum()
            ).reset_index(level=0, drop=True)
            df[f"fg3_pct{suffix}"] = fg3_made / fg3_att.replace(0, np.nan)

        return df

    def aggregate_to_team(
        self,
        player_stats: pd.DataFrame,
        aggregation: str = "weighted",
    ) -> pd.DataFrame:
        """
        Aggregate player stats to team level for each game.

        Args:
            player_stats: DataFrame with player rolling stats
            aggregation: "mean", "sum", or "weighted" (by minutes)

        Returns:
            DataFrame with team-level aggregated player stats
        """
        df = player_stats.copy()

        # Get rolling stat columns
        roll_cols = [c for c in df.columns if "_roll" in c]

        if not roll_cols:
            logger.warning("No rolling stat columns found")
            return pd.DataFrame()

        # Group by game and team
        group_cols = ["game_id", "team_abbreviation"]

        if aggregation == "weighted":
            # Weight by minutes played
            if "min" not in df.columns:
                logger.warning("No minutes column for weighted aggregation, using mean")
                aggregation = "mean"
            else:
                # Convert minutes to numeric if needed
                if df["min"].dtype == object:
                    df["min"] = df["min"].apply(self._parse_minutes)

                # Calculate weighted averages
                results = []
                for (game_id, team), group in df.groupby(group_cols):
                    total_mins = group["min"].sum()
                    if total_mins == 0:
                        continue

                    row = {"game_id": game_id, "team_abbreviation": team}

                    for col in roll_cols:
                        if col in group.columns:
                            weighted_sum = (group[col] * group["min"]).sum()
                            row[f"team_{col}"] = weighted_sum / total_mins

                    results.append(row)

                return pd.DataFrame(results)

        if aggregation == "mean":
            agg_funcs = {col: "mean" for col in roll_cols}
        else:  # sum
            agg_funcs = {col: "sum" for col in roll_cols}

        team_stats = df.groupby(group_cols).agg(agg_funcs).reset_index()

        # Rename columns
        rename_map = {col: f"team_{col}" for col in roll_cols}
        team_stats = team_stats.rename(columns=rename_map)

        return team_stats

    @staticmethod
    def _parse_minutes(min_str) -> float:
        """Parse minutes string like '32:45' to float."""
        if pd.isna(min_str):
            return 0.0
        if isinstance(min_str, (int, float)):
            return float(min_str)

        try:
            if ":" in str(min_str):
                parts = str(min_str).split(":")
                return float(parts[0]) + float(parts[1]) / 60
            return float(min_str)
        except (ValueError, IndexError):
            return 0.0

    def build_game_player_features(
        self,
        games_df: pd.DataFrame,
        player_logs: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build player-based features for games.

        Args:
            games_df: DataFrame with game data (must have game_id, home_team, away_team)
            player_logs: DataFrame with player game logs

        Returns:
            DataFrame with player features for each game (home and away)
        """
        # Calculate player rolling stats
        logger.info("Calculating player rolling stats...")
        player_stats = self.build_player_rolling_stats(player_logs)

        # Aggregate to team level
        logger.info("Aggregating to team level...")
        team_player_stats = self.aggregate_to_team(player_stats)

        if team_player_stats.empty:
            logger.warning("No team player stats generated")
            return pd.DataFrame()

        # Standardize team abbreviations
        team_player_stats["team_abbreviation"] = (
            team_player_stats["team_abbreviation"]
            .replace(TEAM_ABBREV_REVERSE)
        )

        # Get feature columns (excluding identifiers)
        feature_cols = [c for c in team_player_stats.columns
                       if c.startswith("team_")]

        # Create game-level features by joining with games to determine home/away
        game_player_features = games_df[["game_id", "home_team", "away_team"]].copy()

        # For home team stats
        home_merge = game_player_features.merge(
            team_player_stats,
            left_on=["game_id", "home_team"],
            right_on=["game_id", "team_abbreviation"],
            how="left"
        )
        # Rename feature columns for home
        for col in feature_cols:
            new_col = col.replace("team_", "home_player_")
            if col in home_merge.columns:
                home_merge = home_merge.rename(columns={col: new_col})

        # Keep only numeric feature columns (drop any string columns like team_abbreviation)
        home_cols = ["game_id"] + [
            c for c in home_merge.columns
            if c.startswith("home_player_") and pd.api.types.is_numeric_dtype(home_merge[c])
        ]
        home_merge = home_merge[home_cols].drop_duplicates()

        # For away team stats
        away_merge = game_player_features.merge(
            team_player_stats,
            left_on=["game_id", "away_team"],
            right_on=["game_id", "team_abbreviation"],
            how="left"
        )
        # Rename feature columns for away
        for col in feature_cols:
            new_col = col.replace("team_", "away_player_")
            if col in away_merge.columns:
                away_merge = away_merge.rename(columns={col: new_col})

        # Keep only numeric feature columns
        away_cols = ["game_id"] + [
            c for c in away_merge.columns
            if c.startswith("away_player_") and pd.api.types.is_numeric_dtype(away_merge[c])
        ]
        away_merge = away_merge[away_cols].drop_duplicates()

        # Combine home and away features
        game_player_features = game_player_features.merge(
            home_merge,
            on="game_id",
            how="left"
        )
        game_player_features = game_player_features.merge(
            away_merge,
            on="game_id",
            how="left"
        )

        # Calculate differential features
        game_player_features = self._add_player_differentials(game_player_features)

        logger.info(f"Built {len(game_player_features.columns) - 3} player features")

        return game_player_features

    def _add_player_differentials(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add home - away differentials for player stats."""
        home_cols = [c for c in df.columns if c.startswith("home_player_")]

        for home_col in home_cols:
            away_col = home_col.replace("home_player_", "away_player_")
            if away_col in df.columns:
                # Only calculate differentials for numeric columns
                if pd.api.types.is_numeric_dtype(df[home_col]) and pd.api.types.is_numeric_dtype(df[away_col]):
                    diff_col = home_col.replace("home_player_", "diff_player_")
                    df[diff_col] = df[home_col] - df[away_col]

        return df


def build_player_feature_dataset(
    games_path: str = "data/raw/games.parquet",
    output_path: str = "data/features/player_features.parquet",
    seasons: List[int] = None,
) -> pd.DataFrame:
    """
    Build complete player feature dataset.

    This is a slower operation that fetches player data from the NBA API.

    Args:
        games_path: Path to games data
        output_path: Path to save player features
        seasons: List of seasons to process (default: last 3 seasons)

    Returns:
        DataFrame with player features for all games
    """
    if not NBA_API_AVAILABLE:
        logger.error("nba_api not installed. Run: pip install nba_api")
        return pd.DataFrame()

    # Load games
    games_df = pd.read_parquet(games_path)
    logger.info(f"Loaded {len(games_df)} games")

    # Default to last 3 seasons
    if seasons is None:
        seasons = sorted(games_df["season"].unique())[-3:]

    logger.info(f"Processing seasons: {seasons}")

    # Initialize clients
    stats_client = PlayerStatsClient()
    feature_builder = PlayerFeatureBuilder(stats_client=stats_client)

    # Get player logs for each season
    all_player_logs = []
    for season in seasons:
        logs = stats_client.get_season_player_stats(season)
        if not logs.empty:
            logs["season"] = season
            all_player_logs.append(logs)

    if not all_player_logs:
        logger.error("Failed to fetch any player logs")
        return pd.DataFrame()

    player_logs = pd.concat(all_player_logs, ignore_index=True)
    logger.info(f"Total player logs: {len(player_logs)}")

    # Build features
    season_games = games_df[games_df["season"].isin(seasons)]
    player_features = feature_builder.build_game_player_features(
        season_games, player_logs
    )

    if not player_features.empty:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        player_features.to_parquet(output_path)
        logger.info(f"Saved player features to {output_path}")

    return player_features


if __name__ == "__main__":
    # Quick test with a few games
    print("Testing player feature builder...")

    if not NBA_API_AVAILABLE:
        print("nba_api not installed. Skipping test.")
    else:
        # Test fetching a single game
        client = PlayerStatsClient()

        # Use a known game ID (format: 002240XXXX for 2024-25 season)
        test_game_id = "0022400001"
        box = client.get_player_box_scores(test_game_id)

        if box is not None:
            print(f"\nBox score for game {test_game_id}:")
            print(f"  Players: {len(box)}")
            print(f"  Columns: {list(box.columns)}")
            print(f"\nSample row:")
            print(box.iloc[0])
        else:
            print(f"Could not fetch game {test_game_id}")
