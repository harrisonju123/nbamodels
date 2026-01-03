"""
Team Feature Engineering

Builds rolling team statistics for game prediction.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger


class TeamFeatureBuilder:
    """Builds rolling team performance features."""

    # Default rolling windows (in games) - reduced from [5, 10, 20] to decrease feature noise
    WINDOWS = [5, 20]

    # Season phase definitions (game number thresholds)
    SEASON_PHASES = {
        'early_season': (0, 20),      # First ~6 weeks
        'pre_allstar': (20, 55),      # December through All-Star
        'post_allstar': (55, 72),     # All-Star through regular season end
        'playoff_push': (72, 82),     # Final stretch
    }

    # All-Star break approximate dates by season (mid-February typically)
    ALL_STAR_DATES = {
        2022: "2023-02-19",  # 2022-23 season
        2023: "2024-02-18",  # 2023-24 season
        2024: "2025-02-16",  # 2024-25 season
        2025: "2026-02-15",  # 2025-26 season (estimated)
    }

    # Regular season start dates by season
    SEASON_START_DATES = {
        2022: "2022-10-18",
        2023: "2023-10-24",
        2024: "2024-10-22",
        2025: "2025-10-21",  # Estimated
    }

    # Team locations for travel distance calculation
    TEAM_LOCATIONS = {
        "ATL": (33.7573, -84.3963),   # Atlanta
        "BOS": (42.3662, -71.0621),   # Boston
        "BKN": (40.6826, -73.9754),   # Brooklyn
        "CHA": (35.2251, -80.8392),   # Charlotte
        "CHI": (41.8807, -87.6742),   # Chicago
        "CLE": (41.4965, -81.6882),   # Cleveland
        "DAL": (32.7905, -96.8103),   # Dallas
        "DEN": (39.7487, -105.0077),  # Denver
        "DET": (42.3410, -83.0551),   # Detroit
        "GSW": (37.7680, -122.3878),  # Golden State
        "HOU": (29.7508, -95.3621),   # Houston
        "IND": (39.7640, -86.1555),   # Indiana
        "LAC": (34.0430, -118.2673),  # LA Clippers
        "LAL": (34.0430, -118.2673),  # LA Lakers
        "MEM": (35.1382, -90.0505),   # Memphis
        "MIA": (25.7814, -80.1870),   # Miami
        "MIL": (43.0451, -87.9173),   # Milwaukee
        "MIN": (44.9795, -93.2760),   # Minnesota
        "NOP": (29.9490, -90.0821),   # New Orleans
        "NYK": (40.7505, -73.9934),   # New York
        "OKC": (35.4634, -97.5151),   # Oklahoma City
        "ORL": (28.5392, -81.3839),   # Orlando
        "PHI": (39.9012, -75.1720),   # Philadelphia
        "PHX": (33.4457, -112.0712),  # Phoenix
        "POR": (45.5316, -122.6668),  # Portland
        "SAC": (38.5802, -121.4997),  # Sacramento
        "SAS": (29.4270, -98.4375),   # San Antonio
        "TOR": (43.6435, -79.3791),   # Toronto
        "UTA": (40.7683, -111.9011),  # Utah
        "WAS": (38.8981, -77.0209),   # Washington
    }

    # Team altitudes (feet above sea level)
    TEAM_ALTITUDES = {
        "DEN": 5280,  # Mile High City
        "UTA": 4226,  # Salt Lake City
        "PHX": 1086,  # Phoenix
        "DAL": 430,   # Dallas
        # Most other cities are near sea level
    }

    def __init__(self, windows: List[int] = None):
        self.windows = windows or self.WINDOWS

    def build_team_rolling_stats(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build rolling team statistics from game data.

        Args:
            games_df: DataFrame with game results

        Returns:
            DataFrame with team rolling stats for each game
        """
        # Create team-game level data (each game appears twice, once per team)
        team_games = self._create_team_game_records(games_df)

        # Sort by date for proper rolling calculation
        team_games = team_games.sort_values(["team", "date"]).reset_index(drop=True)

        # Calculate rolling stats for each window
        for window in self.windows:
            team_games = self._add_rolling_stats(team_games, window)

        return team_games

    def _create_team_game_records(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Convert game-level data to team-game level records."""
        records = []

        for _, game in games_df.iterrows():
            # Home team record
            records.append({
                "game_id": game["game_id"],
                "date": game["date"],
                "season": game["season"],
                "team": game["home_team"],
                "opponent": game["away_team"],
                "is_home": 1,
                "points_for": game["home_score"],
                "points_against": game["away_score"],
                "win": 1 if game["home_score"] > game["away_score"] else 0,
            })

            # Away team record
            records.append({
                "game_id": game["game_id"],
                "date": game["date"],
                "season": game["season"],
                "team": game["away_team"],
                "opponent": game["home_team"],
                "is_home": 0,
                "points_for": game["away_score"],
                "points_against": game["home_score"],
                "win": 1 if game["away_score"] > game["home_score"] else 0,
            })

        df = pd.DataFrame(records)
        df["point_diff"] = df["points_for"] - df["points_against"]
        df["total_points"] = df["points_for"] + df["points_against"]

        return df

    def _add_rolling_stats(
        self,
        team_games: pd.DataFrame,
        window: int,
    ) -> pd.DataFrame:
        """Add rolling statistics for a given window size."""
        suffix = f"_{window}g"

        # Group by team and calculate rolling stats
        # Use shift(1) to avoid data leakage (don't include current game)
        grouped = team_games.groupby("team")

        # Win rate
        team_games[f"win_rate{suffix}"] = (
            grouped["win"]
            .apply(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )

        # Points for (offensive rating proxy)
        team_games[f"pts_for{suffix}"] = (
            grouped["points_for"]
            .apply(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )

        # Points against (defensive rating proxy)
        team_games[f"pts_against{suffix}"] = (
            grouped["points_against"]
            .apply(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )

        # Net rating (point differential)
        team_games[f"net_rating{suffix}"] = (
            grouped["point_diff"]
            .apply(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )

        # Pace proxy (total points)
        team_games[f"pace{suffix}"] = (
            grouped["total_points"]
            .apply(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )

        # Win streak
        team_games[f"win_streak{suffix}"] = (
            grouped["win"]
            .apply(lambda x: self._calculate_streak(x.shift(1), window))
            .reset_index(level=0, drop=True)
        )

        return team_games

    @staticmethod
    def _calculate_streak(series: pd.Series, max_window: int) -> pd.Series:
        """Calculate current win/loss streak."""
        streak = []
        current_streak = 0

        for val in series:
            if pd.isna(val):
                streak.append(0)
            elif val == 1:
                current_streak = max(current_streak, 0) + 1
                streak.append(current_streak)
            else:
                current_streak = min(current_streak, 0) - 1
                streak.append(current_streak)

        return pd.Series(streak, index=series.index)

    def calculate_rest_days(self, team_games: pd.DataFrame) -> pd.DataFrame:
        """Add rest days between games."""
        df = team_games.copy()
        df = df.sort_values(["team", "date"])

        # Calculate days since last game
        df["prev_game_date"] = df.groupby("team")["date"].shift(1)
        df["rest_days"] = (df["date"] - df["prev_game_date"]).dt.days

        # Cap rest days and handle season start
        df["rest_days"] = df["rest_days"].clip(upper=10).fillna(7)

        # Back-to-back flag
        df["is_b2b"] = (df["rest_days"] == 1).astype(int)

        # REMOVED: games_last_7d - correlated with rest_days and is_b2b, adds noise

        return df.drop(columns=["prev_game_date"])

    def calculate_travel_distance(
        self,
        team_games: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add travel distance between games."""
        df = team_games.copy()
        df = df.sort_values(["team", "date"])

        def get_location(row):
            if row["is_home"]:
                return self.TEAM_LOCATIONS.get(row["team"])
            else:
                return self.TEAM_LOCATIONS.get(row["opponent"])

        df["game_location"] = df.apply(get_location, axis=1)

        # Calculate distance from previous game location
        df["prev_location"] = df.groupby("team")["game_location"].shift(1)

        df["travel_distance"] = df.apply(
            lambda row: self._haversine_distance(row["prev_location"], row["game_location"])
            if pd.notna(row["prev_location"]) and pd.notna(row["game_location"])
            else 0,
            axis=1
        )

        return df.drop(columns=["game_location", "prev_location"])

    @staticmethod
    def _haversine_distance(loc1: tuple, loc2: tuple) -> float:
        """Calculate distance between two lat/lon points in miles."""
        if loc1 is None or loc2 is None:
            return 0

        lat1, lon1 = loc1
        lat2, lon2 = loc2

        R = 3959  # Earth's radius in miles

        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c

    def add_altitude_feature(self, team_games: pd.DataFrame) -> pd.DataFrame:
        """Add altitude of game location."""
        df = team_games.copy()

        def get_altitude(row):
            if row["is_home"]:
                return self.TEAM_ALTITUDES.get(row["team"], 0)
            else:
                return self.TEAM_ALTITUDES.get(row["opponent"], 0)

        df["altitude_ft"] = df.apply(get_altitude, axis=1)

        return df

    def add_season_phase_features(self, team_games: pd.DataFrame) -> pd.DataFrame:
        """
        Add simplified season phase features.

        Simplified to only output:
        - season_progress (0-1 continuous): captures timing without discrete categories
        - is_post_allstar: single binary for the real regime change (All-Star break)

        REMOVED redundant features:
        - is_early_season, is_pre_allstar, is_playoff_push (one-hot encoded phases)
        - days_since_season_start (redundant with season_progress)
        - season_phase categorical (captured by season_progress)

        Args:
            team_games: DataFrame with team-game records

        Returns:
            DataFrame with simplified season phase features
        """
        df = team_games.copy()

        # Initialize only the features we keep
        df['is_post_allstar'] = 0
        df['season_progress'] = 0.0

        for season in df['season'].unique():
            season_mask = df['season'] == season

            # Get season start date
            start_date_str = self.SEASON_START_DATES.get(season)
            if start_date_str:
                start_date = pd.to_datetime(start_date_str)
            else:
                # Estimate: late October
                start_date = pd.to_datetime(f"{season}-10-20")

            # Get All-Star date
            allstar_date_str = self.ALL_STAR_DATES.get(season)
            if allstar_date_str:
                allstar_date = pd.to_datetime(allstar_date_str)
            else:
                # Estimate: mid-February of following year
                allstar_date = pd.to_datetime(f"{season + 1}-02-15")

            dates = df.loc[season_mask, 'date']

            # Is post All-Star? (real regime change - playoff push)
            df.loc[season_mask, 'is_post_allstar'] = (dates > allstar_date).astype(int)

            # Season progress (0 to 1) - captures timing continuously
            # Typical regular season is ~170 days
            df.loc[season_mask, 'season_progress'] = (
                (dates - start_date).dt.days / 170
            ).clip(0, 1)

        logger.info(f"Added simplified season features: season_progress, is_post_allstar")

        return df

    def add_schedule_difficulty(self, team_games: pd.DataFrame) -> pd.DataFrame:
        """
        Add schedule difficulty features based on opponent strength.

        Args:
            team_games: DataFrame with team-game records

        Returns:
            DataFrame with schedule difficulty features
        """
        df = team_games.copy()

        # Calculate rolling opponent win rate (proxy for schedule difficulty)
        df = df.sort_values(['team', 'date'])

        # Get opponent's season win percentage at time of game
        # (This should already be in the data if add_season_record was called)
        if 'away_season_win_pct' not in df.columns or 'home_season_win_pct' not in df.columns:
            logger.warning("Season win pct columns not found, skipping schedule difficulty")
            return df

        # Opponent win pct based on whether we're home or away
        def get_opponent_strength(row):
            if row['is_home'] == 1:
                return row.get('away_season_win_pct', 0.5)
            else:
                return row.get('home_season_win_pct', 0.5)

        df['opponent_strength'] = df.apply(get_opponent_strength, axis=1)

        # Rolling schedule difficulty (last 10 games)
        df['schedule_difficulty_10g'] = (
            df.groupby('team')['opponent_strength']
            .apply(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
            .reset_index(level=0, drop=True)
        )

        return df

    def add_nationally_televised(self, team_games: pd.DataFrame) -> pd.DataFrame:
        """
        Add indicator for nationally televised games.

        Nationally televised games (ESPN, TNT, ABC) tend to be more
        efficiently priced as they get more attention.

        Args:
            team_games: DataFrame with team-game records

        Returns:
            DataFrame with TV indicator
        """
        df = team_games.copy()

        # Approximate national TV detection based on game time and day
        # Prime time games (7-10 PM ET) on certain days are more likely to be nationally televised
        if 'date' in df.columns:
            # Check day of week (0=Monday, 6=Sunday)
            df['day_of_week'] = df['date'].dt.dayofweek

            # TNT: Tuesday, Thursday
            # ESPN: Wednesday, Friday
            # ABC: Saturday, Sunday (marquee matchups)
            national_tv_days = [1, 2, 3, 4, 5, 6]  # Tues through Sun

            df['is_national_tv_day'] = df['day_of_week'].isin(national_tv_days).astype(int)

            # Weekend games more likely to be marquee
            df['is_weekend_game'] = df['day_of_week'].isin([5, 6]).astype(int)

        return df

    def add_four_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Four Factors features from team advanced stats.

        Four Factors are the key basketball efficiency metrics:
        - eFG% (Effective Field Goal %)
        - TOV% (Turnover Rate)
        - OREB% (Offensive Rebound %)
        - FT Rate (Free Throw Rate, approximated by TS% - eFG%)
        """
        import os
        from pathlib import Path

        adv_stats_path = Path("data/features/team_advanced_stats.parquet")
        if not adv_stats_path.exists():
            logger.warning("No advanced stats file found, skipping Four Factors")
            return df

        try:
            adv = pd.read_parquet(adv_stats_path)

            # Rename columns for merging
            adv = adv.rename(columns={
                "TEAM_ABBREVIATION": "team",
                "GAME_DATE": "date",
                "GAME_ID": "game_id",
            })

            # Select Four Factors columns (use rolling averages to avoid leakage)
            four_factors_cols = ["team", "date", "TS_PCT_10G", "TOV_PCT_10G", "OREB_PCT", "EFG_PCT"]
            available_cols = [c for c in four_factors_cols if c in adv.columns]

            if len(available_cols) < 3:
                logger.warning("Not enough Four Factors columns available")
                return df

            adv_subset = adv[available_cols].copy()

            # Merge for each team in the game
            merged = df.copy()

            # For home team
            home_adv = adv_subset.copy()
            home_adv.columns = ["team", "date"] + [f"home_{c}" for c in available_cols[2:]]
            merged = merged.merge(
                home_adv,
                left_on=["team", "date"],
                right_on=["team", "date"],
                how="left"
            )

            logger.info(f"Added Four Factors features: {[c for c in merged.columns if 'TS_PCT' in c or 'TOV_PCT' in c or 'OREB' in c or 'EFG' in c]}")
            return merged

        except Exception as e:
            logger.warning(f"Could not add Four Factors: {e}")
            return df

    def build_all_features(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build all team features from game data.

        Args:
            games_df: DataFrame with game results

        Returns:
            DataFrame with all team features
        """
        logger.info("Building team rolling stats...")
        df = self.build_team_rolling_stats(games_df)

        logger.info("Adding rest days...")
        df = self.calculate_rest_days(df)

        logger.info("Adding travel distance...")
        df = self.calculate_travel_distance(df)

        # REMOVED: altitude_ft - only 4 teams have non-zero values, adds noise
        # REMOVED: add_nationally_televised - fake proxy (Tues-Sun = 6 of 7 days)
        # REMOVED: add_schedule_difficulty - correlated with other features

        logger.info("Adding season record...")
        df = self.add_season_record(df, games_df)

        logger.info("Adding Four Factors...")
        df = self.add_four_factors(df)

        logger.info("Adding season phase features (simplified)...")
        df = self.add_season_phase_features(df)

        return df

    def add_season_record(
        self,
        team_games: pd.DataFrame,
        games_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add current season win percentage for each team."""
        df = team_games.copy()
        games_sorted = games_df.sort_values("date")

        # Track wins/losses per team per season
        team_records = {}  # (team, season) -> [wins, losses]
        win_pcts = []

        for _, game in games_sorted.iterrows():
            ht, at = game["home_team"], game["away_team"]
            season = game["season"]
            home_won = game["home_score"] > game["away_score"]

            # Initialize records
            for team in [ht, at]:
                key = (team, season)
                if key not in team_records:
                    team_records[key] = [0, 0]

            # Record current win pct BEFORE this game
            ht_key = (ht, season)
            at_key = (at, season)
            ht_w, ht_l = team_records[ht_key]
            at_w, at_l = team_records[at_key]

            # Store for both team records in team_games
            win_pcts.append({
                "game_id": game["game_id"],
                "home_team": ht,
                "away_team": at,
                "home_season_win_pct": ht_w / max(ht_w + ht_l, 1),
                "away_season_win_pct": at_w / max(at_w + at_l, 1),
            })

            # Update records
            if home_won:
                team_records[ht_key][0] += 1
                team_records[at_key][1] += 1
            else:
                team_records[ht_key][1] += 1
                team_records[at_key][0] += 1

        win_pct_df = pd.DataFrame(win_pcts)

        # Merge - only need game_id level info
        df = df.merge(
            win_pct_df[["game_id", "home_season_win_pct", "away_season_win_pct"]].drop_duplicates(),
            on="game_id",
            how="left"
        )

        return df


if __name__ == "__main__":
    # Example usage
    import pandas as pd

    # Create sample data
    games = pd.DataFrame({
        "game_id": [1, 2, 3, 4, 5],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
        "season": [2023, 2023, 2023, 2023, 2023],
        "home_team": ["LAL", "BOS", "LAL", "DEN", "LAL"],
        "away_team": ["BOS", "LAL", "DEN", "LAL", "GSW"],
        "home_score": [110, 105, 115, 120, 108],
        "away_score": [105, 108, 110, 105, 112],
    })

    builder = TeamFeatureBuilder()
    features = builder.build_all_features(games)

    print("Team features shape:", features.shape)
    print("\nSample features:")
    print(features.head(10))
