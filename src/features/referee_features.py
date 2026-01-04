"""
Referee Feature Builder

Generates betting-relevant features from referee assignments and historical statistics.
Tracks crew tendencies for totals, pace, and home/away bias.

Optimized with batch queries and caching to avoid N+1 query problems.
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

    Optimized with:
    - Batch queries to avoid N+1 problem
    - 6-hour caching for referee stats
    """

    def __init__(self, db_path: str = None):
        """
        Initialize referee feature builder.

        Args:
            db_path: Path to database (defaults to BETS_DB_PATH)
        """
        self.client = RefereeDataClient(db_path=db_path or BETS_DB_PATH)

        # Caching for referee stats (6-hour TTL)
        self._stats_cache = {}
        self._stats_cache_time = None
        self._stats_cache_ttl = timedelta(hours=6)

        # Caching for game assignments (1-hour TTL)
        self._assignments_cache = {}
        self._assignments_cache_time = {}
        self._assignments_cache_ttl = timedelta(hours=1)

    def _get_all_referee_stats(self) -> Dict[str, Dict]:
        """
        Get all referee stats with caching.

        Loads all referee stats once and caches for 6 hours.
        Avoids N+1 queries when processing multiple games.

        Returns:
            Dictionary mapping referee name to their latest stats
        """
        now = datetime.now()

        # Check cache validity
        if self._stats_cache_time and (now - self._stats_cache_time) < self._stats_cache_ttl:
            logger.debug(f"Using cached referee stats (age: {(now - self._stats_cache_time).seconds}s)")
            return self._stats_cache

        # Refresh cache - batch query all referee stats
        logger.debug("Refreshing referee stats cache")

        try:
            # Get all referee stats in single query
            conn = self.client._get_connection()
            cursor = conn.execute("""
                SELECT
                    ref_name,
                    season,
                    games_worked,
                    avg_total_points,
                    pace_factor,
                    over_rate,
                    home_win_rate
                FROM referee_stats
                WHERE season >= ?
                ORDER BY ref_name, season DESC
            """, (datetime.now().year - 1,))  # Last 2 seasons

            # Build cache: {ref_name: {latest_stats}}
            cache = {}
            for row in cursor.fetchall():
                ref_name = row['ref_name']
                if ref_name not in cache:
                    # Store first (most recent) stats for each referee
                    cache[ref_name] = {
                        'ref_name': ref_name,
                        'season': row['season'],
                        'games_worked': row['games_worked'],
                        'avg_total_points': row['avg_total_points'] or 0.0,
                        'pace_factor': row['pace_factor'] or 1.0,
                        'over_rate': row['over_rate'] or 0.5,
                        'home_win_rate': row['home_win_rate'] or 0.5,
                    }

            conn.close()

            self._stats_cache = cache
            self._stats_cache_time = now

            logger.debug(f"Cached stats for {len(cache)} referees")
            return cache

        except Exception as e:
            logger.error(f"Failed to load referee stats: {e}")
            return {}

    def _get_batch_assignments(self, game_ids: List[str]) -> pd.DataFrame:
        """
        Get referee assignments for multiple games in batch.

        Args:
            game_ids: List of game IDs (NBA format: 10-digit numeric strings)

        Returns:
            DataFrame with assignments for all games
        """
        if not game_ids:
            return pd.DataFrame()

        # Validate game_ids to prevent SQL injection
        # NBA game IDs are 10-digit numeric strings (e.g., "0022100001")
        import re

        def _validate_game_id(game_id: str) -> bool:
            """Validate game_id format (NBA format: 10 digits)."""
            if not isinstance(game_id, str):
                return False
            # Accept 10-digit format (standard) or alphanumeric up to 20 chars (flexible)
            return bool(re.match(r'^[\w-]{1,20}$', game_id))

        if not all(_validate_game_id(gid) for gid in game_ids):
            logger.error("Invalid game_id format in batch query - must be alphanumeric/dash/underscore, max 20 chars")
            return pd.DataFrame()

        try:
            # Get assignments in single query
            conn = self.client._get_connection()

            placeholders = ','.join(['?'] * len(game_ids))
            cursor = conn.execute(f"""
                SELECT
                    game_id,
                    ref_name,
                    ref_role,
                    game_date
                FROM referee_assignments
                WHERE game_id IN ({placeholders})
            """, game_ids)

            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return pd.DataFrame()

            # Convert to DataFrame
            assignments = pd.DataFrame([dict(row) for row in rows])

            logger.debug(f"Loaded assignments for {assignments['game_id'].nunique()} games")
            return assignments

        except Exception as e:
            logger.error(f"Failed to batch load referee assignments: {e}")
            return pd.DataFrame()

    def get_game_features(self, game_id: str) -> Dict[str, float]:
        """
        Get referee features for a specific game.

        Args:
            game_id: NBA game ID

        Returns:
            Dictionary of referee features
        """
        # Check assignment cache
        if game_id in self._assignments_cache_time:
            cache_age = datetime.now() - self._assignments_cache_time[game_id]
            if cache_age < self._assignments_cache_ttl:
                return self._assignments_cache.get(game_id, self._empty_features())

        # Get referee assignments for this game
        assignments = self.client.get_referee_assignments(game_id=game_id)

        if assignments.empty:
            logger.debug(f"No referee assignments found for game {game_id}")
            features = self._empty_features()
            self._assignments_cache[game_id] = features
            self._assignments_cache_time[game_id] = datetime.now()
            return features

        # Get unique referees for this game
        ref_names = assignments["ref_name"].unique().tolist()

        # Get all referee stats (cached)
        all_stats = self._get_all_referee_stats()

        # Get statistics for each referee from cache
        crew_stats = []
        for ref_name in ref_names:
            if ref_name in all_stats:
                crew_stats.append(all_stats[ref_name])

        if not crew_stats:
            logger.debug(f"No referee stats available for game {game_id}")
            features = self._empty_features()
        else:
            # Aggregate crew statistics
            features = self._aggregate_crew_features(crew_stats)

        # Cache the result
        self._assignments_cache[game_id] = features
        self._assignments_cache_time[game_id] = datetime.now()

        return features

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

        Optimized with batch queries to avoid N+1 problem.

        Args:
            games_df: DataFrame with game information
            game_id_col: Column name for game ID

        Returns:
            DataFrame with referee features added
        """
        if games_df.empty:
            return games_df

        # Extract game IDs
        game_ids = games_df[game_id_col].dropna().unique().tolist()

        if not game_ids:
            logger.warning("No game IDs found in games_df")
            # Add empty features
            for col, default in self._empty_features().items():
                games_df[col] = default
            return games_df

        # Pre-load all referee stats (single query, cached)
        all_stats = self._get_all_referee_stats()

        # Batch load assignments for all games (single query)
        all_assignments = self._get_batch_assignments(game_ids)

        # Build features for each game
        features_list = []

        for _, game in games_df.iterrows():
            game_id = game.get(game_id_col)

            if pd.isna(game_id) or game_id not in game_ids:
                features = self._empty_features()
            else:
                # Get assignments for this specific game
                game_assignments = all_assignments[all_assignments['game_id'] == game_id]

                if game_assignments.empty:
                    features = self._empty_features()
                else:
                    # Get referee names for this game
                    ref_names = game_assignments['ref_name'].unique().tolist()

                    # Get stats from cache
                    crew_stats = []
                    for ref_name in ref_names:
                        if ref_name in all_stats:
                            crew_stats.append(all_stats[ref_name])

                    if crew_stats:
                        features = self._aggregate_crew_features(crew_stats)
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

        # Get all stats from cache
        all_stats = self._get_all_referee_stats()

        crew_info = []
        for ref_name in ref_names:
            has_stats = ref_name in all_stats

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
    print("\nFeatures without game_id (should be neutral):")
    for k, v in features.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("To get actual features, need game_id from today's games")
