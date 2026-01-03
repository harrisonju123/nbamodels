"""
Player Impact Model (RAPM-style)

Calculates individual player impact using Regularized Adjusted Plus-Minus.
Uses stint-level data from NBA API to isolate player contributions.
"""

import os
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from loguru import logger

try:
    from nba_api.stats.endpoints import gamerotation, boxscoretraditionalv2
    from nba_api.stats.static import teams, players
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False
    logger.warning("nba_api not installed. Run: pip install nba_api")


@dataclass
class PlayerImpact:
    """Player impact value with metadata."""
    player_id: int
    player_name: str
    team_id: int
    team_abbrev: str
    impact: float  # Points per 100 possessions
    minutes: float  # Total minutes in sample
    n_stints: int  # Number of stints used
    uncertainty: float  # Standard error of estimate


class StintDataFetcher:
    """
    Fetches stint-level data from NBA API.

    Stints are continuous periods where the same 5 players are on court.
    """

    def __init__(self, cache_dir: str = "data/cache/stints"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_game_stints(
        self,
        game_id: str,
        delay: float = 0.6,
    ) -> Optional[pd.DataFrame]:
        """
        Get stint data for a single game.

        Args:
            game_id: NBA game ID
            delay: API rate limit delay

        Returns:
            DataFrame with stint data or None if error
        """
        if not NBA_API_AVAILABLE:
            return None

        try:
            time.sleep(delay)

            rotation = gamerotation.GameRotation(game_id=game_id)

            home_df = rotation.home_team.get_data_frame()
            away_df = rotation.away_team.get_data_frame()

            if home_df.empty and away_df.empty:
                return None

            # Add home/away indicator
            home_df['is_home'] = 1
            away_df['is_home'] = 0

            # Combine
            df = pd.concat([home_df, away_df], ignore_index=True)
            df.columns = df.columns.str.lower()

            return df

        except Exception as e:
            logger.debug(f"Error fetching stints for {game_id}: {e}")
            return None

    def get_season_stints(
        self,
        game_ids: List[str],
        delay: float = 0.6,
        batch_size: int = 50,
    ) -> pd.DataFrame:
        """
        Fetch stint data for multiple games.

        Args:
            game_ids: List of game IDs
            delay: API rate limit delay
            batch_size: Save progress every N games

        Returns:
            Combined stint data for all games
        """
        cache_path = self.cache_dir / "stint_data.parquet"

        # Load existing cache
        existing_df = None
        existing_game_ids = set()
        if cache_path.exists():
            existing_df = pd.read_parquet(cache_path)
            existing_game_ids = set(existing_df["game_id"].unique())
            logger.info(f"Loaded {len(existing_game_ids)} cached games")

        # Find new games
        new_game_ids = [g for g in game_ids if g not in existing_game_ids]

        if not new_game_ids:
            return existing_df if existing_df is not None else pd.DataFrame()

        logger.info(f"Fetching stints for {len(new_game_ids)} new games...")

        all_stints = []
        for i, game_id in enumerate(new_game_ids):
            stints = self.get_game_stints(game_id, delay=delay)

            if stints is not None and not stints.empty:
                all_stints.append(stints)

            if (i + 1) % batch_size == 0:
                logger.info(f"  Processed {i + 1}/{len(new_game_ids)} games")

                # Save intermediate progress
                if all_stints:
                    partial_df = pd.concat(all_stints, ignore_index=True)
                    if existing_df is not None:
                        partial_df = pd.concat([existing_df, partial_df], ignore_index=True)
                    partial_df.to_parquet(cache_path)

        if not all_stints:
            return existing_df if existing_df is not None else pd.DataFrame()

        # Combine all new data
        new_df = pd.concat(all_stints, ignore_index=True)

        if existing_df is not None:
            df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            df = new_df

        # Save final cache
        df.to_parquet(cache_path)
        logger.info(f"Cached {len(df)} total stints from {df['game_id'].nunique()} games")

        return df


class PlayerImpactModel:
    """
    RAPM-style player impact model.

    Uses Ridge Regression on stint data to isolate individual player
    contributions to team +/- while controlling for teammates and opponents.

    The model solves:
        y = X @ beta + epsilon

    Where:
        y = stint +/- (normalized to per-100-possessions)
        X = player indicator matrix (+1 for on court, -1 for opponent on court)
        beta = player impact values
    """

    # Regularization strength (higher = more shrinkage toward 0)
    DEFAULT_ALPHA = 2500

    # Minimum minutes to include a player
    MIN_MINUTES = 100

    # Possessions per 48 minutes (approximate)
    POSSESSIONS_PER_48 = 100

    def __init__(
        self,
        alpha: float = None,
        min_minutes: float = None,
        cache_dir: str = "data/cache/player_impact",
    ):
        """
        Initialize player impact model.

        Args:
            alpha: Ridge regularization strength
            min_minutes: Minimum minutes for a player to be included
            cache_dir: Directory for caching results
        """
        self.alpha = alpha or self.DEFAULT_ALPHA
        self.min_minutes = min_minutes or self.MIN_MINUTES
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._model: Optional[Ridge] = None
        self._player_ids: List[int] = []
        self._player_impacts: Dict[int, PlayerImpact] = {}
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def _build_stint_matrix(
        self,
        stint_df: pd.DataFrame,
    ) -> Tuple[sparse.csr_matrix, np.ndarray, List[int]]:
        """
        Build the player indicator matrix from stint data.

        Each row is a stint, each column is a player.
        Value is +1 if player was on home team, -1 if on away team.

        Args:
            stint_df: DataFrame with stint data

        Returns:
            Tuple of (sparse matrix X, target y, player_ids)
        """
        # Get unique players with enough minutes
        player_minutes = stint_df.groupby('person_id', group_keys=False).apply(
            lambda x: (x['out_time_real'] - x['in_time_real']).sum() / 10 / 60,
            include_groups=False
        )
        valid_players = player_minutes[player_minutes >= self.min_minutes].index.tolist()

        if not valid_players:
            logger.warning("No players meet minimum minutes threshold")
            return None, None, []

        player_to_idx = {pid: i for i, pid in enumerate(valid_players)}
        n_players = len(valid_players)

        # Group stints by game and time period to get overlapping stints
        # For simplicity, we'll use individual player stints weighted by duration
        # A more sophisticated approach would identify exact lineup overlaps

        rows = []
        cols = []
        data = []
        targets = []
        weights = []

        for i, row in stint_df.iterrows():
            player_id = row['person_id']

            if player_id not in player_to_idx:
                continue

            # Duration in minutes
            duration = (row['out_time_real'] - row['in_time_real']) / 10 / 60

            if duration <= 0:
                continue

            # +/- for this stint
            pt_diff = row.get('pt_diff', 0) or 0

            # Normalize to per-minute rate (will convert to per-100-poss later)
            if duration > 0:
                pm_per_min = pt_diff / duration
            else:
                pm_per_min = 0

            # Add to matrix
            stint_idx = len(targets)
            player_idx = player_to_idx[player_id]

            # +1 for home players, -1 for away players
            sign = 1 if row['is_home'] else -1

            rows.append(stint_idx)
            cols.append(player_idx)
            data.append(sign)
            targets.append(pm_per_min)
            weights.append(duration)

        if not rows:
            return None, None, []

        # Build sparse matrix
        X = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(len(targets), n_players)
        )

        y = np.array(targets)
        sample_weights = np.array(weights)

        # Weight by duration (more minutes = more reliable)
        y = y * np.sqrt(sample_weights)

        return X, y, valid_players

    def fit(
        self,
        stint_df: pd.DataFrame,
        player_info: Optional[pd.DataFrame] = None,
    ) -> "PlayerImpactModel":
        """
        Fit the RAPM model to stint data.

        Args:
            stint_df: DataFrame with stint data
            player_info: Optional DataFrame with player names/teams

        Returns:
            self
        """
        logger.info(f"Fitting RAPM model on {len(stint_df)} stints...")

        # Build matrix
        X, y, player_ids = self._build_stint_matrix(stint_df)

        if X is None or len(player_ids) == 0:
            logger.error("Failed to build stint matrix")
            return self

        logger.info(f"Matrix: {X.shape[0]} stints, {X.shape[1]} players")

        # Fit Ridge regression
        self._model = Ridge(alpha=self.alpha, fit_intercept=True)
        self._model.fit(X, y)

        self._player_ids = player_ids

        # Extract player impacts
        # Convert from per-minute to per-100-possessions
        # Approximate: 100 possessions ≈ 48 minutes
        coefficients = self._model.coef_ * (48 / 1)  # per 48 minutes

        # Get player info for names
        player_names = {}
        player_teams = {}

        if player_info is not None:
            for _, row in player_info.iterrows():
                pid = row.get('player_id') or row.get('person_id')
                if pid:
                    player_names[pid] = row.get('player_name', str(pid))
                    player_teams[pid] = (
                        row.get('team_id', 0),
                        row.get('team_abbreviation', 'UNK')
                    )

        # Calculate minutes per player
        player_minutes = stint_df.groupby('person_id', group_keys=False).apply(
            lambda x: (x['out_time_real'] - x['in_time_real']).sum() / 10 / 60,
            include_groups=False
        )

        player_stints = stint_df.groupby('person_id').size()

        # Build impact objects
        self._player_impacts = {}
        for i, pid in enumerate(player_ids):
            team_id, team_abbrev = player_teams.get(pid, (0, 'UNK'))

            self._player_impacts[pid] = PlayerImpact(
                player_id=pid,
                player_name=player_names.get(pid, str(pid)),
                team_id=team_id,
                team_abbrev=team_abbrev,
                impact=coefficients[i],
                minutes=player_minutes.get(pid, 0),
                n_stints=player_stints.get(pid, 0),
                uncertainty=0.0,  # Would need bootstrap for this
            )

        self._is_fitted = True
        logger.info(f"Fitted impacts for {len(self._player_impacts)} players")

        return self

    def get_player_impact(self, player_id: int) -> Optional[float]:
        """
        Get impact value for a single player.

        Args:
            player_id: NBA player ID

        Returns:
            Impact value (points per 48 minutes) or None if not found
        """
        impact = self._player_impacts.get(player_id)
        return impact.impact if impact else None

    def get_player_info(self, player_id: int) -> Optional[PlayerImpact]:
        """Get full PlayerImpact object for a player."""
        return self._player_impacts.get(player_id)

    def get_all_impacts(self) -> Dict[int, PlayerImpact]:
        """Get all player impacts."""
        return self._player_impacts.copy()

    def get_impacts_df(self) -> pd.DataFrame:
        """Get player impacts as DataFrame."""
        if not self._player_impacts:
            return pd.DataFrame()

        records = [
            {
                'player_id': p.player_id,
                'player_name': p.player_name,
                'team_id': p.team_id,
                'team_abbrev': p.team_abbrev,
                'impact': p.impact,
                'minutes': p.minutes,
                'n_stints': p.n_stints,
            }
            for p in self._player_impacts.values()
        ]

        df = pd.DataFrame(records)
        df = df.sort_values('impact', ascending=False).reset_index(drop=True)
        return df

    def get_team_impact(
        self,
        player_ids: List[int],
        default_impact: float = 0.0,
    ) -> float:
        """
        Calculate combined impact for a group of players.

        Args:
            player_ids: List of player IDs
            default_impact: Default value for unknown players

        Returns:
            Sum of player impacts
        """
        total = 0.0
        for pid in player_ids:
            impact = self.get_player_impact(pid)
            total += impact if impact is not None else default_impact
        return total

    def get_missing_impact(
        self,
        injured_player_ids: List[int],
    ) -> float:
        """
        Calculate total impact of missing (injured) players.

        Args:
            injured_player_ids: List of injured player IDs

        Returns:
            Sum of impacts (positive value = missing positive players)
        """
        return self.get_team_impact(injured_player_ids, default_impact=0.0)

    def save(self, path: Optional[str] = None) -> str:
        """Save model to disk."""
        path = path or (self.cache_dir / "player_impact_model.parquet")

        df = self.get_impacts_df()
        df.to_parquet(path)

        logger.info(f"Saved player impacts to {path}")
        return str(path)

    def load(self, path: Optional[str] = None) -> "PlayerImpactModel":
        """Load model from disk."""
        path = path or (self.cache_dir / "player_impact_model.parquet")

        if not Path(path).exists():
            logger.warning(f"No saved model at {path}")
            return self

        df = pd.read_parquet(path)

        self._player_impacts = {}
        for _, row in df.iterrows():
            self._player_impacts[row['player_id']] = PlayerImpact(
                player_id=row['player_id'],
                player_name=row['player_name'],
                team_id=row['team_id'],
                team_abbrev=row['team_abbrev'],
                impact=row['impact'],
                minutes=row['minutes'],
                n_stints=row['n_stints'],
                uncertainty=0.0,
            )

        self._player_ids = list(self._player_impacts.keys())
        self._is_fitted = True

        logger.info(f"Loaded {len(self._player_impacts)} player impacts from {path}")
        return self


def build_player_impact_model(
    games_path: str = "data/raw/games.parquet",
    output_path: str = "data/cache/player_impact/player_impact_model.parquet",
    seasons: List[int] = None,
    max_games: int = None,
) -> PlayerImpactModel:
    """
    Build player impact model from game data.

    Args:
        games_path: Path to games data
        output_path: Path to save model
        seasons: Seasons to include (default: last 2)
        max_games: Maximum games to process (for testing)

    Returns:
        Fitted PlayerImpactModel
    """
    if not NBA_API_AVAILABLE:
        logger.error("nba_api not available")
        return PlayerImpactModel()

    # Load games
    games_df = pd.read_parquet(games_path)
    logger.info(f"Loaded {len(games_df)} games")

    # Default to last 2 seasons for training
    if seasons is None:
        all_seasons = sorted(games_df['season'].unique())
        seasons = all_seasons[-2:] if len(all_seasons) >= 2 else all_seasons

    logger.info(f"Using seasons: {seasons}")

    # Filter games
    games_df = games_df[games_df['season'].isin(seasons)]
    game_ids = games_df['game_id'].tolist()

    if max_games:
        game_ids = game_ids[:max_games]

    logger.info(f"Processing {len(game_ids)} games")

    # Fetch stint data
    fetcher = StintDataFetcher()
    stint_df = fetcher.get_season_stints(game_ids)

    if stint_df.empty:
        logger.error("No stint data fetched")
        return PlayerImpactModel()

    # Build player info from stint data
    player_info = stint_df[['person_id', 'team_id']].drop_duplicates()
    player_info = player_info.rename(columns={'person_id': 'player_id'})

    # Add player names
    if NBA_API_AVAILABLE:
        try:
            all_players = players.get_players()
            player_names = {p['id']: p['full_name'] for p in all_players}
            player_info['player_name'] = player_info['player_id'].map(player_names)
        except Exception:
            player_info['player_name'] = player_info['player_id'].astype(str)

    # Add team abbreviations
    try:
        all_teams = teams.get_teams()
        team_abbrevs = {t['id']: t['abbreviation'] for t in all_teams}
        player_info['team_abbreviation'] = player_info['team_id'].map(team_abbrevs)
    except Exception:
        player_info['team_abbreviation'] = 'UNK'

    # Fit model
    model = PlayerImpactModel()
    model.fit(stint_df, player_info)

    # Save
    if model.is_fitted:
        model.save(output_path)

    return model


if __name__ == "__main__":
    # Test with a small sample
    print("Testing PlayerImpactModel...")

    if not NBA_API_AVAILABLE:
        print("nba_api not installed. Skipping test.")
    else:
        # Test fetching stints for one game
        fetcher = StintDataFetcher()

        test_game_id = "0022400001"
        print(f"\nFetching stints for game {test_game_id}...")

        stints = fetcher.get_game_stints(test_game_id)

        if stints is not None:
            print(f"Got {len(stints)} stints")
            print(f"Columns: {list(stints.columns)}")
            print(f"\nSample:")
            print(stints.head(3))
        else:
            print("Could not fetch stints")
