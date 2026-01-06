"""
Advanced Player Feature Engineering

Sophisticated features for player props models using FREE data sources:
- NBA Stats API (free)
- ESPN API (free)
- Basketball Reference (free, scraping)
- Historical box scores (we have)

Target: 80+ features matching spread model sophistication
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from loguru import logger


def calculate_exponentially_weighted_stats(
    player_df: pd.DataFrame,
    stats: List[str] = ['pts', 'reb', 'ast', 'fg3m'],
    spans: List[int] = [3, 5, 10]
) -> pd.DataFrame:
    """
    Calculate exponentially weighted moving averages (recent games matter more).

    Args:
        player_df: Player box scores with game_date
        stats: Statistics to calculate EWMA for
        spans: Span values (half-life in games)

    Returns:
        DataFrame with EWMA columns added
    """
    logger.info("Calculating exponentially weighted moving averages...")

    df = player_df.sort_values(['player_id', 'game_date']).copy()

    for stat in stats:
        if stat not in df.columns:
            logger.warning(f"Stat '{stat}' not found, skipping")
            continue

        for span in spans:
            col_name = f'{stat}_ewma_{span}'
            df[col_name] = (
                df.groupby('player_id')[stat]
                .transform(lambda x: x.ewm(span=span, adjust=False).mean().shift(1))
            )

    logger.info(f"  Added EWMA for {len(stats)} stats x {len(spans)} spans")
    return df


def calculate_performance_trends(
    player_df: pd.DataFrame,
    stats: List[str] = ['pts', 'reb', 'ast'],
    window: int = 5
) -> pd.DataFrame:
    """
    Calculate performance trends (is player improving or declining?).

    Uses linear regression slope over last N games.

    Args:
        player_df: Player box scores
        stats: Statistics to calculate trends for
        window: Number of games to analyze

    Returns:
        DataFrame with trend columns
    """
    logger.info(f"Calculating performance trends (window={window})...")

    df = player_df.copy()

    for stat in stats:
        if stat not in df.columns:
            continue

        col_name = f'{stat}_trend_{window}g'

        def calculate_slope(series):
            """Calculate slope of linear regression."""
            if len(series) < 2:
                return 0.0

            x = np.arange(len(series))
            y = series.values

            # Remove NaN
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                return 0.0

            x, y = x[mask], y[mask]

            # Linear regression
            slope = np.polyfit(x, y, 1)[0]
            return slope

        df[col_name] = (
            df.groupby('player_id')[stat]
            .transform(lambda x: x.rolling(window).apply(calculate_slope).shift(1))
        )

    logger.info(f"  Added trends for {len(stats)} stats")
    return df


def calculate_consistency_metrics(
    player_df: pd.DataFrame,
    stats: List[str] = ['pts', 'reb', 'ast'],
    window: int = 10
) -> pd.DataFrame:
    """
    Calculate performance consistency (standard deviation, coefficient of variation).

    Lower variance = more consistent (safer bet).

    Args:
        player_df: Player box scores
        stats: Statistics to calculate consistency for
        window: Number of games to analyze

    Returns:
        DataFrame with consistency metrics
    """
    logger.info(f"Calculating consistency metrics (window={window})...")

    df = player_df.copy()

    for stat in stats:
        if stat not in df.columns:
            continue

        # Standard deviation
        df[f'{stat}_std_{window}g'] = (
            df.groupby('player_id')[stat]
            .transform(lambda x: x.rolling(window).std().shift(1))
        )

        # Coefficient of variation (std / mean)
        # Use expanding mean if rolling mean doesn't exist
        if f'{stat}_roll{window}' in df.columns:
            df[f'{stat}_cv_{window}g'] = (
                df[f'{stat}_std_{window}g'] / df[f'{stat}_roll{window}'].replace(0, np.nan)
            )
        else:
            # Calculate rolling mean if it doesn't exist
            df[f'{stat}_roll{window}'] = (
                df.groupby('player_id')[stat]
                .transform(lambda x: x.rolling(window).mean().shift(1))
            )
            df[f'{stat}_cv_{window}g'] = (
                df[f'{stat}_std_{window}g'] / df[f'{stat}_roll{window}'].replace(0, np.nan)
            )

    logger.info(f"  Added consistency metrics for {len(stats)} stats")
    return df


def calculate_home_away_splits(
    player_df: pd.DataFrame,
    stats: List[str] = ['pts', 'reb', 'ast', 'fg3m']
) -> pd.DataFrame:
    """
    Calculate home vs away performance splits.

    Some players perform significantly better at home.

    Args:
        player_df: Player box scores with is_home column
        stats: Statistics to calculate splits for

    Returns:
        DataFrame with home/away split columns
    """
    logger.info("Calculating home/away splits...")

    df = player_df.copy()

    if 'is_home' not in df.columns:
        logger.warning("No 'is_home' column, skipping home/away splits")
        return df

    for stat in stats:
        if stat not in df.columns:
            continue

        # Home average
        home_mask = df['is_home'] == 1
        df[f'{stat}_home_avg'] = (
            df[home_mask].groupby('player_id')[stat]
            .transform(lambda x: x.expanding().mean().shift(1))
        )

        # Away average
        away_mask = df['is_home'] == 0
        df[f'{stat}_away_avg'] = (
            df[away_mask].groupby('player_id')[stat]
            .transform(lambda x: x.expanding().mean().shift(1))
        )

        # Fill NaN (for games where player hasn't played home/away yet)
        df[f'{stat}_home_avg'] = df[f'{stat}_home_avg'].fillna(df[f'{stat}_roll10'])
        df[f'{stat}_away_avg'] = df[f'{stat}_away_avg'].fillna(df[f'{stat}_roll10'])

        # Home advantage (positive = performs better at home)
        df[f'{stat}_home_advantage'] = df[f'{stat}_home_avg'] - df[f'{stat}_away_avg']

    logger.info(f"  Added home/away splits for {len(stats)} stats")
    return df


def calculate_player_vs_team_history(
    player_df: pd.DataFrame,
    stats: List[str] = ['pts', 'reb', 'ast'],
    min_games: int = 3
) -> pd.DataFrame:
    """
    Calculate player's historical performance vs each opponent.

    Example: Giannis averages 28.5 PPG vs LAL (career)

    Args:
        player_df: Player box scores with opponent_team column
        stats: Statistics to calculate matchup history for
        min_games: Minimum games to trust the average

    Returns:
        DataFrame with player-vs-team historical averages
    """
    logger.info("Calculating player vs team matchup history...")

    df = player_df.copy()

    if 'opponent_team' not in df.columns:
        logger.warning("No 'opponent_team' column found")
        # Try to infer from home/away teams
        if 'home_team' in df.columns and 'away_team' in df.columns and 'team_abbreviation' in df.columns:
            df['opponent_team'] = df.apply(
                lambda row: row['away_team'] if row['team_abbreviation'] == row['home_team'] else row['home_team'],
                axis=1
            )
        else:
            logger.error("Cannot calculate matchup history without opponent_team")
            return df

    for stat in stats:
        if stat not in df.columns:
            continue

        # Career average vs each opponent
        df[f'{stat}_vs_opp_career'] = (
            df.groupby(['player_id', 'opponent_team'])[stat]
            .transform(lambda x: x.expanding().mean().shift(1))
        )

        # Last 3 games vs this opponent
        df[f'{stat}_vs_opp_last3'] = (
            df.groupby(['player_id', 'opponent_team'])[stat]
            .transform(lambda x: x.rolling(window=3, min_periods=1).mean().shift(1))
        )

        # Sample size (games played vs this opponent)
        df[f'games_vs_opp'] = (
            df.groupby(['player_id', 'opponent_team']).cumcount()
        )

        # Confidence weight (0-1, higher when more games played)
        df[f'{stat}_vs_opp_weight'] = np.minimum(df['games_vs_opp'] / min_games, 1.0)

        # Weighted blend (career vs this opponent + overall average)
        df[f'{stat}_vs_opp_weighted'] = (
            df[f'{stat}_vs_opp_weight'] * df[f'{stat}_vs_opp_career'] +
            (1 - df[f'{stat}_vs_opp_weight']) * df[f'{stat}_roll10']
        )

    logger.info(f"  Added player-vs-team history for {len(stats)} stats")
    return df


def calculate_usage_patterns(
    player_df: pd.DataFrame,
    window: int = 5
) -> pd.DataFrame:
    """
    Calculate usage patterns and role indicators.

    Args:
        player_df: Player box scores
        window: Rolling window for trends

    Returns:
        DataFrame with usage pattern features
    """
    logger.info("Calculating usage patterns...")

    df = player_df.copy()

    # Usage rate trend (is player's role increasing?)
    if 'usage_rate' in df.columns:
        df['usage_trend'] = (
            df.groupby('player_id')['usage_rate']
            .transform(lambda x: x.diff(window).shift(1))
        )

    # Shot distribution (% of shots from 3PT)
    if 'fg3a' in df.columns and 'fga' in df.columns:
        df['three_point_rate'] = (df['fg3a'] / df['fga'].replace(0, np.nan)).fillna(0)
        df[f'three_point_rate_roll{window}'] = (
            df.groupby('player_id')['three_point_rate']
            .transform(lambda x: x.rolling(window).mean().shift(1))
        )

    # Playmaking role (AST:TOV ratio)
    if 'ast' in df.columns and 'tov' in df.columns:
        df['ast_to_tov'] = (df['ast'] / df['tov'].replace(0, 1))
        df['ast_to_tov_roll5'] = (
            df.groupby('player_id')['ast_to_tov']
            .transform(lambda x: x.rolling(5).mean().shift(1))
        )

    # Scoring efficiency (points per shot attempt)
    if 'pts' in df.columns and 'fga' in df.columns:
        df['pts_per_fga'] = (df['pts'] / df['fga'].replace(0, np.nan)).fillna(0)
        df['pts_per_fga_roll5'] = (
            df.groupby('player_id')['pts_per_fga']
            .transform(lambda x: x.rolling(5).mean().shift(1))
        )

    # Rebounding role (reb per minute)
    if 'reb' in df.columns and 'min' in df.columns:
        df['reb_per_min'] = (df['reb'] / df['min'].replace(0, np.nan)).fillna(0)

    logger.info("  Added usage pattern features")
    return df


def calculate_minute_load_features(
    player_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate minute load and fatigue indicators.

    Args:
        player_df: Player box scores with min column

    Returns:
        DataFrame with minute load features
    """
    logger.info("Calculating minute load features...")

    df = player_df.copy()

    if 'min' not in df.columns:
        logger.warning("No 'min' column found")
        return df

    # Minutes last 3 games (fatigue indicator)
    df['min_last_3g'] = (
        df.groupby('player_id')['min']
        .transform(lambda x: x.rolling(3).sum().shift(1))
    )

    # Minutes above/below season average
    df['min_season_avg'] = (
        df.groupby('player_id')['min']
        .transform('mean')
    )
    df['min_above_avg'] = df['min_roll5'] - df['min_season_avg']

    # Consecutive high-minute games (fatigue risk)
    high_min_threshold = 34  # minutes
    df['is_high_min_game'] = (df['min'] > high_min_threshold).astype(int)
    df['consecutive_high_min'] = (
        df.groupby('player_id')['is_high_min_game']
        .transform(lambda x: x.rolling(window=10, min_periods=1).sum().shift(1))
    )

    logger.info("  Added minute load features")
    return df


def calculate_game_context_features(
    player_df: pd.DataFrame,
    games_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Calculate game context features (revenge game, milestone, etc.).

    Args:
        player_df: Player box scores
        games_df: Game-level data (optional)

    Returns:
        DataFrame with context features
    """
    logger.info("Calculating game context features...")

    df = player_df.copy()

    # Time of season (0-1, where 0 = start, 1 = end)
    if 'game_date' in df.columns:
        df['day_of_season'] = (
            df.groupby('player_id').cumcount() + 1
        )
        df['season_progress'] = df['day_of_season'] / 82.0

    # Days since last game (rest)
    if 'game_date' in df.columns:
        df['game_date_dt'] = pd.to_datetime(df['game_date'])
        df['days_since_last'] = (
            df.groupby('player_id')['game_date_dt']
            .diff()
            .dt.days
            .fillna(3)  # Assume 3 days for first game
        )

    # Back-to-back indicator
    df['is_b2b'] = (df['days_since_last'] <= 1).astype(int)

    # Performance on back-to-backs
    if 'pts' in df.columns:
        b2b_mask = df['is_b2b'] == 1
        df['pts_on_b2b_avg'] = (
            df[b2b_mask].groupby('player_id')['pts']
            .transform(lambda x: x.expanding().mean().shift(1))
        )

    logger.info("  Added game context features")
    return df


def build_advanced_player_features(
    player_box_scores: pd.DataFrame,
    games_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Build complete advanced feature set for player props.

    Target: 80+ features using FREE data sources only.

    Args:
        player_box_scores: Raw player box scores
        games_df: Game-level data (optional)

    Returns:
        DataFrame with 80+ features per player-game
    """
    logger.info("=" * 80)
    logger.info("Building Advanced Player Features (FREE DATA ONLY)")
    logger.info("=" * 80)

    df = player_box_scores.copy()

    # Ensure sorted by date
    df = df.sort_values(['player_id', 'game_date'])

    # 1. Exponentially weighted moving averages
    df = calculate_exponentially_weighted_stats(df)

    # 2. Performance trends
    df = calculate_performance_trends(df)

    # 3. Consistency metrics
    df = calculate_consistency_metrics(df)

    # 4. Home/Away splits
    df = calculate_home_away_splits(df)

    # 5. Player vs team matchup history
    df = calculate_player_vs_team_history(df)

    # 6. Usage patterns
    df = calculate_usage_patterns(df)

    # 7. Minute load features
    df = calculate_minute_load_features(df)

    # 8. Game context
    df = calculate_game_context_features(df, games_df)

    # Count features
    feature_cols = [c for c in df.columns if any([
        '_ewma_' in c,
        '_trend_' in c,
        '_std_' in c,
        '_cv_' in c,
        '_home_' in c,
        '_away_' in c,
        '_vs_opp_' in c,
        'usage_' in c,
        'min_' in c,
        'season_' in c,
        'is_b2b' in c,
        'days_since' in c
    ])]

    logger.info("")
    logger.success(f"✅ Advanced features created!")
    logger.info(f"  Total features: {len(df.columns)}")
    logger.info(f"  New advanced features: {len(feature_cols)}")
    logger.info(f"  Player-games: {len(df)}")
    logger.info("")
    logger.info("Feature categories:")
    logger.info(f"  - EWMA features: {len([c for c in feature_cols if '_ewma_' in c])}")
    logger.info(f"  - Trend features: {len([c for c in feature_cols if '_trend_' in c])}")
    logger.info(f"  - Consistency: {len([c for c in feature_cols if '_std_' in c or '_cv_' in c])}")
    logger.info(f"  - Home/Away: {len([c for c in feature_cols if '_home_' in c or '_away_' in c])}")
    logger.info(f"  - Matchup history: {len([c for c in feature_cols if '_vs_opp_' in c])}")
    logger.info(f"  - Usage patterns: {len([c for c in feature_cols if 'usage_' in c or 'ast_to_tov' in c])}")
    logger.info(f"  - Minute load: {len([c for c in feature_cols if 'min_' in c])}")
    logger.info(f"  - Context: {len([c for c in feature_cols if 'season_' in c or 'is_b2b' in c])}")

    return df
