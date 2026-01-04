#!/usr/bin/env python3
"""
Daily Betting Pipeline - Integrated System

Combines models, market signals, line movement, and CLV optimization
for comprehensive daily bet recommendations.

Usage:
    # Paper trading mode (default)
    python scripts/daily_betting_pipeline.py

    # Live mode (real bets)
    python scripts/daily_betting_pipeline.py --live

    # Dry run (show recommendations without logging)
    python scripts/daily_betting_pipeline.py --dry-run

    # Use specific strategy
    python scripts/daily_betting_pipeline.py --strategy clv_filtered
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import argparse
from datetime import datetime, timezone
from typing import List, Dict
import pandas as pd
from loguru import logger

from src.data.odds_api import OddsAPIClient
from src.data.line_history import LineHistoryManager
from src.betting.edge_strategy import EdgeStrategy, BetSignal
from src.bet_tracker import log_bet


# Paper trading flag (global)
PAPER_TRADING = True


def get_todays_games() -> pd.DataFrame:
    """
    Get today's games with real model predictions.

    Loads trained spread model and generates pred_diff values for today's games.

    Returns:
        DataFrame with columns:
        - game_id: Unique game identifier
        - home_team: Home team abbreviation
        - away_team: Away team abbreviation
        - commence_time: Game start time (ISO format)
        - pred_diff: Model prediction (home margin)
        - home_b2b: Boolean, home team on back-to-back
        - away_b2b: Boolean, away team on back-to-back
        - rest_advantage: Home rest days - away rest days
    """
    logger.info("Generating predictions for today's NBA games...")

    try:
        import pickle
        from src.data import NBAStatsClient
        from src.features import GameFeatureBuilder
        from datetime import datetime, timedelta

        # Load trained model
        logger.info("Loading trained spread model...")
        with open("models/spread_model.pkl", "rb") as f:
            model_data = pickle.load(f)

        model = model_data["model"]
        feature_cols = model_data["feature_columns"]
        logger.info(f"Loaded model with {len(feature_cols)} features")

        # Initialize data clients
        stats_client = NBAStatsClient()
        odds_client = OddsAPIClient()
        feature_builder = GameFeatureBuilder()

        # Get recent games for feature building
        logger.info("Fetching recent games for feature generation...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        recent_games = stats_client.get_games(
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )

        if recent_games.empty:
            logger.warning("No recent games found - cannot generate features")
            return _get_games_with_placeholder_predictions()

        # Get current odds to find upcoming games
        logger.info("Fetching current odds...")
        odds = odds_client.get_current_odds()

        if odds.empty:
            logger.warning("No games with odds available")
            return pd.DataFrame()

        # Build team features from recent games
        logger.info("Building team features...")
        team_features = feature_builder.team_builder.build_all_features(recent_games)
        latest_features = (
            team_features
            .sort_values("date")
            .groupby("team")
            .last()
            .reset_index()
        )

        # Get unique upcoming games
        upcoming_games = odds[["game_id", "home_team", "away_team", "commence_time"]].drop_duplicates()

        # Build features for each game
        logger.info(f"Building features for {len(upcoming_games)} games...")
        records = []
        for _, game in upcoming_games.iterrows():
            home_team = _normalize_team_name(game["home_team"])
            away_team = _normalize_team_name(game["away_team"])

            home_stats = latest_features[latest_features["team"] == home_team]
            away_stats = latest_features[latest_features["team"] == away_team]

            if home_stats.empty or away_stats.empty:
                logger.warning(f"Missing stats for {away_team} @ {home_team}")
                continue

            # Build feature row
            record = {
                "game_id": game["game_id"],
                "commence_time": game["commence_time"],
                "home_team": game["home_team"],
                "away_team": game["away_team"],
            }

            # Add home team features
            for col in home_stats.columns:
                if col not in ["team", "game_id", "date", "season", "opponent"]:
                    record[f"home_{col}"] = home_stats[col].values[0]

            # Add away team features
            for col in away_stats.columns:
                if col not in ["team", "game_id", "date", "season", "opponent"]:
                    record[f"away_{col}"] = away_stats[col].values[0]

            records.append(record)

        features_df = pd.DataFrame(records)

        if features_df.empty:
            logger.warning("Could not build features for any games")
            return _get_games_with_placeholder_predictions()

        # Add differential features
        features_df = _add_differentials(features_df)

        # Add placeholder schedule info (B2B, rest)
        features_df['home_b2b'] = False
        features_df['away_b2b'] = False
        features_df['rest_advantage'] = features_df.get('rest_diff', 0)

        # Generate predictions
        logger.info("Generating model predictions...")
        missing_cols = [c for c in feature_cols if c not in features_df.columns]
        for col in missing_cols:
            features_df[col] = 0  # Fill missing features with 0

        X = features_df[feature_cols].fillna(0)
        features_df['pred_diff'] = model.predict(X)

        # Extract required columns
        games_df = features_df[[
            'game_id', 'home_team', 'away_team', 'commence_time',
            'pred_diff', 'home_b2b', 'away_b2b', 'rest_advantage'
        ]].copy()

        # Ensure commence_time is in ISO format
        if 'commence_time' in games_df.columns:
            games_df['commence_time'] = pd.to_datetime(games_df['commence_time']).dt.strftime('%Y-%m-%dT%H:%M:%S%z')

        logger.success(f"Generated predictions for {len(games_df)} games")
        logger.info(f"Average model prediction: {games_df['pred_diff'].mean():+.2f} pts")
        logger.info(f"Prediction range: {games_df['pred_diff'].min():+.2f} to {games_df['pred_diff'].max():+.2f} pts")

        return games_df

    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        logger.warning("Falling back to placeholder predictions")
        return _get_games_with_placeholder_predictions()

    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        logger.warning("Falling back to placeholder predictions")
        return _get_games_with_placeholder_predictions()


def _normalize_team_name(name: str) -> str:
    """Convert full team name to abbreviation."""
    name_map = {
        "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
        "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
        "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
        "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
        "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
        "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
        "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
        "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
        "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
        "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
    }
    return name_map.get(name, name)


def _add_differentials(df: pd.DataFrame) -> pd.DataFrame:
    """Add differential features between home and away teams."""
    windows = [5, 10, 20]

    for window in windows:
        suffix = f"_{window}g"
        for stat in ["pts_for", "pts_against", "net_rating", "pace", "win_rate"]:
            home_col = f"home_{stat}{suffix}"
            away_col = f"away_{stat}{suffix}"
            if home_col in df.columns and away_col in df.columns:
                df[f"diff_{stat}{suffix}"] = df[home_col] - df[away_col]

    if "home_rest_days" in df.columns and "away_rest_days" in df.columns:
        df["rest_diff"] = df["home_rest_days"] - df["away_rest_days"]

    if "home_travel_distance" in df.columns and "away_travel_distance" in df.columns:
        df["travel_diff"] = df["home_travel_distance"] - df["away_travel_distance"]

    return df


def _get_games_with_placeholder_predictions() -> pd.DataFrame:
    """
    Fallback: Get games with placeholder predictions if model loading fails.

    This is the original implementation using placeholder pred_diff = 0.0
    """
    logger.info("Fetching today's NBA games from Odds API (placeholder mode)...")

    odds_client = OddsAPIClient()

    try:
        # Fetch current games
        odds_df = odds_client.get_current_odds(markets=['spreads'])

        if odds_df.empty:
            logger.warning("No games found for today")
            return pd.DataFrame()

        # Group by game to get unique games
        games_df = odds_df.groupby('game_id').first().reset_index()

        # Extract teams from game_id or use available data
        games_df['home_team'] = games_df['home_team'] if 'home_team' in games_df.columns else 'HOME'
        games_df['away_team'] = games_df['away_team'] if 'away_team' in games_df.columns else 'AWAY'

        # Add placeholder predictions
        logger.warning("Using placeholder model predictions (pred_diff = 0.0)")
        games_df['pred_diff'] = 0.0  # Placeholder: no edge

        # Add placeholder schedule info
        games_df['home_b2b'] = False
        games_df['away_b2b'] = False
        games_df['rest_advantage'] = 0

        # Ensure commence_time is in ISO format
        if 'commence_time' in games_df.columns:
            games_df['commence_time'] = pd.to_datetime(games_df['commence_time']).dt.strftime('%Y-%m-%dT%H:%M:%S%z')

        logger.info(f"Found {len(games_df)} games for today")
        return games_df[['game_id', 'home_team', 'away_team', 'commence_time',
                        'pred_diff', 'home_b2b', 'away_b2b', 'rest_advantage']]

    except Exception as e:
        logger.error(f"Error fetching today's games: {e}")
        return pd.DataFrame()


def fetch_current_odds(game_ids: List[str] = None) -> pd.DataFrame:
    """
    Fetch current odds from Odds API and reshape to wide format.

    Args:
        game_ids: Optional list of game IDs to filter

    Returns:
        DataFrame with columns: game_id, market_spread, home_odds, away_odds, bookmaker
    """
    odds_client = OddsAPIClient()

    try:
        odds_df = odds_client.get_current_odds(markets=['spreads'])

        if odds_df.empty:
            return pd.DataFrame()

        if game_ids:
            odds_df = odds_df[odds_df['game_id'].isin(game_ids)]

        # Reshape from long to wide format
        # The 'team' column contains 'home' or 'away', not team names
        home_odds = odds_df[odds_df['team'] == 'home'].copy()
        away_odds = odds_df[odds_df['team'] == 'away'].copy()

        # Rename columns for clarity
        home_odds = home_odds.rename(columns={'line': 'market_spread', 'odds': 'home_odds'})
        away_odds = away_odds.rename(columns={'odds': 'away_odds'})

        # Merge home and away odds
        merged = pd.merge(
            home_odds[['game_id', 'bookmaker', 'market_spread', 'home_odds']],
            away_odds[['game_id', 'bookmaker', 'away_odds']],
            on=['game_id', 'bookmaker'],
            how='inner'
        )

        # Get best odds per game (use first bookmaker for simplicity)
        result = merged.groupby('game_id').first().reset_index()

        return result

    except Exception as e:
        logger.error(f"Error fetching odds: {e}")
        return pd.DataFrame()


def analyze_line_movements(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze line movement patterns for each game.

    Args:
        games_df: Games DataFrame

    Returns:
        Enhanced DataFrame with movement analysis
    """
    line_manager = LineHistoryManager()

    # Add columns for movement data
    games_df['movement_pattern'] = None
    games_df['line_velocity'] = None
    games_df['has_reversal'] = False
    games_df['opening_line'] = None

    for idx, game in games_df.iterrows():
        game_id = game['game_id']

        try:
            # Get line history
            line_history = line_manager.get_line_history(
                game_id=game_id,
                bet_type='spread',
                side='home'
            )

            if not line_history.empty:
                # Analyze movement pattern
                movement = line_manager.analyze_movement_pattern(
                    game_id=game_id,
                    bet_type='spread',
                    bet_side='home'
                )

                if movement:
                    games_df.at[idx, 'movement_pattern'] = movement.get('pattern')
                    games_df.at[idx, 'line_velocity'] = movement.get('velocity')

                # Check for reversals
                reversals = line_manager.detect_line_reversals(
                    game_id=game_id,
                    bet_type='spread'
                )

                if reversals:
                    games_df.at[idx, 'has_reversal'] = True

            # Get opening line
            opening = line_manager.get_opening_line(
                game_id=game_id,
                bet_type='spread',
                bet_side='home'
            )

            if opening:
                games_df.at[idx, 'opening_line'] = opening.get('opening_line')

        except Exception as e:
            logger.debug(f"Could not analyze line movement for {game_id}: {e}")

    return games_df


def evaluate_games(
    games_df: pd.DataFrame,
    strategy: EdgeStrategy
) -> List[BetSignal]:
    """
    Evaluate all games and generate bet signals.

    Args:
        games_df: Games DataFrame with predictions and odds
        strategy: EdgeStrategy instance

    Returns:
        List of BetSignal objects
    """
    signals = []

    for _, game in games_df.iterrows():
        signal = strategy.evaluate_game(
            game_id=game['game_id'],
            home_team=game['home_team'],
            away_team=game['away_team'],
            pred_diff=game['pred_diff'],
            market_spread=game['market_spread'],
            home_b2b=game.get('home_b2b', False),
            away_b2b=game.get('away_b2b', False),
            rest_advantage=game.get('rest_advantage', 0),
        )

        signals.append(signal)

    return signals


def log_bet_recommendation(
    signal: BetSignal,
    game_data: pd.Series,
    paper_mode: bool = True,
    dry_run: bool = False
) -> Dict:
    """
    Log bet recommendation to database.

    Args:
        signal: BetSignal object
        game_data: Game row from DataFrame
        paper_mode: If True, mark as paper trade
        dry_run: If True, don't actually log

    Returns:
        Logged bet record
    """
    # Get odds for the bet side
    if signal.bet_side == "HOME":
        odds = game_data.get('home_odds', -110)
        line = signal.market_spread
        team = signal.home_team
    else:
        odds = game_data.get('away_odds', -110)
        line = -signal.market_spread
        team = signal.away_team

    if dry_run:
        logger.info(f"[DRY RUN] Would log bet: {signal.bet_side} {team}")
        return {}

    # Log to database
    bet_record = log_bet(
        game_id=signal.game_id,
        home_team=signal.home_team,
        away_team=signal.away_team,
        commence_time=game_data['commence_time'],
        bet_type='spread',
        bet_side=signal.bet_side.lower(),
        odds=odds,
        line=line,
        model_prob=0.50 + (signal.model_edge * 0.01),
        market_prob=0.50,
        edge=signal.model_edge,
        kelly=signal.model_edge * 0.01,
        bookmaker='PAPER_TRADE' if paper_mode else 'draftkings',
    )

    return bet_record


def print_recommendations(signals: List[BetSignal], games_df: pd.DataFrame):
    """
    Print formatted bet recommendations.

    Args:
        signals: List of BetSignal objects
        games_df: Games DataFrame
    """
    actionable = [s for s in signals if s.is_actionable]

    if not actionable:
        logger.info("\n📊 No actionable bets found today")
        logger.info("   All games filtered out by strategy criteria\n")
        return

    logger.info("\n" + "=" * 80)
    logger.info("🎯 BET RECOMMENDATIONS")
    logger.info("=" * 80)

    for i, signal in enumerate(actionable, 1):
        game = games_df[games_df['game_id'] == signal.game_id].iloc[0]

        team = signal.home_team if signal.bet_side == "HOME" else signal.away_team
        opponent = signal.away_team if signal.bet_side == "HOME" else signal.home_team

        logger.info(f"\n#{i}. {signal.bet_side} {team} vs {opponent}")
        logger.info(f"   Game: {signal.away_team} @ {signal.home_team}")
        logger.info(f"   Commence: {game['commence_time']}")
        logger.info(f"   Model Edge: {signal.model_edge:+.2f} pts")
        logger.info(f"   Market Spread: {signal.market_spread:+.1f}")
        logger.info(f"   Confidence: {signal.confidence}")
        logger.info(f"   Filters Passed: {', '.join(signal.filters_passed)}")

        # Show movement data if available
        if pd.notna(game.get('movement_pattern')):
            logger.info(f"   Movement Pattern: {game['movement_pattern']}")
        if pd.notna(game.get('line_velocity')):
            logger.info(f"   Line Velocity: {game['line_velocity']:.2f} pts/hr")
        if game.get('has_reversal'):
            logger.info(f"   ⚠️  Line Reversal Detected")

    logger.info("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Daily betting pipeline with integrated signals')
    parser.add_argument('--live', action='store_true', help='Live mode (real bets, not paper)')
    parser.add_argument('--dry-run', action='store_true', help='Show recommendations without logging')
    parser.add_argument('--strategy', type=str, default='clv_filtered',
                       choices=['baseline', 'clv_filtered', 'optimal_timing', 'team_filtered'],
                       help='Strategy to use (default: clv_filtered)')
    args = parser.parse_args()

    # Set paper trading mode
    global PAPER_TRADING
    PAPER_TRADING = not args.live

    # Print header
    logger.info("=" * 80)
    logger.info("🏀 DAILY BETTING PIPELINE - Integrated System")
    logger.info("=" * 80)
    logger.info(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🎮 Mode: {'🔴 LIVE' if args.live else '📝 PAPER TRADING'}")
    logger.info(f"🎯 Strategy: {args.strategy}")
    if args.dry_run:
        logger.info("⚠️  DRY RUN MODE - No bets will be logged")
    logger.info("=" * 80)

    # Step 1: Get today's games
    logger.info("\n1️⃣  Getting today's games...")
    games_df = get_todays_games()
    logger.info(f"   ✓ Found {len(games_df)} games")

    if games_df.empty:
        logger.info("\n⚠️  No games today - exiting")
        return 0

    # Step 2: Fetch current odds
    logger.info("\n2️⃣  Fetching current market odds...")
    current_odds = fetch_current_odds(games_df['game_id'].tolist())

    if current_odds.empty:
        logger.error("   ✗ Could not fetch odds - exiting")
        return 1

    # Merge odds with games
    games_df = games_df.merge(
        current_odds[['game_id', 'market_spread', 'home_odds', 'away_odds', 'bookmaker']],
        on='game_id',
        how='left'
    )
    logger.info(f"   ✓ Fetched odds for {len(current_odds)} games")

    # Step 3: Calculate model edge
    logger.info("\n3️⃣  Calculating model edge...")
    games_df['model_edge'] = games_df['pred_diff'] + games_df['market_spread']
    logger.info(f"   ✓ Calculated edge for all games")
    logger.info(f"   📊 Average model edge: {games_df['model_edge'].mean():+.2f} pts")

    # Step 4: Analyze line movements
    logger.info("\n4️⃣  Analyzing line movements...")
    games_df = analyze_line_movements(games_df)
    patterns_found = games_df['movement_pattern'].notna().sum()
    reversals_found = games_df['has_reversal'].sum()
    logger.info(f"   ✓ Analyzed line history")
    logger.info(f"   📈 Movement patterns: {patterns_found}")
    logger.info(f"   🔄 Reversals detected: {reversals_found}")

    # Step 5: Configure strategy
    logger.info("\n5️⃣  Configuring betting strategy...")
    if args.strategy == 'clv_filtered':
        strategy = EdgeStrategy.clv_filtered_strategy()
    elif args.strategy == 'optimal_timing':
        strategy = EdgeStrategy.optimal_timing_strategy()
    elif args.strategy == 'team_filtered':
        strategy = EdgeStrategy.team_filtered_strategy()
    else:
        strategy = EdgeStrategy.primary_strategy()

    logger.info(f"   ✓ Using {args.strategy} strategy")
    logger.info(f"   ⚙️  Edge threshold: {strategy.edge_threshold}")
    logger.info(f"   ⚙️  CLV filter: {'Enabled' if strategy.clv_filter_enabled else 'Disabled'}")
    logger.info(f"   ⚙️  Timing filter: {'Enabled' if strategy.optimal_timing_filter else 'Disabled'}")

    # Step 6: Evaluate games
    logger.info("\n6️⃣  Evaluating games...")
    signals = evaluate_games(games_df, strategy)
    actionable = [s for s in signals if s.is_actionable]
    logger.info(f"   ✓ Evaluated {len(signals)} games")
    logger.info(f"   🎯 Actionable bets: {len(actionable)}")

    # Step 7: Print recommendations
    print_recommendations(signals, games_df)

    # Step 8: Log bets
    if actionable and not args.dry_run:
        logger.info("\n7️⃣  Logging bets to database...")

        for signal in actionable:
            game = games_df[games_df['game_id'] == signal.game_id].iloc[0]
            log_bet_recommendation(signal, game, paper_mode=PAPER_TRADING, dry_run=args.dry_run)

        logger.info(f"   ✓ Logged {len(actionable)} bets")
        logger.info(f"   💾 Mode: {'Paper trade' if PAPER_TRADING else 'Live bet'}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("✅ Pipeline complete!")
    logger.info("=" * 80)

    if actionable:
        logger.info(f"\n📝 Next steps:")
        logger.info(f"   1. Review recommendations above")
        logger.info(f"   2. {'Place paper trades' if PAPER_TRADING else 'Place live bets'}")
        logger.info(f"   3. Monitor with: python scripts/generate_clv_report.py")
    else:
        logger.info(f"\n💡 No bets recommended today - all games filtered")

    return 0


if __name__ == "__main__":
    sys.exit(main())
