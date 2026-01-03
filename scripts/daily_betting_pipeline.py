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
    Get today's games with features ready for prediction.

    TODO: Replace this with your actual feature pipeline.

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
    # PLACEHOLDER IMPLEMENTATION
    # Replace with your actual data pipeline
    logger.warning("Using placeholder get_todays_games() - replace with your pipeline!")

    # Example structure - replace with real data
    games = pd.DataFrame({
        'game_id': ['game_001', 'game_002'],
        'home_team': ['LAL', 'GSW'],
        'away_team': ['BOS', 'DEN'],
        'commence_time': [
            datetime.now(timezone.utc).isoformat(),
            datetime.now(timezone.utc).isoformat()
        ],
        'pred_diff': [5.5, -3.2],  # Model predictions
        'home_b2b': [False, True],
        'away_b2b': [False, False],
        'rest_advantage': [1, -1],
    })

    return games


def fetch_current_odds(game_ids: List[str] = None) -> pd.DataFrame:
    """
    Fetch current odds from Odds API.

    Args:
        game_ids: Optional list of game IDs to filter

    Returns:
        DataFrame with current odds
    """
    odds_client = OddsAPIClient()

    try:
        odds_df = odds_client.get_current_odds(markets=['spreads'])

        if game_ids:
            odds_df = odds_df[odds_df['game_id'].isin(game_ids)]

        return odds_df

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
