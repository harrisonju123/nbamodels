#!/usr/bin/env python3
"""
Daily Multi-Strategy Betting Pipeline

Unified pipeline using StrategyOrchestrator to coordinate:
- Spread betting (existing EdgeStrategy)
- Totals betting (TotalsStrategy)
- Live betting (LiveBettingStrategy)
- Arbitrage (ArbitrageStrategy)
- Player props (PlayerPropsStrategy)

Usage:
    # Paper trading mode (default)
    python scripts/daily_multi_strategy_pipeline.py

    # Live mode (real bets)
    python scripts/daily_multi_strategy_pipeline.py --live

    # Dry run (show recommendations without logging)
    python scripts/daily_multi_strategy_pipeline.py --dry-run

    # Enable specific strategies
    python scripts/daily_multi_strategy_pipeline.py --strategies totals,live,arbitrage
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import argparse
from datetime import datetime
from typing import List, Dict
import pandas as pd
from loguru import logger

# Data sources
from src.data.odds_api import OddsAPIClient
from src.data.nba_stats import NBAStatsClient
from src.features import GameFeatureBuilder

# Multi-strategy framework
from src.betting.orchestrator import StrategyOrchestrator, OrchestratorConfig
from src.betting.strategies import (
    StrategyType,
    TotalsStrategy,
    LiveBettingStrategy,
    ArbitrageStrategy,
    PlayerPropsStrategy,
)
from src.betting.strategies.b2b_rest_strategy import B2BRestStrategy

# Legacy spread strategy (wraps existing EdgeStrategy)
from src.betting.edge_strategy import EdgeStrategy

# Lineup and injury data for player props safety
from src.data.lineup_scrapers import ESPNLineupClient
from src.data.espn_injuries import ESPNClient

# Risk management
from src.risk import RiskConfig, ExposureManager
from src.betting.kelly import KellyBetSizer
from src.bankroll.bankroll_manager import BankrollManager
from src.bet_tracker import log_bet

# Configuration
from src.config.strategy_config import load_config, get_enabled_strategies


def get_todays_games_and_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get today's games with features for all strategies.

    Returns:
        (games_df, features_df) tuple
    """
    logger.info("Fetching today's NBA games and building features...")

    try:
        # Initialize clients
        stats_client = NBAStatsClient()
        odds_client = OddsAPIClient()
        feature_builder = GameFeatureBuilder()

        # Get current odds to identify games
        odds_df = odds_client.get_current_odds()

        if odds_df.empty:
            logger.warning("No games with odds available")
            return pd.DataFrame(), pd.DataFrame()

        # Extract unique games
        games = odds_df[['game_id', 'home_team', 'away_team', 'commence_time']].drop_duplicates()

        # Build features (reuse logic from daily_betting_pipeline.py)
        from datetime import timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)

        recent_games = stats_client.get_games(
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )

        if recent_games.empty:
            logger.warning("No recent games found for feature building")
            return games, pd.DataFrame()

        # Build team features
        team_features = feature_builder.team_builder.build_all_features(recent_games)
        latest_features = (
            team_features
            .sort_values("date")
            .groupby("team")
            .last()
            .reset_index()
        )

        # Build game features
        features_records = []
        for _, game in games.iterrows():
            home_team = game['home_team']
            away_team = game['away_team']

            home_stats = latest_features[latest_features['team'] == home_team]
            away_stats = latest_features[latest_features['team'] == away_team]

            if home_stats.empty or away_stats.empty:
                logger.warning(f"Missing stats for {away_team} @ {home_team}")
                continue

            # Build feature dict
            record = {
                'game_id': game['game_id'],
                'home_team': home_team,
                'away_team': away_team,
            }

            # Add home features
            for col in home_stats.columns:
                if col not in ['team', 'game_id', 'date', 'season', 'opponent']:
                    record[f'home_{col}'] = home_stats[col].values[0]

            # Add away features
            for col in away_stats.columns:
                if col not in ['team', 'game_id', 'date', 'season', 'opponent']:
                    record[f'away_{col}'] = away_stats[col].values[0]

            features_records.append(record)

        features_df = pd.DataFrame(features_records)

        # Add schedule info
        games['home_b2b'] = False  # TODO: Implement B2B detection
        games['away_b2b'] = False

        logger.success(f"Built features for {len(features_df)} games")
        return games, features_df

    except Exception as e:
        logger.error(f"Error fetching games and features: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return pd.DataFrame(), pd.DataFrame()


def fetch_all_odds() -> Dict[str, pd.DataFrame]:
    """
    Fetch odds for all markets needed by strategies.

    Returns:
        Dict with keys: 'spreads', 'totals', 'h2h' (moneylines), 'player_props'
    """
    logger.info("Fetching odds from all markets...")
    odds_client = OddsAPIClient()

    odds_data = {}

    try:
        # Spreads (for spread strategy and arbitrage)
        odds_data['spreads'] = odds_client.get_current_odds(markets=['spreads'])

        # Totals (for totals strategy)
        odds_data['totals'] = odds_client.get_current_odds(markets=['totals'])

        # Moneylines (for live betting)
        odds_data['h2h'] = odds_client.get_current_odds(markets=['h2h'])

        # Player props (for player props strategy)
        # Get all games
        games = odds_data['spreads']['game_id'].unique() if not odds_data['spreads'].empty else []

        all_props = []
        for game_id in games:
            try:
                props = odds_client.get_player_props(
                    event_id=game_id,
                    markets=['player_points', 'player_rebounds', 'player_assists', 'player_threes']
                )
                all_props.append(props)
            except Exception as e:
                logger.debug(f"Could not fetch props for {game_id}: {e}")

        if all_props:
            odds_data['player_props'] = pd.concat(all_props, ignore_index=True)
        else:
            odds_data['player_props'] = pd.DataFrame()

        logger.success(f"Fetched odds: {len(odds_data['spreads'])} spreads, {len(odds_data['totals'])} totals, "
                      f"{len(odds_data.get('player_props', []))} props")

        return odds_data

    except Exception as e:
        logger.error(f"Error fetching odds: {e}")
        return {k: pd.DataFrame() for k in ['spreads', 'totals', 'h2h', 'player_props']}


def create_orchestrator(config: 'MultiStrategyConfig', bankroll: float) -> StrategyOrchestrator:
    """
    Create StrategyOrchestrator with enabled strategies from config.

    Args:
        config: MultiStrategyConfig object
        bankroll: Current bankroll

    Returns:
        Configured StrategyOrchestrator
    """
    strategies = []
    enabled = get_enabled_strategies(config)

    # Totals Strategy
    if 'totals' in enabled:
        totals_config = config.strategies['totals']
        strategies.append(
            TotalsStrategy(
                model_path=totals_config.get('model_path', "models/totals_model.pkl"),
                min_edge=totals_config.get('min_edge', 0.05),
                require_no_b2b=totals_config.get('require_no_b2b', False),
                min_pace=totals_config.get('min_pace'),
                max_pace=totals_config.get('max_pace'),
            )
        )
        logger.info("✓ Enabled TotalsStrategy")

    # Live Betting Strategy
    if 'live' in enabled:
        live_config = config.strategies['live']
        strategies.append(
            LiveBettingStrategy(
                min_edge=live_config.get('min_edge', 0.05),
                min_confidence=live_config.get('min_confidence', 50.0),
                max_quarter=live_config.get('max_quarter', 4),
            )
        )
        logger.info("✓ Enabled LiveBettingStrategy")

    # Arbitrage Strategy
    if 'arbitrage' in enabled:
        arb_config = config.strategies['arbitrage']
        strategies.append(
            ArbitrageStrategy(
                min_arb_profit=arb_config.get('min_arb_profit', 0.01),
            )
        )
        logger.info("✓ Enabled ArbitrageStrategy")

    # Player Props Strategy
    if 'props' in enabled:
        props_config = config.strategies['props']

        # Initialize lineup and injury clients for safety filtering
        lineup_client = ESPNLineupClient()
        injury_client = ESPNClient()

        # Fetch latest lineups and injuries
        logger.info("  Fetching lineups and injury reports for player props safety...")
        try:
            # This will cache lineup data in the database
            lineup_client.collect_today_lineups()
            logger.info("  ✓ Lineups fetched")
        except Exception as e:
            logger.warning(f"  Could not fetch lineups: {e}")

        try:
            # This will fetch current injury report
            injuries = injury_client.get_injuries()
            logger.info(f"  ✓ Injury report fetched ({len(injuries)} players)")
        except Exception as e:
            logger.warning(f"  Could not fetch injuries: {e}")

        strategies.append(
            PlayerPropsStrategy(
                models_dir=props_config.get('models_dir', "models/player_props"),
                min_edge=props_config.get('min_edge', 0.05),
                lineup_client=lineup_client,
                injury_client=injury_client,
                require_starter=props_config.get('require_starter', True),
                skip_questionable=props_config.get('skip_questionable', True),
            )
        )
        logger.info("✓ Enabled PlayerPropsStrategy with lineup/injury filtering")

    # B2B Rest Advantage Strategy
    if 'b2b_rest' in enabled:
        b2b_config = config.strategies['b2b_rest']
        strategies.append(
            B2BRestStrategy(
                min_rest_advantage=b2b_config.get('min_rest_advantage', 2),
                min_edge=b2b_config.get('min_edge', 0.03),
            )
        )
        logger.info("✓ Enabled B2BRestStrategy")

    if not strategies:
        logger.warning("No strategies enabled!")
        return None

    # Configure orchestrator from config
    orchestrator_config = OrchestratorConfig(
        bankroll=bankroll,
        kelly_fraction=config.kelly_fraction,
        max_daily_exposure=config.max_daily_exposure,
        strategy_allocation={
            StrategyType.TOTALS: config.allocation.get('totals', 0.30),
            StrategyType.LIVE: config.allocation.get('live', 0.20),
            StrategyType.ARBITRAGE: config.allocation.get('arbitrage', 0.25),
            StrategyType.PLAYER_PROPS: config.allocation.get('props', 0.15),
            StrategyType.B2B_REST: config.allocation.get('b2b_rest', 0.10),
            StrategyType.SPREAD: config.allocation.get('spread', 0.10),
        },
        max_bets_per_strategy={
            StrategyType.TOTALS: config.daily_limits.get('totals', 5),
            StrategyType.LIVE: config.daily_limits.get('live', 3),
            StrategyType.ARBITRAGE: config.daily_limits.get('arbitrage', 10),
            StrategyType.PLAYER_PROPS: config.daily_limits.get('props', 8),
            StrategyType.B2B_REST: config.daily_limits.get('b2b_rest', 5),
            StrategyType.SPREAD: config.daily_limits.get('spread', 3),
        },
        min_bet_size=config.min_bet_size,
    )

    orchestrator = StrategyOrchestrator(strategies, orchestrator_config)
    logger.success(f"Orchestrator initialized with {len(strategies)} strategies")

    return orchestrator


def log_recommendations(
    recommendations: List[Dict],
    paper_mode: bool = True,
    dry_run: bool = False
):
    """
    Log bet recommendations to database.

    Args:
        recommendations: List of bet recommendations from orchestrator
        paper_mode: Whether to mark as paper trades
        dry_run: If True, don't actually log
    """
    if not recommendations:
        logger.info("No bets to log")
        return

    logger.info(f"Logging {len(recommendations)} bets...")

    for rec in recommendations:
        signal = rec['signal']
        bet_size = rec['bet_size']

        # Determine bet type and side
        bet_type = signal.bet_type
        bet_side = signal.bet_side.lower()

        # Get odds and line
        odds = signal.odds if signal.odds else -110
        line = signal.line

        if dry_run:
            logger.info(f"[DRY RUN] {signal.strategy_type.value}: {bet_type} {bet_side} "
                       f"${bet_size:.2f} @ {odds:+.0f} (edge: {signal.edge:.2%})")
            continue

        # Log to database
        try:
            bet_record = log_bet(
                game_id=signal.game_id,
                home_team=signal.home_team,
                away_team=signal.away_team,
                commence_time=signal.commence_time if hasattr(signal, 'commence_time') else datetime.now().isoformat(),
                bet_type=bet_type,
                bet_side=bet_side,
                odds=odds,
                line=line,
                model_prob=signal.model_prob,
                market_prob=signal.market_prob,
                edge=signal.edge,
                kelly=signal.edge,
                bookmaker=f"PAPER/{signal.bookmaker}" if paper_mode else signal.bookmaker,
                bet_amount=bet_size,
                strategy_type=signal.strategy_type.value,
                player_id=signal.player_id,
                player_name=signal.player_name,
                prop_type=signal.prop_type,
            )

            logger.info(f"✓ Logged {signal.strategy_type.value}: {bet_type} {bet_side} "
                       f"${bet_size:.2f} @ {odds:+.0f}")

        except Exception as e:
            logger.error(f"Error logging bet: {e}")


def print_summary(recommendations: List[Dict], stats: Dict):
    """
    Print summary of recommendations.

    Args:
        recommendations: List of bet recommendations
        stats: Daily stats from orchestrator
    """
    logger.info("\n" + "=" * 80)
    logger.info("📊 MULTI-STRATEGY BETTING SUMMARY")
    logger.info("=" * 80)

    if not recommendations:
        logger.info("\n💡 No bets recommended today - all games filtered")
        return

    # Group by strategy
    by_strategy = {}
    for rec in recommendations:
        strategy = rec['signal'].strategy_type.value
        if strategy not in by_strategy:
            by_strategy[strategy] = []
        by_strategy[strategy].append(rec)

    # Print by strategy
    total_exposure = 0
    for strategy, recs in sorted(by_strategy.items()):
        strategy_exposure = sum(r['bet_size'] for r in recs)
        total_exposure += strategy_exposure

        logger.info(f"\n{strategy.upper()}:")
        logger.info(f"  Bets: {len(recs)}")
        logger.info(f"  Exposure: ${strategy_exposure:.2f}")
        logger.info(f"  Avg edge: {sum(r['signal'].edge for r in recs) / len(recs):.2%}")

    # Overall stats
    logger.info(f"\nOVERALL:")
    logger.info(f"  Total bets: {stats['total_bets']}")
    logger.info(f"  Total exposure: ${stats['total_exposure']:.2f}")
    logger.info(f"  Remaining daily limit: ${stats.get('remaining_daily_exposure', 0):.2f}")

    logger.info("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Multi-strategy daily betting pipeline')
    parser.add_argument('--live', action='store_true', help='Live mode (real bets)')
    parser.add_argument('--dry-run', action='store_true', help='Show recommendations without logging')
    parser.add_argument('--config', type=str, help='Path to config file (optional)')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    enabled_strategies = get_enabled_strategies(config)

    # Override paper_trading if --live flag is set
    if args.live:
        config.paper_trading = False

    # Override dry_run if flag is set
    if args.dry_run:
        config.dev['dry_run'] = True

    # Print header
    logger.info("=" * 80)
    logger.info("🏀 MULTI-STRATEGY BETTING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🎮 Mode: {'🔴 LIVE' if not config.paper_trading else '📝 PAPER TRADING'}")
    logger.info(f"🎯 Strategies: {', '.join(enabled_strategies)}")
    if config.dev['dry_run']:
        logger.info("⚠️  DRY RUN MODE")
    logger.info("=" * 80)

    # Get bankroll
    bankroll_mgr = BankrollManager()
    current_bankroll = bankroll_mgr.get_current_bankroll()

    if current_bankroll == 0:
        logger.info("⚠️  Bankroll not initialized - starting with $1000")
        bankroll_mgr.initialize_bankroll(1000.0)
        current_bankroll = 1000.0

    logger.info(f"💰 Current bankroll: ${current_bankroll:,.2f}\n")

    # Step 1: Get games and features
    logger.info("1️⃣  Fetching games and building features...")
    games_df, features_df = get_todays_games_and_features()

    if games_df.empty:
        logger.info("⚠️  No games today - exiting")
        return 0

    logger.info(f"   ✓ Found {len(games_df)} games\n")

    # Step 2: Fetch odds
    logger.info("2️⃣  Fetching market odds...")
    odds_data = fetch_all_odds()
    logger.info("")

    # Step 3: Create orchestrator
    logger.info("3️⃣  Configuring strategies...")
    orchestrator = create_orchestrator(config, current_bankroll)

    if not orchestrator:
        logger.error("Failed to create orchestrator - exiting")
        return 1

    logger.info("")

    # Step 4: Run all strategies
    logger.info("4️⃣  Evaluating games across all strategies...")

    # Combine odds into single DataFrame (orchestrator expects this format)
    all_odds = pd.concat([
        odds_data.get('spreads', pd.DataFrame()),
        odds_data.get('totals', pd.DataFrame()),
        odds_data.get('player_props', pd.DataFrame()),
    ], ignore_index=True)

    signals = orchestrator.run_all_strategies(
        games_df=games_df,
        features_df=features_df,
        odds_df=all_odds,
    )

    logger.info(f"   ✓ Found {len(signals)} signals\n")

    # Step 5: Size and filter bets
    logger.info("5️⃣  Sizing bets with risk management...")
    recommendations = orchestrator.size_and_filter_bets(signals, bankroll=current_bankroll)
    logger.info(f"   ✓ {len(recommendations)} bets passed filters\n")

    # Step 6: Get daily stats
    stats = orchestrator.get_daily_stats()

    # Step 7: Print summary
    print_summary(recommendations, stats)

    # Step 8: Log bets
    if recommendations and not config.dev['dry_run']:
        logger.info("\n6️⃣  Logging bets to database...")
        log_recommendations(recommendations, paper_mode=config.paper_trading, dry_run=config.dev['dry_run'])
        logger.info("")

    # Done
    logger.info("=" * 80)
    logger.info("✅ Pipeline complete!")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
