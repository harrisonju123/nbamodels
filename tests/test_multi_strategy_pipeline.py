#!/usr/bin/env python3
"""
Test Multi-Strategy Pipeline with Mock Data

Verifies the pipeline works end-to-end without requiring API access.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

# Import pipeline components
from src.betting.orchestrator import StrategyOrchestrator, OrchestratorConfig
from src.betting.strategies import (
    StrategyType,
    TotalsStrategy,
    ArbitrageStrategy,
)
from src.config.strategy_config import load_config, get_enabled_strategies

# Import mock data generators
from tests.test_mock_data import (
    generate_mock_games,
    generate_mock_features,
    generate_mock_totals_odds,
    generate_mock_spread_odds,
)


def test_pipeline_with_mock_data():
    """Test the multi-strategy pipeline with mock data."""

    logger.info("=" * 80)
    logger.info("🧪 TESTING MULTI-STRATEGY PIPELINE WITH MOCK DATA")
    logger.info("=" * 80)

    # Load config
    logger.info("\n1️⃣  Loading configuration...")
    config = load_config()
    enabled = get_enabled_strategies(config)
    logger.info(f"   ✓ Config loaded, enabled strategies: {', '.join(enabled)}")

    # Generate mock data
    logger.info("\n2️⃣  Generating mock data...")
    games_df = generate_mock_games(n_games=10)
    features_df = generate_mock_features(games_df)

    # Generate odds for both totals and spreads
    totals_odds = generate_mock_totals_odds(games_df)
    spread_odds = generate_mock_spread_odds(games_df)

    # Combine odds
    all_odds = pd.concat([totals_odds, spread_odds], ignore_index=True)

    logger.info(f"   ✓ Generated {len(games_df)} games")
    logger.info(f"   ✓ Generated {len(features_df)} feature sets")
    logger.info(f"   ✓ Generated {len(all_odds)} odds entries")

    # Create strategies
    logger.info("\n3️⃣  Initializing strategies...")
    strategies = []

    if 'totals' in enabled:
        try:
            strategies.append(TotalsStrategy(
                model_path="models/totals_model.pkl",
                min_edge=0.05,
            ))
            logger.info("   ✓ TotalsStrategy initialized")
        except Exception as e:
            logger.warning(f"   ⚠️  Could not load TotalsStrategy: {e}")

    if 'arbitrage' in enabled:
        strategies.append(ArbitrageStrategy(min_arb_profit=0.01))
        logger.info("   ✓ ArbitrageStrategy initialized")

    if not strategies:
        logger.error("   ✗ No strategies could be initialized")
        return False

    # Create orchestrator
    logger.info("\n4️⃣  Creating orchestrator...")
    orchestrator_config = OrchestratorConfig(
        bankroll=1000.0,
        kelly_fraction=config.kelly_fraction,
        max_daily_exposure=config.max_daily_exposure,
        strategy_allocation={
            StrategyType.TOTALS: config.allocation.get('totals', 0.30),
            StrategyType.ARBITRAGE: config.allocation.get('arbitrage', 0.25),
        },
        max_bets_per_strategy={
            StrategyType.TOTALS: config.daily_limits.get('totals', 5),
            StrategyType.ARBITRAGE: config.daily_limits.get('arbitrage', 10),
        },
        min_bet_size=config.min_bet_size,
    )

    orchestrator = StrategyOrchestrator(strategies, orchestrator_config)
    logger.info(f"   ✓ Orchestrator initialized with {len(strategies)} strategies")

    # Run strategies
    logger.info("\n5️⃣  Running strategies...")
    signals = orchestrator.run_all_strategies(
        games_df=games_df,
        features_df=features_df,
        odds_df=all_odds,
    )
    logger.info(f"   ✓ Generated {len(signals)} signals")

    # Size bets
    logger.info("\n6️⃣  Sizing bets...")
    recommendations = orchestrator.size_and_filter_bets(signals, bankroll=1000.0)
    logger.info(f"   ✓ {len(recommendations)} bets passed filters")

    # Get stats
    logger.info("\n7️⃣  Getting daily stats...")
    stats = orchestrator.get_daily_stats()
    logger.info(f"   ✓ Total bets: {stats['total_bets']}")
    logger.info(f"   ✓ Total exposure: ${stats['total_exposure']:.2f}")

    # Print recommendations
    if recommendations:
        logger.info("\n" + "=" * 80)
        logger.info("📊 RECOMMENDATIONS")
        logger.info("=" * 80)

        for i, rec in enumerate(recommendations, 1):
            signal = rec['signal']
            logger.info(f"\n#{i}. {signal.strategy_type.value.upper()}")
            logger.info(f"   Game: {signal.away_team} @ {signal.home_team}")
            logger.info(f"   Type: {signal.bet_type} {signal.bet_side}")
            logger.info(f"   Edge: {signal.edge:.2%}")
            logger.info(f"   Bet size: ${rec['bet_size']:.2f}")
            logger.info(f"   Kelly: {rec['kelly_fraction']:.1%}")
    else:
        logger.info("\n💡 No bets recommended (all filtered out)")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("✅ PIPELINE TEST COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\n📊 Summary:")
    logger.info(f"   Strategies tested: {len(strategies)}")
    logger.info(f"   Signals generated: {len(signals)}")
    logger.info(f"   Bets recommended: {len(recommendations)}")
    logger.info(f"   Total exposure: ${stats['total_exposure']:.2f}")

    return True


if __name__ == "__main__":
    try:
        success = test_pipeline_with_mock_data()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
