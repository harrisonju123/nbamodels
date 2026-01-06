"""
Multi-Strategy Framework Example

Demonstrates how to use the strategy orchestrator with multiple betting strategies.
"""

import pandas as pd
from src.betting.orchestrator import StrategyOrchestrator, OrchestratorConfig
from src.betting.strategies import (
    TotalsStrategy,
    LiveBettingStrategy,
    ArbitrageStrategy,
)

def main():
    """Run multi-strategy framework example."""

    # Configure orchestrator
    config = OrchestratorConfig(
        bankroll=1000.0,
        kelly_fraction=0.2,
        max_daily_exposure=0.15,  # Max 15% of bankroll per day
        max_pending_exposure=0.25,  # Max 25% in unsettled bets
        strategy_allocation={
            'totals': 0.40,      # 40% allocation to totals
            'live': 0.30,        # 30% to live betting
            'arbitrage': 0.30,   # 30% to arbitrage
        },
        max_bets_per_strategy={
            'totals': 5,
            'live': 3,
            'arbitrage': 5,
        }
    )

    # Initialize strategies
    strategies = [
        TotalsStrategy(
            model_path="models/totals_model.pkl",
            min_edge=0.05,
            require_no_b2b=True,
        ),
        LiveBettingStrategy(
            min_edge=0.05,
            min_confidence=0.5,
            max_quarter=4,
        ),
        ArbitrageStrategy(
            min_arb_profit=0.01,  # 1% minimum
            bookmakers=["draftkings", "fanduel", "betmgm"],
        ),
    ]

    # Create orchestrator
    orchestrator = StrategyOrchestrator(strategies, config)

    # Load data (example - you'd load real data)
    games_df = pd.DataFrame({
        'game_id': ['game1', 'game2'],
        'home_team': ['LAL', 'GSW'],
        'away_team': ['BOS', 'PHX'],
        'home_b2b': [False, False],
        'away_b2b': [False, True],
    })

    features_df = pd.DataFrame({
        'game_id': ['game1', 'game2'],
        # ... model features here
    })

    odds_df = pd.DataFrame({
        'game_id': ['game1', 'game2'],
        'market': ['total', 'total'],
        'total_line': [220.5, 225.0],
        'over_odds': [-110, -115],
        'under_odds': [-110, -105],
        'bookmaker': ['draftkings', 'fanduel'],
    })

    # Run all strategies
    print("=== Running All Strategies ===\n")
    signals = orchestrator.run_all_strategies(
        games_df=games_df,
        features_df=features_df,
        odds_df=odds_df,
    )

    print(f"\nFound {len(signals)} total signals")

    # Size and filter bets
    print("\n=== Sizing Bets ===\n")
    recommendations = orchestrator.size_and_filter_bets(signals)

    print(f"Final recommendations: {len(recommendations)} bets")

    # Display recommendations
    for i, rec in enumerate(recommendations, 1):
        signal = rec['signal']
        print(f"\nBet #{i}:")
        print(f"  Strategy: {signal.strategy_type.value}")
        print(f"  Game: {signal.away_team} @ {signal.home_team}")
        print(f"  Bet: {signal.bet_type} {signal.bet_side}")
        print(f"  Edge: {signal.edge:.1%}")
        print(f"  Confidence: {signal.confidence}")
        print(f"  Bet Size: ${rec['bet_size']:.2f}")
        print(f"  Kelly Fraction: {rec['kelly_fraction']:.2%}")
        if rec['limiting_factors']:
            print(f"  Limiting Factors: {', '.join(rec['limiting_factors'])}")

    # Show daily stats
    print("\n=== Daily Stats ===\n")
    stats = orchestrator.get_daily_stats()
    print(f"Date: {stats['date']}")
    print(f"Total Bets: {stats['total_bets']}")
    print(f"Total Exposure: ${stats['total_exposure']:.2f}")

    print("\nBy Strategy:")
    for strategy_name, strategy_stats in stats['by_strategy'].items():
        print(f"  {strategy_name}:")
        print(f"    Bets: {strategy_stats['bets']}")
        print(f"    Exposure: ${strategy_stats['exposure']:.2f}")
        print(f"    Utilization: {strategy_stats['utilization']:.1%}")


if __name__ == "__main__":
    main()
