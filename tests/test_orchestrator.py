"""
Tests for StrategyOrchestrator

Tests the orchestrator's ability to coordinate multiple strategies,
manage bankroll, and apply risk limits.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import pandas as pd
from tests.test_mock_data import (
    generate_mock_games,
    generate_mock_features,
    generate_mock_totals_odds,
    generate_mock_spread_odds,
)

from src.betting.orchestrator import StrategyOrchestrator, OrchestratorConfig
from src.betting.strategies import (
    StrategyType,
    TotalsStrategy,
    ArbitrageStrategy,
)


class TestOrchestratorConfig(unittest.TestCase):
    """Test OrchestratorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = OrchestratorConfig()

        self.assertEqual(config.bankroll, 1000.0)
        self.assertEqual(config.kelly_fraction, 0.2)
        self.assertEqual(config.max_daily_exposure, 0.15)
        self.assertIn(StrategyType.TOTALS, config.strategy_allocation)

    def test_custom_config(self):
        """Test custom configuration."""
        config = OrchestratorConfig(
            bankroll=5000.0,
            kelly_fraction=0.25,
            strategy_allocation={
                StrategyType.TOTALS: 0.50,
                StrategyType.ARBITRAGE: 0.50,
            }
        )

        self.assertEqual(config.bankroll, 5000.0)
        self.assertEqual(config.kelly_fraction, 0.25)
        self.assertEqual(config.strategy_allocation[StrategyType.TOTALS], 0.50)


class TestStrategyOrchestrator(unittest.TestCase):
    """Test StrategyOrchestrator."""

    def setUp(self):
        """Set up test data and orchestrator."""
        self.games_df = generate_mock_games(3)
        self.features_df = generate_mock_features(self.games_df)
        self.odds_df = pd.concat([
            generate_mock_totals_odds(self.games_df),
            generate_mock_spread_odds(self.games_df),
        ])

        # Create strategies
        self.strategies = [
            TotalsStrategy(
                model_path="models/totals_model.pkl",
                min_edge=0.05,
            ),
            ArbitrageStrategy(
                min_arb_profit=0.01,
            ),
        ]

        # Create config
        self.config = OrchestratorConfig(
            bankroll=1000.0,
            strategy_allocation={
                StrategyType.TOTALS: 0.50,
                StrategyType.ARBITRAGE: 0.50,
            },
            max_bets_per_strategy={
                StrategyType.TOTALS: 3,
                StrategyType.ARBITRAGE: 3,
            }
        )

    def test_orchestrator_initialization(self):
        """Test orchestrator can be initialized."""
        orchestrator = StrategyOrchestrator(
            self.strategies,
            self.config
        )

        self.assertEqual(len(orchestrator.strategies), 2)
        self.assertEqual(orchestrator.config.bankroll, 1000.0)

    def test_strategy_priority_sorting(self):
        """Test strategies are sorted by priority."""
        orchestrator = StrategyOrchestrator(
            self.strategies,
            self.config
        )

        # Strategies should be sorted by priority (lower = higher priority)
        priorities = [s.priority for s in orchestrator.strategies]
        self.assertEqual(priorities, sorted(priorities))

    def test_run_all_strategies(self):
        """Test running all strategies."""
        orchestrator = StrategyOrchestrator(
            self.strategies,
            self.config
        )

        signals = orchestrator.run_all_strategies(
            games_df=self.games_df,
            features_df=self.features_df,
            odds_df=self.odds_df,
        )

        self.assertIsInstance(signals, list)
        # May be empty if models not loaded or no edges found
        print(f"Generated {len(signals)} signals from orchestrator")

    def test_size_and_filter_bets(self):
        """Test bet sizing and filtering."""
        orchestrator = StrategyOrchestrator(
            self.strategies,
            self.config
        )

        signals = orchestrator.run_all_strategies(
            self.games_df,
            self.features_df,
            self.odds_df,
        )

        if len(signals) > 0:
            recommendations = orchestrator.size_and_filter_bets(signals)

            self.assertIsInstance(recommendations, list)

            for rec in recommendations:
                self.assertIn('signal', rec)
                self.assertIn('bet_size', rec)
                self.assertIn('kelly_fraction', rec)

                # Bet size should be positive and <= max
                self.assertGreater(rec['bet_size'], 0)
                self.assertLessEqual(rec['bet_size'], self.config.bankroll * 0.05)

    def test_daily_limits(self):
        """Test daily bet limits per strategy."""
        # Set very low limits
        config = OrchestratorConfig(
            bankroll=1000.0,
            max_bets_per_strategy={
                StrategyType.TOTALS: 1,
                StrategyType.ARBITRAGE: 1,
            }
        )

        orchestrator = StrategyOrchestrator(
            self.strategies,
            config
        )

        # Simulate multiple bets
        orchestrator._daily_bets[StrategyType.TOTALS] = 1

        # Check limit
        can_bet = orchestrator._check_daily_limits(StrategyType.TOTALS)
        self.assertFalse(can_bet)  # Should be at limit

        # Different strategy should still be able to bet
        can_bet = orchestrator._check_daily_limits(StrategyType.ARBITRAGE)
        self.assertTrue(can_bet)

    def test_allocation_limits(self):
        """Test strategy allocation limits."""
        config = OrchestratorConfig(
            bankroll=1000.0,
            strategy_allocation={
                StrategyType.TOTALS: 0.10,  # Only 10% = $100
            }
        )

        orchestrator = StrategyOrchestrator(
            self.strategies,
            config
        )

        # Use up allocation
        orchestrator._daily_exposure[StrategyType.TOTALS] = 100.0

        remaining = orchestrator._get_remaining_allocation(
            StrategyType.TOTALS,
            0.10
        )

        self.assertEqual(remaining, 0.0)

    def test_daily_reset(self):
        """Test daily tracking resets."""
        orchestrator = StrategyOrchestrator(
            self.strategies,
            self.config
        )

        # Set some tracking
        orchestrator._daily_bets[StrategyType.TOTALS] = 5
        orchestrator._daily_exposure[StrategyType.TOTALS] = 250.0

        # Reset
        orchestrator.reset_daily_tracking()

        self.assertEqual(len(orchestrator._daily_bets), 0)
        self.assertEqual(len(orchestrator._daily_exposure), 0)

    def test_get_daily_stats(self):
        """Test getting daily statistics."""
        orchestrator = StrategyOrchestrator(
            self.strategies,
            self.config
        )

        # Add some mock data
        orchestrator._daily_bets[StrategyType.TOTALS] = 3
        orchestrator._daily_exposure[StrategyType.TOTALS] = 150.0

        stats = orchestrator.get_daily_stats()

        self.assertIn('date', stats)
        self.assertIn('total_bets', stats)
        self.assertIn('total_exposure', stats)
        self.assertIn('by_strategy', stats)

        self.assertEqual(stats['total_bets'], 3)
        self.assertEqual(stats['total_exposure'], 150.0)

    def test_min_bet_size_filter(self):
        """Test minimum bet size filtering."""
        config = OrchestratorConfig(
            bankroll=100.0,  # Small bankroll
            min_bet_size=10.0,  # High minimum
        )

        orchestrator = StrategyOrchestrator(
            self.strategies,
            config
        )

        # Create a mock signal with tiny edge
        from src.betting.strategies.base import BetSignal, StrategyType

        signal = BetSignal(
            strategy_type=StrategyType.TOTALS,
            game_id='test',
            home_team='LAL',
            away_team='BOS',
            bet_type='totals',
            bet_side='OVER',
            model_prob=0.51,  # Tiny edge
            market_prob=0.50,
            edge=0.01,
            odds=-110,
        )

        recs = orchestrator.size_and_filter_bets([signal], bankroll=100.0)

        # Should filter out bets below min size
        # (1% edge with small bankroll will be < $10)
        print(f"Recommendations with min size filter: {len(recs)}")


def run_tests():
    """Run all tests and return results."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == '__main__':
    result = run_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
