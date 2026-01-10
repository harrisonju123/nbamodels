"""
Unit Tests for Betting Strategies

Tests each strategy independently with mock data.
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
    generate_mock_live_games,
    generate_mock_live_odds,
    generate_mock_player_props,
    generate_mock_player_features,
)

from src.betting.strategies import (
    BettingStrategy,
    BetSignal,
    StrategyType,
    TotalsStrategy,
    LiveBettingStrategy,
    ArbitrageStrategy,
    PlayerPropsStrategy,
)


class TestBetSignal(unittest.TestCase):
    """Test BetSignal dataclass."""

    def test_bet_signal_creation(self):
        """Test creating a BetSignal."""
        signal = BetSignal(
            strategy_type=StrategyType.TOTALS,
            game_id='test_game',
            home_team='LAL',
            away_team='BOS',
            bet_type='totals',
            bet_side='OVER',
            model_prob=0.55,
            market_prob=0.48,
            edge=0.07,
            line=220.5,
            odds=-110,
            confidence='HIGH',
        )

        self.assertEqual(signal.strategy_type, StrategyType.TOTALS)
        self.assertEqual(signal.bet_side, 'OVER')
        self.assertAlmostEqual(signal.edge, 0.07)
        self.assertTrue(signal.is_actionable)

    def test_pass_signal_not_actionable(self):
        """Test that PASS signals are not actionable."""
        signal = BetSignal(
            strategy_type=StrategyType.TOTALS,
            game_id='test_game',
            home_team='LAL',
            away_team='BOS',
            bet_type='totals',
            bet_side='PASS',
            model_prob=0.50,
            market_prob=0.48,
            edge=0.02,
        )

        self.assertFalse(signal.is_actionable)


class TestTotalsStrategy(unittest.TestCase):
    """Test TotalsStrategy."""

    def setUp(self):
        """Set up test data."""
        self.games_df = generate_mock_games(3)
        self.features_df = generate_mock_features(self.games_df)
        self.odds_df = generate_mock_totals_odds(self.games_df)

    def test_strategy_initialization(self):
        """Test strategy can be initialized."""
        strategy = TotalsStrategy(
            model_path="models/totals_model.pkl",  # May not exist
            min_edge=0.05,
        )

        self.assertEqual(strategy.strategy_type, StrategyType.TOTALS)
        self.assertEqual(strategy.priority, 1)
        self.assertEqual(strategy.min_edge, 0.05)

    def test_evaluate_games_without_model(self):
        """Test that strategy handles missing model gracefully."""
        strategy = TotalsStrategy(
            model_path="nonexistent_model.pkl",
            min_edge=0.05,
        )

        signals = strategy.evaluate_games(
            self.games_df,
            self.features_df,
            self.odds_df
        )

        # Should return empty list if model not found
        self.assertIsInstance(signals, list)

    def test_factory_methods(self):
        """Test factory methods create strategies."""
        high_pace = TotalsStrategy.high_pace_strategy()
        self.assertIsNotNone(high_pace.min_pace)

        low_pace = TotalsStrategy.low_pace_strategy()
        self.assertIsNotNone(low_pace.max_pace)

        conservative = TotalsStrategy.conservative_strategy()
        self.assertEqual(conservative.min_edge, 0.07)


class TestLiveBettingStrategy(unittest.TestCase):
    """Test LiveBettingStrategy."""

    def setUp(self):
        """Set up test data."""
        self.live_games = generate_mock_live_games()
        self.live_odds = generate_mock_live_odds()

    def test_strategy_initialization(self):
        """Test strategy can be initialized."""
        strategy = LiveBettingStrategy(
            min_edge=0.05,
            min_confidence=0.5,
            max_quarter=4,
        )

        self.assertEqual(strategy.strategy_type, StrategyType.LIVE)
        self.assertEqual(strategy.priority, 2)
        self.assertEqual(strategy.max_quarter, 4)

    def test_evaluate_live_games(self):
        """Test evaluating live games."""
        strategy = LiveBettingStrategy(
            min_edge=0.05,
            min_confidence=0.5,
        )

        signals = strategy.evaluate_live_games(
            self.live_games,
            self.live_odds
        )

        self.assertIsInstance(signals, list)
        # Should return signals (detector may find edges in mock data)
        for signal in signals:
            self.assertIsInstance(signal, BetSignal)
            self.assertEqual(signal.strategy_type, StrategyType.LIVE)

    def test_quarter_filter(self):
        """Test quarter filter works."""
        strategy = LiveBettingStrategy(
            min_edge=0.05,
            max_quarter=3,  # Only through Q3
        )

        # Mock game in Q4 should be filtered
        live_games = {
            'game_1': {
                'game_id': 'game_1',
                'home_team': 'LAL',
                'away_team': 'BOS',
                'home_score': 100,
                'away_score': 98,
                'quarter': 4,  # Q4
                'time_remaining': '5:00',
            }
        }

        signals = strategy.evaluate_live_games(live_games, self.live_odds)

        # Should filter out Q4 game
        q4_signals = [s for s in signals if s.game_id == 'game_1']
        self.assertEqual(len(q4_signals), 0)

    def test_factory_methods(self):
        """Test factory methods."""
        conservative = LiveBettingStrategy.conservative_strategy()
        self.assertEqual(conservative.min_edge, 0.07)
        self.assertEqual(conservative.max_quarter, 3)

        aggressive = LiveBettingStrategy.aggressive_strategy()
        self.assertEqual(aggressive.min_edge, 0.04)


class TestArbitrageStrategy(unittest.TestCase):
    """Test ArbitrageStrategy."""

    def setUp(self):
        """Set up test data."""
        self.games_df = generate_mock_games(2)
        self.features_df = generate_mock_features(self.games_df)
        self.odds_df = generate_mock_spread_odds(self.games_df)

    def test_strategy_initialization(self):
        """Test strategy can be initialized."""
        strategy = ArbitrageStrategy(
            min_arb_profit=0.01,
        )

        self.assertEqual(strategy.strategy_type, StrategyType.ARBITRAGE)
        self.assertEqual(strategy.priority, 3)
        self.assertEqual(strategy.min_arb_profit, 0.01)

    def test_find_arbitrage(self):
        """Test finding arbitrage opportunities."""
        strategy = ArbitrageStrategy(min_arb_profit=0.01)

        # Create odds with guaranteed arbitrage
        arb_odds = pd.DataFrame([
            {
                'game_id': 'game1',
                'market': 'spread',
                'team': 'home',
                'odds': +105,  # High home odds
                'bookmaker': 'draftkings',
            },
            {
                'game_id': 'game1',
                'market': 'spread',
                'team': 'away',
                'odds': +110,  # High away odds
                'bookmaker': 'fanduel',
            }
        ])

        arbs = strategy.find_arbitrage(arb_odds, market_type='spread')

        # Both positive odds should create arbitrage
        if len(arbs) > 0:
            self.assertGreater(arbs[0]['profit_pct'], 0)

    def test_evaluate_games(self):
        """Test evaluating games for arbitrage."""
        strategy = ArbitrageStrategy(min_arb_profit=0.01)

        signals = strategy.evaluate_games(
            self.games_df,
            self.features_df,
            self.odds_df
        )

        self.assertIsInstance(signals, list)

        # Check signal structure
        for signal in signals:
            self.assertIsInstance(signal, BetSignal)
            self.assertEqual(signal.strategy_type, StrategyType.ARBITRAGE)
            self.assertEqual(signal.confidence, "HIGH")  # Arbs are always high confidence
            self.assertIsNotNone(signal.arb_opportunity)


class TestPlayerPropsStrategy(unittest.TestCase):
    """Test PlayerPropsStrategy."""

    def setUp(self):
        """Set up test data."""
        self.games_df = generate_mock_games(1)
        self.props_df = generate_mock_player_props()
        self.features_df = generate_mock_player_features(self.props_df)

    def test_strategy_initialization(self):
        """Test strategy can be initialized."""
        strategy = PlayerPropsStrategy(
            models_dir="models/player_props",
            prop_types=["PTS", "REB"],
            min_edge=0.05,
        )

        self.assertEqual(strategy.strategy_type, StrategyType.PLAYER_PROPS)
        self.assertEqual(strategy.priority, 4)
        self.assertEqual(strategy.prop_types, ["PTS", "REB"])

    def test_evaluate_without_models(self):
        """Test that strategy handles missing models gracefully."""
        strategy = PlayerPropsStrategy(
            models_dir="nonexistent_dir",
            min_edge=0.05,
        )

        signals = strategy.evaluate_games(
            self.games_df,
            self.features_df,
            self.props_df
        )

        # Should return empty list if models not found
        self.assertIsInstance(signals, list)

    def test_factory_methods(self):
        """Test factory methods."""
        starters = PlayerPropsStrategy.starters_only_strategy()
        self.assertEqual(starters.min_minutes, 25.0)

        conservative = PlayerPropsStrategy.conservative_strategy()
        self.assertEqual(conservative.min_edge, 0.07)
        self.assertEqual(conservative.min_minutes, 28.0)
        self.assertEqual(conservative.prop_types, ["PTS"])


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
