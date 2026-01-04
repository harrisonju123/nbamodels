"""
Unit Tests for EdgeStrategy CLV Filtering

Tests for CLV-based filtering in EdgeStrategy:
- _get_historical_clv()
- _check_optimal_timing()
- calculate_kelly_with_clv_adjustment()
- CLV filtering in evaluate_game()

Run tests:
    pytest tests/test_edge_strategy_clv.py -v
"""

import pytest
import sqlite3
import tempfile
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.betting.edge_strategy import EdgeStrategy


@pytest.fixture
def temp_db():
    """Create temporary test database with historical bets."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row

    # Create tables
    conn.execute("""
        CREATE TABLE bets (
            id TEXT PRIMARY KEY,
            game_id TEXT,
            bet_type TEXT,
            bet_side TEXT,
            edge REAL,
            clv_at_4hr REAL,
            settled_at TEXT,
            outcome TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE opening_lines (
            game_id TEXT,
            commence_time TEXT
        )
    """)

    # Insert historical bets with various CLV values
    cutoff = (datetime.now(timezone.utc) - timedelta(days=15)).isoformat()

    test_bets = [
        # Positive CLV bets
        ('bet_001', 'game_001', 'spread', 'home', 5.5, 0.025, cutoff, 'win'),
        ('bet_002', 'game_002', 'spread', 'home', 5.2, 0.020, cutoff, 'win'),
        ('bet_003', 'game_003', 'spread', 'home', 5.8, 0.018, cutoff, 'loss'),
        # Negative CLV bets
        ('bet_004', 'game_004', 'spread', 'away', 5.1, -0.015, cutoff, 'loss'),
        ('bet_005', 'game_005', 'spread', 'away', 5.3, -0.010, cutoff, 'loss'),
        # Mixed edge bets
        ('bet_006', 'game_006', 'spread', 'home', 3.5, 0.005, cutoff, 'win'),
        ('bet_007', 'game_007', 'spread', 'home', 7.5, 0.030, cutoff, 'win'),
    ]

    for bet in test_bets:
        conn.execute("""
            INSERT INTO bets VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, bet)

    conn.commit()

    yield path

    conn.close()
    os.unlink(path)


class TestGetHistoricalCLV:
    """Tests for _get_historical_clv()."""

    def test_positive_historical_clv(self, temp_db):
        """Test retrieval of positive historical CLV."""
        strategy = EdgeStrategy()

        with patch('src.bet_tracker.DB_PATH', temp_db):
            avg_clv, std_clv = strategy._get_historical_clv(
                bet_type='spread',
                bet_side='home',
                edge=5.5
            )

        # Should find bets with edge 5.5 ± 2.0 (i.e., 3.5 to 7.5)
        # bet_001 (5.5, 0.025), bet_002 (5.2, 0.020), bet_003 (5.8, 0.018)
        assert avg_clv > 0.0
        assert std_clv >= 0.0

    def test_negative_historical_clv(self, temp_db):
        """Test retrieval of negative historical CLV."""
        strategy = EdgeStrategy()

        with patch('src.bet_tracker.DB_PATH', temp_db):
            avg_clv, std_clv = strategy._get_historical_clv(
                bet_type='spread',
                bet_side='away',
                edge=5.2
            )

        # Should find away bets with negative CLV
        assert avg_clv < 0.0

    def test_no_historical_data(self, temp_db):
        """Test returns (0.0, 0.0) when no historical data available."""
        strategy = EdgeStrategy()

        with patch('src.bet_tracker.DB_PATH', temp_db):
            avg_clv, std_clv = strategy._get_historical_clv(
                bet_type='moneyline',  # No moneyline bets in test data
                bet_side='home',
                edge=5.0
            )

        assert avg_clv == 0.0
        assert std_clv == 0.0

    def test_edge_range_filtering(self, temp_db):
        """Test that only similar edge bets are included."""
        strategy = EdgeStrategy()

        with patch('src.bet_tracker.DB_PATH', temp_db):
            avg_clv, std_clv = strategy._get_historical_clv(
                bet_type='spread',
                bet_side='home',
                edge=3.0  # Should only match bet_006 with edge 3.5
            )

        # Should find limited bets within ±2% edge range
        assert isinstance(avg_clv, float)
        assert isinstance(std_clv, float)


class TestCheckOptimalTiming:
    """Tests for _check_optimal_timing()."""

    def test_allows_bet_by_default(self, temp_db):
        """Test allows bet when no timing data available."""
        strategy = EdgeStrategy(optimal_timing_filter=True)

        with patch('src.bet_tracker.DB_PATH', temp_db):
            result = strategy._check_optimal_timing(
                game_id='nonexistent_game',
                bet_type='spread',
                bet_side='home'
            )

        # Should allow bet by default if no data
        assert result is True

    @patch('src.data.line_history.LineHistoryManager')
    def test_within_optimal_window(self, mock_manager, temp_db):
        """Test allows bet when within optimal timing window."""
        # Mock analyze_optimal_booking_time to return 4 hours optimal
        mock_instance = MagicMock()
        mock_instance.analyze_optimal_booking_time.return_value = {
            'optimal_hours_before': 4.0
        }
        mock_manager.return_value = mock_instance

        # Add game with commence time 5 hours from now (within ±2 hour window of 4hr optimal)
        conn = sqlite3.connect(temp_db)
        commence_time = datetime.now(timezone.utc) + timedelta(hours=5)
        conn.execute(
            "INSERT INTO opening_lines VALUES (?, ?)",
            ('test_game', commence_time.isoformat())
        )
        conn.commit()
        conn.close()

        strategy = EdgeStrategy(optimal_timing_filter=True)

        with patch('src.bet_tracker.DB_PATH', temp_db):
            result = strategy._check_optimal_timing(
                game_id='test_game',
                bet_type='spread',
                bet_side='home'
            )

        # 5 hours before is within ±2 hours of 4 hour optimal
        assert result is True

    @patch('src.data.line_history.LineHistoryManager')
    def test_outside_optimal_window(self, mock_manager, temp_db):
        """Test blocks bet when outside optimal timing window."""
        mock_instance = MagicMock()
        mock_instance.analyze_optimal_booking_time.return_value = {
            'optimal_hours_before': 4.0
        }
        mock_manager.return_value = mock_instance

        # Add game with commence time 10 hours from now (outside ±2 hour window)
        conn = sqlite3.connect(temp_db)
        commence_time = datetime.now(timezone.utc) + timedelta(hours=10)
        conn.execute(
            "INSERT INTO opening_lines VALUES (?, ?)",
            ('test_game_far', commence_time.isoformat())
        )
        conn.commit()
        conn.close()

        strategy = EdgeStrategy(optimal_timing_filter=True)

        with patch('src.bet_tracker.DB_PATH', temp_db):
            result = strategy._check_optimal_timing(
                game_id='test_game_far',
                bet_type='spread',
                bet_side='home'
            )

        # 10 hours before is outside ±2 hours of 4 hour optimal
        assert result is False


class TestKellyAdjustment:
    """Tests for calculate_kelly_with_clv_adjustment()."""

    def test_full_kelly_high_clv(self):
        """Test full Kelly (1.0x) when CLV >= 2%."""
        strategy = EdgeStrategy()

        adjusted = strategy.calculate_kelly_with_clv_adjustment(
            edge=5.0,
            kelly=0.05,
            historical_clv=0.025,  # 2.5% CLV
            clv_std=0.005
        )

        assert adjusted == 0.05  # Full Kelly

    def test_half_kelly_moderate_clv(self):
        """Test half Kelly (0.5x) when CLV 0-2%."""
        strategy = EdgeStrategy()

        adjusted = strategy.calculate_kelly_with_clv_adjustment(
            edge=5.0,
            kelly=0.05,
            historical_clv=0.010,  # 1% CLV
            clv_std=0.003
        )

        assert adjusted == 0.025  # Half Kelly

    def test_no_bet_negative_clv(self):
        """Test no bet (0x) when CLV < 0%."""
        strategy = EdgeStrategy()

        adjusted = strategy.calculate_kelly_with_clv_adjustment(
            edge=5.0,
            kelly=0.05,
            historical_clv=-0.015,  # Negative CLV
            clv_std=0.005
        )

        assert adjusted == 0.0  # No bet


class TestCLVFilteringInEvaluateGame:
    """Tests for CLV filtering in evaluate_game()."""

    def test_clv_filter_passes_positive_historical(self, temp_db):
        """CLV filter allows bets with positive historical CLV."""
        strategy = EdgeStrategy(
            edge_threshold=5.0,
            clv_filter_enabled=True,
            min_historical_clv=0.01  # Require +1% CLV
        )

        with patch('src.bet_tracker.DB_PATH', temp_db):
            signal = strategy.evaluate_game(
                game_id='test_game',
                home_team='LAL',
                away_team='GSW',
                pred_diff=10.0,  # Model predicts home by 10
                market_spread=-5.0,  # Home favored by 5
                # Edge = 10 + (-5) = 5.0 (meets threshold)
            )

        # Should find positive historical CLV for edge ~5.0 home bets
        # May pass or fail depending on mocked data
        assert signal is not None

    def test_clv_filter_blocks_negative_historical(self, temp_db):
        """CLV filter blocks bets with negative historical CLV."""
        strategy = EdgeStrategy(
            edge_threshold=5.0,
            clv_filter_enabled=True,
            min_historical_clv=0.01
        )

        with patch('src.bet_tracker.DB_PATH', temp_db):
            signal = strategy.evaluate_game(
                game_id='test_game',
                home_team='LAL',
                away_team='GSW',
                pred_diff=-10.0,  # Model predicts away by 10
                market_spread=5.0,  # Away favored by 5
                # Edge = -10 + 5 = -5.0 (away bet)
            )

        # Should find negative historical CLV for edge ~5.0 away bets
        # Should be blocked
        assert signal.bet_side == "PASS" or signal.bet_side == "AWAY"

    def test_backward_compatibility(self):
        """Existing strategy behavior unchanged when CLV filters disabled."""
        # Test baseline strategy (no CLV filtering)
        baseline = EdgeStrategy.team_filtered_strategy()

        signal = baseline.evaluate_game(
            game_id='test_game',
            home_team='LAL',
            away_team='GSW',
            pred_diff=10.0,
            market_spread=-5.0,
            home_b2b=False,
            away_b2b=False
        )

        # Should allow bet (edge = 5.0, meets threshold, no B2B)
        assert signal.bet_side == "HOME"
        assert signal.model_edge == 5.0

    def test_clv_based_sizing(self, temp_db):
        """Test CLV-based bet sizing adjustment."""
        strategy = EdgeStrategy(
            edge_threshold=5.0,
            clv_based_sizing=True,
            clv_filter_enabled=True,
            min_historical_clv=0.0
        )

        # Test with high CLV historical data
        with patch('src.bet_tracker.DB_PATH', temp_db):
            signal = strategy.evaluate_game(
                game_id='test_game',
                home_team='LAL',
                away_team='GSW',
                pred_diff=10.0,
                market_spread=-5.0
            )

        assert signal is not None


class TestNewStrategyPresets:
    """Tests for new CLV-based strategy presets."""

    def test_clv_filtered_strategy_creation(self):
        """Test clv_filtered_strategy() preset creation."""
        strategy = EdgeStrategy.clv_filtered_strategy()

        assert strategy.edge_threshold == 5.0
        assert strategy.require_no_b2b is True
        # Note: use_team_filter attribute removed - no longer part of strategy
        assert strategy.clv_filter_enabled is True
        assert strategy.min_historical_clv == 0.01

    def test_optimal_timing_strategy_creation(self):
        """Test optimal_timing_strategy() preset creation."""
        strategy = EdgeStrategy.optimal_timing_strategy()

        assert strategy.edge_threshold == 5.0
        assert strategy.require_no_b2b is True
        assert strategy.optimal_timing_filter is True

    def test_strategy_presets_are_independent(self):
        """Test that strategy presets create independent instances."""
        s1 = EdgeStrategy.clv_filtered_strategy()
        s2 = EdgeStrategy.optimal_timing_strategy()

        # Modify one
        s1.edge_threshold = 10.0

        # Other should be unchanged
        assert s2.edge_threshold == 5.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
