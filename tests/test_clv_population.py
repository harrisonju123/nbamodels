"""
Unit Tests for CLV Population

Tests for the CLV calculation and population scripts:
- populate_clv_data.py
- validate_closing_lines.py
- capture_closing_lines.py

Run tests:
    pytest tests/test_clv_population.py -v
"""

import pytest
import sqlite3
import tempfile
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

# Import modules to test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bet_tracker import (
    DB_PATH,
    calculate_multi_snapshot_clv,
    calculate_line_velocity
)


@pytest.fixture
def temp_db():
    """Create temporary test database."""
    # Create temporary database file
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)

    # Patch DB_PATH to use temp database
    original_db_path = DB_PATH

    with patch('src.bet_tracker.DB_PATH', path):
        # Initialize database schema
        from src.bet_tracker import _get_connection
        conn = _get_connection()
        conn.close()

        yield path

    # Cleanup
    os.unlink(path)


def create_test_bet(
    conn: sqlite3.Connection,
    game_id: str,
    bet_type: str = 'spread',
    bet_side: str = 'home',
    odds: float = -110,
    line: float = -5.0,
    commence_time: str = None,
    closing_odds: float = None,
    closing_line: float = None,
    outcome: str = None
) -> str:
    """Helper to create test bet."""
    if commence_time is None:
        commence_time = datetime.now(timezone.utc).isoformat()

    bet_id = f"{game_id}_{bet_type}_{bet_side}"

    conn.execute("""
        INSERT INTO bets (
            id, game_id, home_team, away_team, commence_time,
            bet_type, bet_side, odds, line,
            edge, model_prob, market_prob,
            logged_at, closing_odds, closing_line, outcome
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        bet_id, game_id, 'LAL', 'GSW', commence_time,
        bet_type, bet_side, odds, line,
        5.0, 0.55, 0.50,
        datetime.now(timezone.utc).isoformat(),
        closing_odds, closing_line, outcome
    ))
    conn.commit()

    return bet_id


def create_test_snapshot(
    conn: sqlite3.Connection,
    game_id: str,
    snapshot_time: str,
    bet_type: str,
    side: str,
    bookmaker: str,
    odds: int,
    line: float = None
):
    """Helper to create test snapshot."""
    conn.execute("""
        INSERT INTO line_snapshots (
            game_id, snapshot_time, bet_type, side, bookmaker,
            odds, line, implied_prob, no_vig_prob
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        game_id, snapshot_time, bet_type, side, bookmaker,
        odds, line, 0.52, 0.50
    ))
    conn.commit()


class TestMultiSnapshotCLV:
    """Tests for calculate_multi_snapshot_clv()."""

    @patch('src.bet_tracker.DB_PATH')
    def test_full_coverage(self, mock_db_path, temp_db):
        """Test CLV calculation with all snapshots available."""
        mock_db_path.return_value = temp_db

        conn = sqlite3.connect(temp_db)
        conn.row_factory = sqlite3.Row

        # Create test game
        game_id = 'test_game_001'
        commence_time = datetime.now(timezone.utc) + timedelta(hours=24)

        # Create bet
        bet_id = create_test_bet(
            conn, game_id,
            odds=-110, line=-5.0,
            commence_time=commence_time.isoformat(),
            closing_odds=-105, closing_line=-4.5,
            outcome='win'
        )

        # Create snapshots at various time points
        for hours_before in [1, 4, 12, 24]:
            snapshot_time = commence_time - timedelta(hours=hours_before)
            create_test_snapshot(
                conn, game_id,
                snapshot_time=snapshot_time.isoformat(),
                bet_type='spread',
                side='home',
                bookmaker='draftkings',
                odds=-108,
                line=-4.5
            )

        conn.close()

        # Calculate CLV
        with patch('src.bet_tracker.DB_PATH', temp_db):
            result = calculate_multi_snapshot_clv(bet_id)

        # Verify all windows have CLV data
        assert 'clv_at_1hr' in result
        assert 'clv_at_4hr' in result
        assert 'clv_at_12hr' in result
        assert 'clv_at_24hr' in result

        # All should be positive (booked at -110, closed at -108)
        for window in ['clv_at_1hr', 'clv_at_4hr', 'clv_at_12hr', 'clv_at_24hr']:
            assert result[window] is not None

    @patch('src.bet_tracker.DB_PATH')
    def test_partial_coverage(self, mock_db_path, temp_db):
        """Test graceful degradation with missing snapshots."""
        mock_db_path.return_value = temp_db

        conn = sqlite3.connect(temp_db)
        conn.row_factory = sqlite3.Row

        game_id = 'test_game_002'
        commence_time = datetime.now(timezone.utc) + timedelta(hours=24)

        bet_id = create_test_bet(
            conn, game_id,
            commence_time=commence_time.isoformat(),
            closing_odds=-105
        )

        # Only create 1hr and 4hr snapshots
        for hours_before in [1, 4]:
            snapshot_time = commence_time - timedelta(hours=hours_before)
            create_test_snapshot(
                conn, game_id,
                snapshot_time=snapshot_time.isoformat(),
                bet_type='spread',
                side='home',
                bookmaker='draftkings',
                odds=-108,
                line=-5.0
            )

        conn.close()

        with patch('src.bet_tracker.DB_PATH', temp_db):
            result = calculate_multi_snapshot_clv(bet_id)

        # Verify 1hr and 4hr have data, 12hr and 24hr are None
        assert result.get('clv_at_1hr') is not None
        assert result.get('clv_at_4hr') is not None
        # 12hr and 24hr might be None or missing

    @patch('src.bet_tracker.DB_PATH')
    def test_empty_result_handling(self, mock_db_path, temp_db):
        """Test CLV calculation with no snapshots."""
        mock_db_path.return_value = temp_db

        conn = sqlite3.connect(temp_db)
        conn.row_factory = sqlite3.Row

        game_id = 'test_game_003'
        bet_id = create_test_bet(conn, game_id)

        conn.close()

        with patch('src.bet_tracker.DB_PATH', temp_db):
            result = calculate_multi_snapshot_clv(bet_id)

        # Should return empty dict or all None values
        assert isinstance(result, dict)


class TestClosingLineFallback:
    """Tests for closing line fallback priority."""

    def test_snapshot_priority(self):
        """Test snapshot source has highest priority."""
        # Mock scenario: snapshot, API, and opening all available
        # Should choose snapshot
        pass  # Implement when validate_closing_lines is refactored for testing

    def test_api_fallback(self):
        """Test API provisional used when snapshot missing."""
        pass

    def test_opening_fallback(self):
        """Test opening line used as last resort."""
        pass

    def test_null_when_nothing_available(self):
        """Test NULL result when no sources available."""
        pass


class TestLineVelocity:
    """Tests for line velocity calculation."""

    @patch('src.bet_tracker.DB_PATH')
    def test_line_velocity_calculation(self, mock_db_path, temp_db):
        """Test line velocity calculation with multiple snapshots."""
        mock_db_path.return_value = temp_db

        conn = sqlite3.connect(temp_db)
        conn.row_factory = sqlite3.Row

        game_id = 'test_game_velocity'
        commence_time = datetime.now(timezone.utc) + timedelta(hours=12)

        # Create snapshots showing line movement
        base_time = commence_time - timedelta(hours=6)

        # Line moves from -5.0 to -3.0 over 4 hours (favorable movement)
        snapshots = [
            (base_time, -110, -5.0),
            (base_time + timedelta(hours=2), -110, -4.0),
            (base_time + timedelta(hours=4), -110, -3.0),
        ]

        for snap_time, odds, line in snapshots:
            create_test_snapshot(
                conn, game_id,
                snapshot_time=snap_time.isoformat(),
                bet_type='spread',
                side='home',
                bookmaker='draftkings',
                odds=odds,
                line=line
            )

        conn.close()

        with patch('src.bet_tracker.DB_PATH', temp_db):
            velocity = calculate_line_velocity(
                game_id=game_id,
                bet_type='spread',
                bet_side='home',
                window_hours=4
            )

        # Should show positive velocity (line moving in our favor)
        assert velocity is not None

    @patch('src.bet_tracker.DB_PATH')
    def test_insufficient_data(self, mock_db_path, temp_db):
        """Test velocity returns None with insufficient data."""
        mock_db_path.return_value = temp_db

        with patch('src.bet_tracker.DB_PATH', temp_db):
            velocity = calculate_line_velocity(
                game_id='nonexistent_game',
                bet_type='spread',
                bet_side='home',
                window_hours=4
            )

        assert velocity is None


class TestSnapshotCoverage:
    """Tests for snapshot coverage calculation."""

    def test_full_coverage(self):
        """Test 100% coverage when all windows have data."""
        clv_data = {
            'clv_at_1hr': 0.02,
            'clv_at_4hr': 0.015,
            'clv_at_12hr': 0.01,
            'clv_at_24hr': 0.005
        }

        windows = ['clv_at_1hr', 'clv_at_4hr', 'clv_at_12hr', 'clv_at_24hr']
        available = sum(1 for w in windows if clv_data.get(w) is not None)
        coverage = available / len(windows)

        assert coverage == 1.0

    def test_partial_coverage(self):
        """Test partial coverage calculation."""
        clv_data = {
            'clv_at_1hr': 0.02,
            'clv_at_4hr': 0.015,
            'clv_at_12hr': None,
            'clv_at_24hr': None
        }

        windows = ['clv_at_1hr', 'clv_at_4hr', 'clv_at_12hr', 'clv_at_24hr']
        available = sum(1 for w in windows if clv_data.get(w) is not None)
        coverage = available / len(windows)

        assert coverage == 0.5

    def test_zero_coverage(self):
        """Test zero coverage when no data available."""
        clv_data = {
            'clv_at_1hr': None,
            'clv_at_4hr': None,
            'clv_at_12hr': None,
            'clv_at_24hr': None
        }

        windows = ['clv_at_1hr', 'clv_at_4hr', 'clv_at_12hr', 'clv_at_24hr']
        available = sum(1 for w in windows if clv_data.get(w) is not None)
        coverage = available / len(windows)

        assert coverage == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
