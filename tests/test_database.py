"""
Database Integration Tests

Tests that the database schema updates work correctly
and that bets can be logged with new strategy fields.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import sqlite3
import tempfile
from datetime import datetime

from src.bet_tracker import log_bet, DB_PATH


class TestDatabaseSchema(unittest.TestCase):
    """Test database schema and bet logging."""

    def setUp(self):
        """Set up temporary database for testing."""
        # Use a temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db_path = self.temp_db.name
        self.temp_db.close()

        # Monkey patch DB_PATH for testing
        import src.bet_tracker as bt
        self.original_db_path = bt.DB_PATH
        bt.DB_PATH = self.temp_db_path

    def tearDown(self):
        """Clean up temporary database."""
        import src.bet_tracker as bt
        bt.DB_PATH = self.original_db_path

        # Remove temp file
        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)

    def test_database_creation(self):
        """Test that database and tables are created."""
        import src.bet_tracker as bt
        conn = bt._get_connection()

        # Check bets table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='bets'"
        )
        result = cursor.fetchone()
        self.assertIsNotNone(result)

        conn.close()

    def test_strategy_columns_exist(self):
        """Test that new strategy columns exist."""
        import src.bet_tracker as bt
        conn = bt._get_connection()

        cursor = conn.execute("PRAGMA table_info(bets)")
        columns = {row[1] for row in cursor.fetchall()}

        # Check new columns
        self.assertIn('strategy_type', columns)
        self.assertIn('player_id', columns)
        self.assertIn('player_name', columns)
        self.assertIn('prop_type', columns)

        conn.close()

    def test_log_spread_bet(self):
        """Test logging a spread bet."""
        import src.bet_tracker as bt

        bet = bt.log_bet(
            game_id='test_game_1',
            home_team='LAL',
            away_team='BOS',
            commence_time=datetime.now().isoformat(),
            bet_type='spread',
            bet_side='home',
            odds=-110,
            line=-5.5,
            model_prob=0.58,
            market_prob=0.52,
            edge=0.06,
            kelly=0.15,
            bet_amount=50.0,
            strategy_type='spread',
        )

        self.assertIsNotNone(bet)
        self.assertEqual(bet['bet_type'], 'spread')
        self.assertEqual(bet['strategy_type'], 'spread')

    def test_log_totals_bet(self):
        """Test logging a totals bet."""
        import src.bet_tracker as bt

        bet = bt.log_bet(
            game_id='test_game_2',
            home_team='GSW',
            away_team='DEN',
            commence_time=datetime.now().isoformat(),
            bet_type='totals',
            bet_side='over',
            odds=-115,
            line=225.5,
            model_prob=0.55,
            market_prob=0.49,
            edge=0.06,
            kelly=0.12,
            bet_amount=40.0,
            strategy_type='totals',
        )

        self.assertIsNotNone(bet)
        self.assertEqual(bet['bet_type'], 'totals')
        self.assertEqual(bet['strategy_type'], 'totals')

    def test_log_player_prop_bet(self):
        """Test logging a player prop bet."""
        import src.bet_tracker as bt

        bet = bt.log_bet(
            game_id='test_game_3',
            home_team='MIL',
            away_team='PHX',
            commence_time=datetime.now().isoformat(),
            bet_type='player_prop',
            bet_side='over',
            odds=-110,
            line=27.5,
            model_prob=0.57,
            market_prob=0.51,
            edge=0.06,
            kelly=0.14,
            bet_amount=45.0,
            strategy_type='player_props',
            player_id='12345',
            player_name='Giannis Antetokounmpo',
            prop_type='PTS',
        )

        self.assertIsNotNone(bet)
        self.assertEqual(bet['bet_type'], 'player_prop')
        self.assertEqual(bet['strategy_type'], 'player_props')
        self.assertEqual(bet['player_name'], 'Giannis Antetokounmpo')
        self.assertEqual(bet['prop_type'], 'PTS')

    def test_query_by_strategy(self):
        """Test querying bets by strategy type."""
        import src.bet_tracker as bt

        # Log bets from different strategies
        bt.log_bet(
            game_id='game1',
            home_team='LAL',
            away_team='BOS',
            commence_time=datetime.now().isoformat(),
            bet_type='totals',
            bet_side='over',
            odds=-110,
            line=220.5,
            model_prob=0.55,
            market_prob=0.50,
            edge=0.05,
            kelly=0.10,
            strategy_type='totals',
        )

        bt.log_bet(
            game_id='game2',
            home_team='GSW',
            away_team='DEN',
            commence_time=datetime.now().isoformat(),
            bet_type='spread',
            bet_side='home',
            odds=-110,
            line=-3.5,
            model_prob=0.56,
            market_prob=0.50,
            edge=0.06,
            kelly=0.12,
            strategy_type='arbitrage',
        )

        # Query by strategy
        conn = bt._get_connection()

        totals_bets = conn.execute(
            "SELECT * FROM bets WHERE strategy_type = ?",
            ('totals',)
        ).fetchall()

        self.assertGreater(len(totals_bets), 0)

        arb_bets = conn.execute(
            "SELECT * FROM bets WHERE strategy_type = ?",
            ('arbitrage',)
        ).fetchall()

        self.assertGreater(len(arb_bets), 0)

        conn.close()


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
