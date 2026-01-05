"""
Live Bet Settlement

Settle live paper bets based on final game results.
"""

import sqlite3
from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger

from src.data.live_betting_db import DB_PATH


class LiveBetSettlement:
    """Settle live paper bets."""

    def settle_game_bets(self, game_id: str, final_home_score: int, final_away_score: int) -> int:
        """
        Settle all paper bets for a completed game.

        Args:
            game_id: NBA Stats API game ID
            final_home_score: Final home team score
            final_away_score: Final away team score

        Returns:
            Number of bets settled
        """
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        try:
            # Get all unsettled paper bets for this game
            bets = conn.execute("""
                SELECT * FROM live_paper_bets
                WHERE nba_game_id = ? AND outcome IS NULL
            """, (game_id,)).fetchall()

            settled_count = 0

            for bet in bets:
                outcome = self._determine_outcome(
                    bet_type=bet['bet_type'],
                    bet_side=bet['bet_side'],
                    line_value=bet['line_value'],
                    final_home_score=final_home_score,
                    final_away_score=final_away_score
                )

                # Calculate profit/loss
                stake = bet['stake'] or 100  # Default $100 if not set
                if outcome == 'win':
                    profit = self._calculate_profit(stake, bet['odds'])
                elif outcome == 'loss':
                    profit = -stake
                else:  # push
                    profit = 0

                # Update bet
                conn.execute("""
                    UPDATE live_paper_bets
                    SET outcome = ?,
                        profit = ?,
                        settled_at = ?
                    WHERE id = ?
                """, (outcome, profit, datetime.now().isoformat(), bet['id']))

                settled_count += 1
                logger.info(f"Settled bet #{bet['id']}: {outcome} (profit: ${profit:+.2f})")

            conn.commit()
            logger.info(f"Settled {settled_count} bets for game {game_id}")

            return settled_count

        except Exception as e:
            logger.error(f"Error settling bets: {e}")
            conn.rollback()
            return 0
        finally:
            conn.close()

    def _determine_outcome(
        self,
        bet_type: str,
        bet_side: str,
        line_value: Optional[float],
        final_home_score: int,
        final_away_score: int
    ) -> str:
        """
        Determine bet outcome (win/loss/push).

        Args:
            bet_type: 'spread', 'moneyline', or 'total'
            bet_side: 'home', 'away', 'over', 'under'
            line_value: Line value (spread or total)
            final_home_score: Final home score
            final_away_score: Final away score

        Returns:
            'win', 'loss', or 'push'
        """
        if bet_type == 'moneyline':
            if bet_side == 'home':
                return 'win' if final_home_score > final_away_score else 'loss'
            else:  # away
                return 'win' if final_away_score > final_home_score else 'loss'

        elif bet_type == 'spread':
            if line_value is None:
                return 'push'

            # Calculate against the spread result
            # If betting home -5.5, home needs to win by 6+
            # If betting away +5.5, away can lose by 5 or less
            home_ats_margin = final_home_score - final_away_score

            if bet_side == 'home':
                # Home covering the spread
                ats_result = home_ats_margin + line_value
                if ats_result > 0:
                    return 'win'
                elif ats_result < 0:
                    return 'loss'
                else:
                    return 'push'
            else:  # away
                # Away covering the spread (line is inverted)
                ats_result = home_ats_margin + line_value
                if ats_result < 0:
                    return 'win'
                elif ats_result > 0:
                    return 'loss'
                else:
                    return 'push'

        elif bet_type == 'total':
            if line_value is None:
                return 'push'

            total_points = final_home_score + final_away_score

            if bet_side == 'over':
                if total_points > line_value:
                    return 'win'
                elif total_points < line_value:
                    return 'loss'
                else:
                    return 'push'
            else:  # under
                if total_points < line_value:
                    return 'win'
                elif total_points > line_value:
                    return 'loss'
                else:
                    return 'push'

        return 'push'  # Default fallback

    def _calculate_profit(self, stake: float, american_odds: int) -> float:
        """
        Calculate profit from American odds.

        Args:
            stake: Bet amount
            american_odds: American odds (e.g., -110, +150)

        Returns:
            Profit (not including stake)
        """
        if american_odds > 0:
            # Positive odds: profit = stake * (odds / 100)
            return stake * (american_odds / 100)
        else:
            # Negative odds: profit = stake / (abs(odds) / 100)
            return stake / (abs(american_odds) / 100)

    def get_settlement_stats(self) -> Dict:
        """
        Get overall settlement statistics.

        Returns:
            Dict with settlement stats
        """
        conn = sqlite3.connect(DB_PATH)

        try:
            # Overall stats
            stats = conn.execute("""
                SELECT
                    COUNT(*) as total_bets,
                    SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
                    SUM(CASE WHEN outcome = 'push' THEN 1 ELSE 0 END) as pushes,
                    SUM(CASE WHEN outcome IS NULL THEN 1 ELSE 0 END) as unsettled,
                    SUM(profit) as total_profit,
                    AVG(profit) as avg_profit
                FROM live_paper_bets
            """).fetchone()

            # Stats by confidence
            confidence_stats = conn.execute("""
                SELECT
                    confidence,
                    COUNT(*) as bets,
                    SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                    SUM(profit) as profit
                FROM live_paper_bets
                WHERE outcome IS NOT NULL
                GROUP BY confidence
                ORDER BY confidence DESC
            """).fetchall()

            return {
                'total_bets': stats[0] or 0,
                'wins': stats[1] or 0,
                'losses': stats[2] or 0,
                'pushes': stats[3] or 0,
                'unsettled': stats[4] or 0,
                'total_profit': stats[5] or 0,
                'avg_profit': stats[6] or 0,
                'win_rate': (stats[1] / max(stats[0] - (stats[3] or 0) - (stats[4] or 0), 1)) if stats[0] else 0,
                'confidence_breakdown': [
                    {
                        'confidence': row[0],
                        'bets': row[1],
                        'wins': row[2],
                        'profit': row[3],
                        'win_rate': row[2] / row[1] if row[1] > 0 else 0
                    }
                    for row in confidence_stats
                ]
            }

        finally:
            conn.close()

    def auto_settle_completed_games(self) -> int:
        """
        Automatically settle bets for games that have finished.

        Returns:
            Number of games settled
        """
        from src.data.live_game_client import LiveGameClient

        client = LiveGameClient()
        conn = sqlite3.connect(DB_PATH)

        try:
            # Get games with unsettled bets
            unsettled_games = conn.execute("""
                SELECT DISTINCT nba_game_id
                FROM live_paper_bets
                WHERE outcome IS NULL
            """).fetchall()

            settled_games = 0

            for (game_id,) in unsettled_games:
                # Check if game is finished
                game_state = conn.execute("""
                    SELECT home_score, away_score, game_status
                    FROM live_game_state
                    WHERE game_id = ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, (game_id,)).fetchone()

                if game_state and game_state[2] == 'Final':
                    # Game is complete, settle bets
                    self.settle_game_bets(game_id, game_state[0], game_state[1])
                    settled_games += 1

            return settled_games

        finally:
            conn.close()
