"""
Live Edge Detector

Detect profitable live betting opportunities by comparing
model probabilities vs market odds.
"""

import sqlite3
from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger

from src.models.live_win_probability import LiveWinProbModel
from src.data.live_betting_db import DB_PATH


class LiveEdgeDetector:
    """Detect profitable live betting edges."""

    def __init__(
        self,
        min_edge: float = 0.05,  # 5% minimum edge
        min_confidence: float = 0.5,  # 50% minimum model confidence
    ):
        """
        Initialize edge detector.

        Args:
            min_edge: Minimum edge to flag (default 5%)
            min_confidence: Minimum model confidence (default 50%)
        """
        self.min_edge = min_edge
        self.min_confidence = min_confidence
        self.win_prob_model = LiveWinProbModel()

    def find_edges(
        self,
        game_state: Dict,
        live_odds: Dict,
        win_prob: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Find betting edges by comparing model vs market.

        Args:
            game_state: Current game state with score, quarter, time
            live_odds: Dict with 'spread', 'moneyline', 'total' odds
            win_prob: Optional pre-calculated win probabilities

        Returns:
            List of edge opportunities
        """
        edges = []

        # Calculate win probability if not provided
        if win_prob is None:
            win_prob = self.win_prob_model.predict(game_state)

        # Only proceed if model is confident enough
        if win_prob['confidence'] < self.min_confidence:
            logger.debug(f"Skipping edge detection - low confidence: {win_prob['confidence']:.1%}")
            return edges

        # Check spread edges
        if 'spread' in live_odds and live_odds['spread']:
            spread_edges = self._check_spread_edge(game_state, live_odds['spread'], win_prob)
            edges.extend(spread_edges)

        # Check moneyline edges
        if 'moneyline' in live_odds and live_odds['moneyline']:
            ml_edges = self._check_moneyline_edge(game_state, live_odds['moneyline'], win_prob)
            edges.extend(ml_edges)

        # Check total edges
        if 'total' in live_odds and live_odds['total']:
            total_edges = self._check_total_edge(game_state, live_odds['total'])
            edges.extend(total_edges)

        # Filter by minimum edge
        edges = [e for e in edges if e['edge'] >= self.min_edge]

        # Add common fields
        for edge in edges:
            edge['detected_at'] = datetime.now().isoformat()
            edge['game_id'] = game_state['game_id']
            edge['home_team'] = game_state.get('home_team', '')
            edge['away_team'] = game_state.get('away_team', '')

        return edges

    def _check_spread_edge(
        self,
        game_state: Dict,
        spread_odds: Dict,
        win_prob: Dict
    ) -> List[Dict]:
        """
        Check for spread betting edges.

        Args:
            game_state: Game state
            spread_odds: {'spread_value': -5.5, 'home_odds': -110, 'away_odds': -110}
            win_prob: Win probabilities

        Returns:
            List of spread edges
        """
        edges = []

        spread = spread_odds.get('spread_value')
        if spread is None:
            return edges

        home_odds = spread_odds.get('home_odds', -110)
        away_odds = spread_odds.get('away_odds', -110)

        # Get cover probabilities
        cover_prob = self.win_prob_model.predict_spread_cover(game_state, spread)

        home_cover_prob = cover_prob['home_cover_prob']
        away_cover_prob = cover_prob['away_cover_prob']

        # Convert odds to implied probabilities
        home_market_prob = self._odds_to_prob(home_odds)
        away_market_prob = self._odds_to_prob(away_odds)

        # Calculate edges
        home_edge = home_cover_prob - home_market_prob
        away_edge = away_cover_prob - away_market_prob

        # Check home spread edge
        if abs(home_edge) >= self.min_edge:
            bet_side = 'home' if home_edge > 0 else 'away'
            model_prob = home_cover_prob if home_edge > 0 else away_cover_prob
            market_prob = home_market_prob if home_edge > 0 else away_market_prob
            odds = home_odds if home_edge > 0 else away_odds

            edges.append({
                'alert_type': 'spread',
                'bet_side': bet_side,
                'model_prob': model_prob,
                'market_prob': market_prob,
                'edge': abs(home_edge),
                'line_value': spread if bet_side == 'home' else -spread,
                'odds': odds,
                'quarter': game_state['quarter'],
                'score_diff': game_state['home_score'] - game_state['away_score'],
                'time_remaining': game_state.get('time_remaining', ''),
                'home_score': game_state['home_score'],
                'away_score': game_state['away_score'],
                'confidence': self._determine_confidence(abs(home_edge), win_prob['confidence'])
            })

        return edges

    def _check_moneyline_edge(
        self,
        game_state: Dict,
        ml_odds: Dict,
        win_prob: Dict
    ) -> List[Dict]:
        """
        Check for moneyline betting edges.

        Args:
            game_state: Game state
            ml_odds: {'home_odds': -200, 'away_odds': +170}
            win_prob: Win probabilities

        Returns:
            List of moneyline edges
        """
        edges = []

        home_odds = ml_odds.get('home_odds')
        away_odds = ml_odds.get('away_odds')

        if home_odds is None or away_odds is None:
            return edges

        # Model win probabilities
        home_win_prob = win_prob['home_win_prob']
        away_win_prob = win_prob['away_win_prob']

        # Market implied probabilities
        home_market_prob = self._odds_to_prob(home_odds)
        away_market_prob = self._odds_to_prob(away_odds)

        # Calculate edges
        home_edge = home_win_prob - home_market_prob
        away_edge = away_win_prob - away_market_prob

        # Check for significant edge
        if abs(home_edge) >= self.min_edge:
            bet_side = 'home' if home_edge > 0 else 'away'
            model_prob = home_win_prob if home_edge > 0 else away_win_prob
            market_prob = home_market_prob if home_edge > 0 else away_market_prob
            odds = home_odds if home_edge > 0 else away_odds

            edges.append({
                'alert_type': 'moneyline',
                'bet_side': bet_side,
                'model_prob': model_prob,
                'market_prob': market_prob,
                'edge': abs(home_edge),
                'line_value': None,  # No line for moneyline
                'odds': odds,
                'quarter': game_state['quarter'],
                'score_diff': game_state['home_score'] - game_state['away_score'],
                'time_remaining': game_state.get('time_remaining', ''),
                'home_score': game_state['home_score'],
                'away_score': game_state['away_score'],
                'confidence': self._determine_confidence(abs(home_edge), win_prob['confidence'])
            })

        return edges

    def _check_total_edge(
        self,
        game_state: Dict,
        total_odds: Dict
    ) -> List[Dict]:
        """
        Check for total (over/under) betting edges.

        Args:
            game_state: Game state
            total_odds: {'total_value': 215.5, 'over_odds': -110, 'under_odds': -110}

        Returns:
            List of total edges
        """
        edges = []

        total_line = total_odds.get('total_value')
        if total_line is None:
            return edges

        over_odds = total_odds.get('over_odds', -110)
        under_odds = total_odds.get('under_odds', -110)

        # Get total probabilities
        total_prob = self.win_prob_model.predict_total(game_state, total_line)

        over_prob = total_prob['over_prob']
        under_prob = total_prob['under_prob']

        # Market implied probabilities
        over_market_prob = self._odds_to_prob(over_odds)
        under_market_prob = self._odds_to_prob(under_odds)

        # Calculate edges
        over_edge = over_prob - over_market_prob
        under_edge = under_prob - under_market_prob

        # Check for significant edge
        if abs(over_edge) >= self.min_edge:
            bet_side = 'over' if over_edge > 0 else 'under'
            model_prob = over_prob if over_edge > 0 else under_prob
            market_prob = over_market_prob if over_edge > 0 else under_market_prob
            odds = over_odds if over_edge > 0 else under_odds

            edges.append({
                'alert_type': 'total',
                'bet_side': bet_side,
                'model_prob': model_prob,
                'market_prob': market_prob,
                'edge': abs(over_edge),
                'line_value': total_line,
                'odds': odds,
                'quarter': game_state['quarter'],
                'score_diff': game_state['home_score'] - game_state['away_score'],
                'time_remaining': game_state.get('time_remaining', ''),
                'home_score': game_state['home_score'],
                'away_score': game_state['away_score'],
                'confidence': self._determine_confidence(abs(over_edge), 0.7)  # Moderate confidence for totals
            })

        return edges

    def _odds_to_prob(self, american_odds: int) -> float:
        """
        Convert American odds to implied probability.

        Args:
            american_odds: American odds (e.g., -110, +150)

        Returns:
            Implied probability (0-1)
        """
        if american_odds < 0:
            # Favorite: -110 means 110/210 = 52.4%
            return abs(american_odds) / (abs(american_odds) + 100)
        else:
            # Underdog: +150 means 100/250 = 40%
            return 100 / (american_odds + 100)

    def _determine_confidence(self, edge: float, model_confidence: float) -> str:
        """
        Determine confidence level for alert.

        Args:
            edge: Size of edge (0-1)
            model_confidence: Model's confidence (0-1)

        Returns:
            'HIGH', 'MEDIUM', or 'LOW'
        """
        # Combine edge size and model confidence
        combined_score = (edge * 0.6) + (model_confidence * 0.4)

        if combined_score >= 0.12:  # 12%+ combined score
            return 'HIGH'
        elif combined_score >= 0.08:  # 8-12% combined score
            return 'MEDIUM'
        else:
            return 'LOW'

    def save_alert(self, edge: Dict) -> int:
        """
        Save edge alert to database.

        Args:
            edge: Edge dict

        Returns:
            Alert ID
        """
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.execute("""
            INSERT INTO live_edge_alerts
            (game_id, timestamp, alert_type, bet_side, model_prob, market_prob,
             edge, quarter, score_diff, time_remaining, home_score, away_score,
             home_team, away_team, line_value, odds, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            edge['game_id'],
            edge['detected_at'],
            edge['alert_type'],
            edge['bet_side'],
            edge['model_prob'],
            edge['market_prob'],
            edge['edge'],
            edge['quarter'],
            edge['score_diff'],
            edge['time_remaining'],
            edge['home_score'],
            edge['away_score'],
            edge.get('home_team', ''),
            edge.get('away_team', ''),
            edge.get('line_value'),
            edge.get('odds'),
            edge['confidence']
        ))

        alert_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.success(
            f"Alert #{alert_id}: {edge['alert_type'].upper()} {edge['bet_side'].upper()} "
            f"edge={edge['edge']:.1%} confidence={edge['confidence']}"
        )

        return alert_id

    def get_recent_alerts(
        self,
        hours: int = 24,
        min_confidence: Optional[str] = None
    ) -> List[Dict]:
        """
        Get recent edge alerts.

        Args:
            hours: Look back N hours
            min_confidence: Optional filter ('HIGH', 'MEDIUM', 'LOW')

        Returns:
            List of alerts
        """
        from datetime import timedelta

        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        query = """
            SELECT * FROM live_edge_alerts
            WHERE timestamp >= ?
        """
        params = [cutoff]

        if min_confidence:
            # Filter by confidence hierarchy
            if min_confidence == 'HIGH':
                query += " AND confidence = 'HIGH'"
            elif min_confidence == 'MEDIUM':
                query += " AND confidence IN ('HIGH', 'MEDIUM')"

        query += " ORDER BY timestamp DESC"

        results = conn.execute(query, params).fetchall()
        conn.close()

        return [dict(row) for row in results]


if __name__ == "__main__":
    # Test edge detection
    detector = LiveEdgeDetector(min_edge=0.05)

    print("=== Live Edge Detector Tests ===\n")

    # Test Case 1: Spread edge opportunity
    print("Test 1: Spread Edge - Home up 8, spread -5.5")
    game1 = {
        'game_id': 'test_game_1',
        'home_team': 'LAL',
        'away_team': 'GSW',
        'home_score': 108,
        'away_score': 100,
        'quarter': 4,
        'time_remaining': '5:00'
    }
    odds1 = {
        'spread': {
            'spread_value': -5.5,
            'home_odds': -110,
            'away_odds': -110
        }
    }

    edges1 = detector.find_edges(game1, odds1)
    print(f"  Found {len(edges1)} edges")
    for edge in edges1:
        print(f"  - {edge['alert_type'].upper()} {edge['bet_side'].upper()}")
        print(f"    Edge: {edge['edge']:.1%}, Confidence: {edge['confidence']}")
        print(f"    Model: {edge['model_prob']:.1%}, Market: {edge['market_prob']:.1%}\n")

    # Test Case 2: Moneyline edge - underdog winning late
    print("Test 2: Moneyline Edge - Underdog up late")
    game2 = {
        'game_id': 'test_game_2',
        'home_team': 'DET',
        'away_team': 'BOS',
        'home_score': 112,
        'away_score': 108,
        'quarter': 4,
        'time_remaining': '2:30'
    }
    odds2 = {
        'moneyline': {
            'home_odds': +150,  # Was underdog pre-game
            'away_odds': -180   # Was favorite pre-game
        }
    }

    edges2 = detector.find_edges(game2, odds2)
    print(f"  Found {len(edges2)} edges")
    for edge in edges2:
        print(f"  - {edge['alert_type'].upper()} {edge['bet_side'].upper()}")
        print(f"    Edge: {edge['edge']:.1%}, Confidence: {edge['confidence']}")
        print(f"    Model: {edge['model_prob']:.1%}, Market: {edge['market_prob']:.1%}")
        print(f"    Odds: {edge['odds']:+d}\n")

    # Test Case 3: Total edge - high scoring game
    print("Test 3: Total Edge - High pace game")
    game3 = {
        'game_id': 'test_game_3',
        'home_team': 'PHX',
        'away_team': 'SAC',
        'home_score': 95,
        'away_score': 92,
        'quarter': 2,
        'time_remaining': '0:00'  # Halftime
    }
    odds3 = {
        'total': {
            'total_value': 220.5,  # Line set expecting lower scoring
            'over_odds': -110,
            'under_odds': -110
        }
    }

    edges3 = detector.find_edges(game3, odds3)
    print(f"  Found {len(edges3)} edges")
    for edge in edges3:
        print(f"  - {edge['alert_type'].upper()} {edge['bet_side'].upper()}")
        print(f"    Edge: {edge['edge']:.1%}, Confidence: {edge['confidence']}")
        print(f"    Line: {edge['line_value']}")
        print(f"    Current Total: {game3['home_score'] + game3['away_score']}")
