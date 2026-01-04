"""
Live Win Probability Model

Calculate win probability based on current game state.
Uses simple formula approach for MVP, can upgrade to ML model later.
"""

import numpy as np
from typing import Dict, Tuple
from loguru import logger


class LiveWinProbModel:
    """Calculate win probability for live games."""

    def __init__(self):
        """Initialize the model."""
        # Home court advantage baseline (can adjust based on historical data)
        self.home_court_advantage = 0.02  # 2% baseline edge for home team

        # Point value multipliers by quarter
        # Points are more valuable later in game
        self.quarter_multipliers = {
            1: 0.8,   # Q1: Points less predictive
            2: 0.9,   # Q2: Getting more predictive
            3: 1.0,   # Q3: Standard value
            4: 1.3,   # Q4: Points very valuable
            5: 1.5    # OT: Each point critical
        }

    def predict(self, game_state: Dict) -> Dict[str, float]:
        """
        Predict win probability for current game state.

        Args:
            game_state: Dict with:
                - home_score: int
                - away_score: int
                - quarter: int (1-4, 5+ for OT)
                - time_remaining: str ("7:23" format)
                - game_clock: int (seconds remaining in quarter)
                OR
                - period_time_remaining: str (alternative format)

        Returns:
            Dict with:
                - home_win_prob: float (0-1)
                - away_win_prob: float (0-1)
                - confidence: float (0-1)
                - method: str (description of calculation)
        """
        try:
            # Extract game state
            home_score = game_state.get('home_score', 0)
            away_score = game_state.get('away_score', 0)
            quarter = game_state.get('quarter', 1)

            # Parse time remaining
            time_remaining_sec = self._get_time_remaining_seconds(game_state)

            # Calculate win probability
            home_win_prob, confidence = self._calculate_win_prob(
                score_diff=home_score - away_score,
                time_remaining_sec=time_remaining_sec,
                quarter=quarter
            )

            return {
                'home_win_prob': home_win_prob,
                'away_win_prob': 1 - home_win_prob,
                'confidence': confidence,
                'method': 'simple_formula',
                'score_diff': home_score - away_score,
                'time_remaining_sec': time_remaining_sec
            }

        except Exception as e:
            logger.error(f"Error calculating win probability: {e}")
            # Return neutral probabilities on error
            return {
                'home_win_prob': 0.5,
                'away_win_prob': 0.5,
                'confidence': 0.0,
                'method': 'error_fallback',
                'error': str(e)
            }

    def _get_time_remaining_seconds(self, game_state: Dict) -> int:
        """
        Extract time remaining in seconds from game state.

        Args:
            game_state: Game state dict

        Returns:
            Total seconds remaining in game
        """
        quarter = game_state.get('quarter', 1)

        # Check if game_clock is provided (in seconds)
        if 'game_clock' in game_state and game_state['game_clock']:
            time_in_quarter = game_state['game_clock']
        else:
            # Parse time string ("7:23" format)
            time_str = game_state.get('time_remaining') or game_state.get('period_time_remaining', '')
            time_in_quarter = self._parse_time_string(time_str)

        # Calculate total time remaining
        # Each quarter is 12 minutes = 720 seconds
        quarters_left = max(0, 4 - quarter)
        total_time_remaining = (quarters_left * 720) + time_in_quarter

        return total_time_remaining

    def _parse_time_string(self, time_str: str) -> int:
        """
        Parse time string to seconds.

        Args:
            time_str: Time in "M:SS" or "MM:SS" format

        Returns:
            Seconds
        """
        if not time_str or time_str == '':
            return 0

        try:
            parts = time_str.strip().split(':')
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = int(parts[1])
                return minutes * 60 + seconds
        except (ValueError, IndexError):
            logger.warning(f"Could not parse time: {time_str}")

        return 0

    def _calculate_win_prob(
        self,
        score_diff: int,
        time_remaining_sec: int,
        quarter: int
    ) -> Tuple[float, float]:
        """
        Calculate win probability using formula.

        Based on research and historical data:
        - Each point of lead increases win prob by ~2-3%
        - Effect increases as time decreases
        - Home court provides small baseline advantage

        Args:
            score_diff: Home score - away score (positive = home ahead)
            time_remaining_sec: Seconds remaining in game
            quarter: Current quarter

        Returns:
            (win_probability, confidence)
        """
        # Full game is 2880 seconds (48 minutes)
        total_game_time = 2880

        # Time factor: How much of game is left (0 = end, 1 = start)
        time_factor = time_remaining_sec / total_game_time

        # Quarter multiplier: Points more valuable later
        quarter_mult = self.quarter_multipliers.get(min(quarter, 5), 1.0)

        # Point value: How much each point changes win probability
        # Starts at ~2.5% early game, increases to ~5% late game
        base_point_value = 0.025
        time_decay = np.sqrt(1 - time_factor)  # More impact as time runs out
        point_value = base_point_value * (1 + time_decay) * quarter_mult

        # Calculate win probability
        # Start with home court advantage
        win_prob = 0.5 + self.home_court_advantage

        # Add score differential effect
        win_prob += score_diff * point_value

        # Apply sigmoid for realistic probabilities (avoids > 1 or < 0)
        # Keeps probabilities in reasonable range even with large leads
        win_prob = self._sigmoid(win_prob * 10 - 5)  # Scale and center

        # Clamp to valid range with small margins for certainty
        win_prob = max(0.01, min(0.99, win_prob))

        # Calculate confidence
        # Higher confidence when: late in game, large lead, or both
        lead_factor = min(abs(score_diff) / 20, 1.0)  # Max at 20-point lead
        time_certainty = 1 - time_factor  # More certain as game progresses
        confidence = 0.4 + (0.3 * lead_factor) + (0.3 * time_certainty)
        confidence = max(0.0, min(1.0, confidence))

        return win_prob, confidence

    def _sigmoid(self, x: float) -> float:
        """Sigmoid function for smooth probability curves."""
        return 1 / (1 + np.exp(-x))

    def predict_spread_cover(
        self,
        game_state: Dict,
        spread: float
    ) -> Dict[str, float]:
        """
        Predict probability of covering spread.

        Args:
            game_state: Current game state
            spread: Spread line (negative = home favored)

        Returns:
            Dict with home_cover_prob, away_cover_prob
        """
        # Get current win probability
        win_prob_result = self.predict(game_state)

        score_diff = game_state['home_score'] - game_state['away_score']
        time_remaining = win_prob_result['time_remaining_sec']

        # Adjusted differential vs spread
        adjusted_diff = score_diff - spread

        # If already covering by large margin late, very high probability
        if time_remaining < 120:  # Last 2 minutes
            if adjusted_diff > 5:
                home_cover_prob = 0.90
            elif adjusted_diff < -5:
                home_cover_prob = 0.10
            else:
                # Close game - use win probability as proxy
                home_cover_prob = win_prob_result['home_win_prob']
        else:
            # Earlier in game - less certain
            # Use win probability adjusted for spread differential
            home_cover_prob = win_prob_result['home_win_prob']

            # Adjust based on current position vs spread
            if adjusted_diff > 0:
                # Already covering
                home_cover_prob = min(0.95, home_cover_prob + 0.05)
            elif adjusted_diff < 0:
                # Not covering
                home_cover_prob = max(0.05, home_cover_prob - 0.05)

        return {
            'home_cover_prob': home_cover_prob,
            'away_cover_prob': 1 - home_cover_prob,
            'spread': spread,
            'adjusted_diff': adjusted_diff
        }

    def predict_total(
        self,
        game_state: Dict,
        total_line: float
    ) -> Dict[str, float]:
        """
        Predict probability of going over/under total.

        Args:
            game_state: Current game state
            total_line: Total line

        Returns:
            Dict with over_prob, under_prob
        """
        home_score = game_state['home_score']
        away_score = game_state['away_score']
        current_total = home_score + away_score

        time_remaining = self._get_time_remaining_seconds(game_state)
        quarter = game_state.get('quarter', 1)

        # Estimate pace (points per minute)
        total_game_time = 2880
        time_elapsed = total_game_time - time_remaining

        if time_elapsed > 0:
            current_pace = (current_total / time_elapsed) * 60  # Points per minute
        else:
            current_pace = 100 / 48  # ~2.08 pts/min (average game)

        # Project final score
        remaining_points = current_pace * (time_remaining / 60)
        projected_total = current_total + remaining_points

        # Calculate over probability
        diff_from_line = projected_total - total_line

        # Larger difference = higher confidence
        # ±10 points = very confident, ±2 points = uncertain
        normalized_diff = diff_from_line / 10

        # Use sigmoid for smooth probability
        over_prob = self._sigmoid(normalized_diff * 3)

        # Clamp
        over_prob = max(0.1, min(0.9, over_prob))

        return {
            'over_prob': over_prob,
            'under_prob': 1 - over_prob,
            'total_line': total_line,
            'current_total': current_total,
            'projected_total': projected_total,
            'current_pace': current_pace
        }


if __name__ == "__main__":
    # Test the model with various game scenarios
    model = LiveWinProbModel()

    print("=== Live Win Probability Model Tests ===\n")

    # Test Case 1: Close game, end of 1st quarter
    print("Test 1: Close game, late Q1")
    game1 = {
        'home_score': 28,
        'away_score': 26,
        'quarter': 1,
        'time_remaining': '1:30'
    }
    result1 = model.predict(game1)
    print(f"  Home: {game1['home_score']}, Away: {game1['away_score']}")
    print(f"  Q{game1['quarter']} - {game1['time_remaining']}")
    print(f"  Home Win Prob: {result1['home_win_prob']:.1%}")
    print(f"  Confidence: {result1['confidence']:.1%}\n")

    # Test Case 2: Home team up 10, middle of 3rd
    print("Test 2: Home up 10, mid Q3")
    game2 = {
        'home_score': 75,
        'away_score': 65,
        'quarter': 3,
        'time_remaining': '6:00'
    }
    result2 = model.predict(game2)
    print(f"  Home: {game2['home_score']}, Away: {game2['away_score']}")
    print(f"  Q{game2['quarter']} - {game2['time_remaining']}")
    print(f"  Home Win Prob: {result2['home_win_prob']:.1%}")
    print(f"  Confidence: {result2['confidence']:.1%}\n")

    # Test Case 3: Close game, final 2 minutes
    print("Test 3: Tied game, final 2 min")
    game3 = {
        'home_score': 105,
        'away_score': 105,
        'quarter': 4,
        'time_remaining': '2:00'
    }
    result3 = model.predict(game3)
    print(f"  Home: {game3['home_score']}, Away: {game3['away_score']}")
    print(f"  Q{game3['quarter']} - {game3['time_remaining']}")
    print(f"  Home Win Prob: {result3['home_win_prob']:.1%}")
    print(f"  Confidence: {result3['confidence']:.1%}\n")

    # Test Case 4: Blowout, late in game
    print("Test 4: Blowout, late Q4")
    game4 = {
        'home_score': 118,
        'away_score': 95,
        'quarter': 4,
        'time_remaining': '3:30'
    }
    result4 = model.predict(game4)
    print(f"  Home: {game4['home_score']}, Away: {game4['away_score']}")
    print(f"  Q{game4['quarter']} - {game4['time_remaining']}")
    print(f"  Home Win Prob: {result4['home_win_prob']:.1%}")
    print(f"  Confidence: {result4['confidence']:.1%}\n")

    # Test spread covering
    print("=== Spread Cover Probability ===\n")
    print("Test: Home up 8, spread -5.5, 5min left")
    game5 = {
        'home_score': 108,
        'away_score': 100,
        'quarter': 4,
        'time_remaining': '5:00'
    }
    spread_result = model.predict_spread_cover(game5, -5.5)
    print(f"  Current: Home +{game5['home_score'] - game5['away_score']}")
    print(f"  Spread: -5.5")
    print(f"  Adjusted diff: {spread_result['adjusted_diff']:+.1f}")
    print(f"  Home Cover Prob: {spread_result['home_cover_prob']:.1%}\n")

    # Test total
    print("=== Total (Over/Under) Probability ===\n")
    print("Test: 180 points scored, line 215.5, halftime")
    game6 = {
        'home_score': 92,
        'away_score': 88,
        'quarter': 2,
        'time_remaining': '0:00'
    }
    total_result = model.predict_total(game6, 215.5)
    print(f"  Current Total: {game6['home_score'] + game6['away_score']}")
    print(f"  Line: {total_result['total_line']}")
    print(f"  Projected: {total_result['projected_total']:.1f}")
    print(f"  Over Prob: {total_result['over_prob']:.1%}")
