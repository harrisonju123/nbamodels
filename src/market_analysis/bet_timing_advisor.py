"""
Bet Timing Advisor

Provides real-time guidance on when to place bets based on:
- Historical optimal timing patterns
- Current line movement and steam detection
- Risk of waiting vs placing now
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from .timing_analysis import HistoricalTimingAnalyzer
from ..data.line_history import LineHistoryManager


@dataclass
class TimingRecommendation:
    """Real-time timing recommendation for a bet."""

    action: str  # 'place_now', 'wait', 'avoid'
    confidence: float  # 0-1
    reasons: List[str]
    suggested_wait_hours: Optional[float]
    expected_clv_now: float
    expected_clv_optimal: float
    risk_of_waiting: str  # 'low', 'medium', 'high'
    current_conditions: Dict


class BetTimingAdvisor:
    """
    Real-time timing guidance for pending bets.

    Based on empirical analysis of 27,982 historical odds records:
    - OPTIMAL WINDOW: 1-4 hours before game (0.67 pts avg movement)
    - AVOID: 48-72 hours (1.71 pts avg movement, 83% of lines move >0.5pts)
    - RISKY: 0-1 hour (closing line too efficient)

    See docs/BET_TIMING_ANALYSIS.md for full analysis.
    """

    # Empirically derived optimal windows (from historical analysis)
    OPTIMAL_WINDOW_MIN = 1.0  # 1 hour before game (start of optimal window)
    OPTIMAL_WINDOW_MAX = 4.0  # 4 hours before game (end of optimal window)

    # Risk thresholds
    HIGH_RISK_HOURS = 1  # Within 1 hour of game, too close to closing line
    EARLY_RISK_HOURS = 48  # More than 48 hours out, opening line volatility
    MEDIUM_RISK_HOURS = 12  # 12-48 hours, moderate volatility

    # CLV improvement thresholds
    MIN_CLV_IMPROVEMENT = 0.005  # 0.5% minimum improvement to wait

    # Expected line movement by window (from empirical analysis)
    EXPECTED_MOVEMENT = {
        'opening': 1.71,  # 48-72hr
        'early': 1.06,    # 24-48hr
        'mid': 0.97,      # 12-24hr
        'late': 1.04,     # 4-12hr
        'optimal': 0.67,  # 1-4hr (BEST)
        'closing': 0.50,  # 0-1hr (too late)
    }

    def __init__(
        self,
        db_path: str = "data/bets/bets.db",
        historical_analyzer: Optional[HistoricalTimingAnalyzer] = None,
    ):
        """
        Initialize bet timing advisor.

        Args:
            db_path: Path to bets database
            historical_analyzer: Historical timing analyzer (creates if None)
        """
        self.db_path = db_path
        self.historical = historical_analyzer or HistoricalTimingAnalyzer(db_path)
        self.line_manager = LineHistoryManager(db_path)

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_timing_recommendation(
        self,
        game_id: str,
        bet_type: str,
        bet_side: str,
        current_edge: float,
        commence_time: Optional[str] = None,
    ) -> TimingRecommendation:
        """
        Get timing recommendation for a specific bet.

        Args:
            game_id: Game ID
            bet_type: Bet type (spread, totals, moneyline)
            bet_side: Bet side
            current_edge: Current model edge
            commence_time: Game start time (ISO format)

        Returns:
            TimingRecommendation object
        """
        reasons = []
        suggested_wait_hours = None

        # Assess current conditions
        conditions = self._assess_current_conditions(game_id, bet_type, bet_side)

        # Calculate hours until game
        if commence_time:
            commence_dt = pd.to_datetime(commence_time)
            hours_to_game = (commence_dt - datetime.now()).total_seconds() / 3600
        else:
            hours_to_game = 24  # Default assumption

        # Calculate wait risk
        risk = self._calculate_wait_risk(game_id, hours_to_game)

        # Check if we're in optimal window (1-4 hours empirically derived)
        in_optimal_window = (self.OPTIMAL_WINDOW_MIN <= hours_to_game <= self.OPTIMAL_WINDOW_MAX)

        # Use empirical optimal window midpoint for calculations
        optimal_hours = (self.OPTIMAL_WINDOW_MIN + self.OPTIMAL_WINDOW_MAX) / 2  # 2.5 hours

        # Estimate expected CLV now vs optimal
        expected_clv_now = self._estimate_clv(current_edge, hours_to_game, bet_type)
        expected_clv_optimal = self._estimate_clv(current_edge, optimal_hours, bet_type)

        clv_improvement = expected_clv_optimal - expected_clv_now

        # Decision logic
        action = 'place_now'
        confidence = 0.5

        # TOO EARLY: Opening line volatility (48+ hours)
        if hours_to_game >= self.EARLY_RISK_HOURS:
            action = 'wait'
            confidence = 0.85
            suggested_wait_hours = hours_to_game - optimal_hours
            reasons.append(f"TOO EARLY: {hours_to_game:.1f}h out - opening lines highly volatile (avg 1.7pt movement)")
            reasons.append(f"Wait until 1-4hr window for best prices (currently in opening window)")

        # TOO LATE: Too close to closing line (<1 hour)
        elif hours_to_game <= self.HIGH_RISK_HOURS:
            action = 'place_now'
            confidence = 0.7  # Lower confidence - closing line already efficient
            reasons.append(f"Within {self.HIGH_RISK_HOURS}h of game - closing line, place now or skip")

        # STEAM DETECTED: Place now
        elif conditions.get('steam_detected', False):
            action = 'place_now'
            confidence = 0.85
            reasons.append("Steam move detected - line moving against us, place now")

        # ✅ OPTIMAL WINDOW: 1-4 hours (BEST)
        elif in_optimal_window:
            action = 'place_now'
            confidence = 0.9  # High confidence - empirically validated
            reasons.append(f"✅ OPTIMAL WINDOW: {hours_to_game:.1f}h before game (1-4hr window)")
            reasons.append(f"Empirically best timing: lowest volatility (0.67pt avg movement)")

        # SHARP ALIGNED: Consider waiting if far from game
        elif conditions.get('sharp_aligned', False) and hours_to_game > self.MEDIUM_RISK_HOURS:
            # Sharp money may improve line
            if clv_improvement > self.MIN_CLV_IMPROVEMENT:
                action = 'wait'
                confidence = 0.65
                suggested_wait_hours = optimal_hours - hours_to_game
                reasons.append(f"Sharp aligned, wait until ~{optimal_hours}h before game")
                reasons.append(f"Expected CLV improvement: {clv_improvement:.2%}")
            else:
                action = 'place_now'
                confidence = 0.7
                reasons.append("Sharp aligned but minimal CLV improvement expected")

        # LINE MOVING FAVORABLY: Wait
        elif conditions.get('line_velocity', 0) > 0.1 and hours_to_game > self.MEDIUM_RISK_HOURS:
            action = 'wait'
            confidence = 0.6
            suggested_wait_hours = min(4, hours_to_game - self.HIGH_RISK_HOURS)
            reasons.append("Line moving in our favor, wait for better price")

        # LINE MOVING AGAINST US: Place now
        elif conditions.get('line_velocity', 0) < -0.1:
            action = 'place_now'
            confidence = 0.75
            reasons.append("Line moving against us, place before it gets worse")

        # DEFAULT: Wait if time allows and improvement expected
        elif hours_to_game > optimal_hours and clv_improvement > self.MIN_CLV_IMPROVEMENT:
            action = 'wait'
            confidence = 0.6
            suggested_wait_hours = hours_to_game - optimal_hours
            reasons.append(f"Wait until optimal window (~{optimal_hours}h before game)")
            reasons.append(f"Expected CLV improvement: {clv_improvement:.2%}")

        # EDGE TOO SMALL: Avoid
        elif current_edge < 0.03:
            action = 'avoid'
            confidence = 0.7
            reasons.append("Edge too small, timing unlikely to help significantly")

        else:
            action = 'place_now'
            confidence = 0.6
            reasons.append("No strong timing signal, place now")

        return TimingRecommendation(
            action=action,
            confidence=confidence,
            reasons=reasons,
            suggested_wait_hours=suggested_wait_hours,
            expected_clv_now=expected_clv_now,
            expected_clv_optimal=expected_clv_optimal,
            risk_of_waiting=risk,
            current_conditions=conditions,
        )

    def _assess_current_conditions(
        self,
        game_id: str,
        bet_type: str,
        bet_side: str,
    ) -> Dict:
        """
        Assess current market conditions.

        Args:
            game_id: Game ID
            bet_type: Bet type
            bet_side: Bet side

        Returns:
            Dict with current conditions
        """
        conditions = {
            'steam_detected': False,
            'sharp_aligned': False,
            'rlm_detected': False,
            'line_velocity': 0.0,
            'recent_reversals': 0,
        }

        try:
            # Check for line movement pattern
            pattern = self.line_manager.analyze_movement_pattern(game_id, bet_type)

            if pattern:
                conditions['line_velocity'] = pattern.velocity

                if pattern.pattern == 'late_steam':
                    conditions['steam_detected'] = True

                conditions['recent_reversals'] = pattern.reversals

        except Exception as e:
            logger.debug(f"Could not analyze movement pattern: {e}")

        try:
            # Check for sharp divergence
            conn = self._get_connection()

            sharp_query = """
                SELECT sharp_side, divergence
                FROM sharp_divergences
                WHERE game_id = ? AND bet_type = ?
                ORDER BY snapshot_time DESC
                LIMIT 1
            """

            sharp_result = conn.execute(sharp_query, (game_id, bet_type)).fetchone()

            if sharp_result and sharp_result['sharp_side'] == bet_side:
                conditions['sharp_aligned'] = True
                conditions['sharp_divergence'] = sharp_result['divergence']

            conn.close()

        except Exception as e:
            logger.debug(f"Could not check sharp divergence: {e}")

        return conditions

    def _predict_line_movement(
        self,
        game_id: str,
        bet_type: str,
    ) -> Dict:
        """
        Predict likely line movement based on historical patterns.

        Args:
            game_id: Game ID
            bet_type: Bet type

        Returns:
            Dict with movement prediction
        """
        # Simplified implementation
        # Full version would use line movement patterns and ML

        try:
            history = self.line_manager.get_line_history(
                game_id=game_id,
                bet_type=bet_type,
            )

            if len(history) >= 2:
                recent_movement = history.iloc[-1]['line'] - history.iloc[0]['line']

                return {
                    'predicted_direction': 'up' if recent_movement > 0 else 'down',
                    'confidence': 0.5,
                }

        except Exception as e:
            logger.debug(f"Could not predict line movement: {e}")

        return {'predicted_direction': 'stable', 'confidence': 0.3}

    def _calculate_wait_risk(
        self,
        game_id: str,
        hours_to_game: float,
    ) -> str:
        """
        Calculate risk of waiting to place bet.

        Args:
            game_id: Game ID
            hours_to_game: Hours until game starts

        Returns:
            Risk level: 'low', 'medium', 'high'
        """
        if hours_to_game <= self.HIGH_RISK_HOURS:
            return 'high'
        elif hours_to_game <= self.MEDIUM_RISK_HOURS:
            return 'medium'
        else:
            return 'low'

    def _estimate_clv(
        self,
        current_edge: float,
        hours_before: float,
        bet_type: str,
    ) -> float:
        """
        Estimate expected CLV at a given time before game.

        Args:
            current_edge: Current model edge
            hours_before: Hours before game
            bet_type: Bet type

        Returns:
            Estimated CLV
        """
        # Simplified heuristic
        # CLV tends to be better 12-24 hours before game
        # and worse very close to game

        if hours_before < 1:
            clv_factor = 0.3  # 30% of edge
        elif hours_before < 4:
            clv_factor = 0.5
        elif hours_before < 12:
            clv_factor = 0.7
        elif hours_before < 24:
            clv_factor = 0.9  # Best window
        else:
            clv_factor = 0.6  # Too early, less liquidity

        estimated_clv = current_edge * clv_factor * 0.5  # Assume ~50% of edge becomes CLV

        return estimated_clv

    def batch_recommendations(
        self,
        pending_signals: List[Dict],
    ) -> List[TimingRecommendation]:
        """
        Get timing recommendations for multiple pending bets.

        Args:
            pending_signals: List of bet signal dicts with game_id, bet_type, bet_side, edge

        Returns:
            List of TimingRecommendation objects
        """
        recommendations = []

        for signal in pending_signals:
            try:
                rec = self.get_timing_recommendation(
                    game_id=signal['game_id'],
                    bet_type=signal.get('bet_type', 'spread'),
                    bet_side=signal.get('bet_side', 'home'),
                    current_edge=signal.get('edge', 0.0),
                    commence_time=signal.get('commence_time'),
                )
                recommendations.append(rec)

            except Exception as e:
                logger.error(f"Failed to get recommendation for {signal.get('game_id')}: {e}")

        return recommendations
