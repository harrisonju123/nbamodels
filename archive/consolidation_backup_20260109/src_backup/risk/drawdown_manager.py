"""
Drawdown Manager

Scales bet sizes based on current drawdown level using piecewise linear
interpolation. Implements both gradual reduction and hard circuit breakers.

Scaling curve:
- DD < 10%: 100% sizing
- DD 10-15%: Linear reduction from 100% to 50%
- DD 15-25%: Linear reduction from 50% to 25%
- DD >= 25%: 25% sizing
- DD >= 30%: Hard stop (0% sizing)
"""

from typing import Tuple, Dict, Optional
from loguru import logger

from .config import RiskConfig


class DrawdownScaler:
    """
    Scales bet sizes based on current drawdown level.

    Integrates with BankrollManager to get current drawdown percentage.
    """

    def __init__(self, config: RiskConfig, bankroll_manager=None):
        """
        Initialize drawdown scaler.

        Args:
            config: Risk configuration
            bankroll_manager: Optional BankrollManager instance for live integration
        """
        self.config = config
        self.bankroll_manager = bankroll_manager

        logger.debug(
            f"DrawdownScaler initialized with thresholds: "
            f"start={config.drawdown_scale_start:.1%}, "
            f"50%={config.drawdown_scale_50pct:.1%}, "
            f"25%={config.drawdown_scale_25pct:.1%}, "
            f"stop={config.drawdown_hard_stop:.1%}"
        )

    def get_current_drawdown(self) -> float:
        """
        Get current drawdown from bankroll manager.

        Returns:
            Drawdown as decimal (0.15 = 15%)
        """
        if self.bankroll_manager:
            stats = self.bankroll_manager.get_bankroll_stats()
            return stats.get('max_drawdown', 0.0)
        else:
            # For backtest mode, drawdown will be passed explicitly
            logger.debug("No bankroll_manager - drawdown must be passed explicitly")
            return 0.0

    def get_scale_factor(self, drawdown: Optional[float] = None) -> float:
        """
        Calculate bet size scaling factor based on drawdown.

        Uses piecewise linear interpolation for smooth scaling:
        - DD < start: 1.0 (100%)
        - DD start to 50pct: Linear 1.0 → 0.5
        - DD 50pct to 25pct: Linear 0.5 → 0.25
        - DD >= 25pct: 0.25 (25%)
        - DD >= hard_stop: 0.0 (circuit breaker)

        Args:
            drawdown: Optional explicit drawdown value (for backtest).
                     If None, will query bankroll_manager.

        Returns:
            Factor between 0.0 and 1.0 to multiply bet sizes by
        """
        if drawdown is None:
            drawdown = self.get_current_drawdown()

        # Hard stop circuit breaker
        if drawdown >= self.config.drawdown_hard_stop:
            logger.warning(
                f"HARD STOP: Drawdown {drawdown:.1%} >= "
                f"{self.config.drawdown_hard_stop:.1%}"
            )
            return 0.0

        # Minimum sizing (25%)
        if drawdown >= self.config.drawdown_scale_25pct:
            logger.info(
                f"Minimum sizing: Drawdown {drawdown:.1%} >= "
                f"{self.config.drawdown_scale_25pct:.1%}"
            )
            return 0.25

        # Linear interpolation: 50% → 25%
        if drawdown >= self.config.drawdown_scale_50pct:
            progress = (
                (drawdown - self.config.drawdown_scale_50pct) /
                (self.config.drawdown_scale_25pct - self.config.drawdown_scale_50pct)
            )
            factor = 0.5 - (0.25 * progress)
            logger.info(
                f"Scaling: Drawdown {drawdown:.1%} → {factor:.1%} sizing"
            )
            return factor

        # Linear interpolation: 100% → 50%
        if drawdown >= self.config.drawdown_scale_start:
            progress = (
                (drawdown - self.config.drawdown_scale_start) /
                (self.config.drawdown_scale_50pct - self.config.drawdown_scale_start)
            )
            factor = 1.0 - (0.5 * progress)
            logger.info(
                f"Scaling: Drawdown {drawdown:.1%} → {factor:.1%} sizing"
            )
            return factor

        # Normal sizing (100%)
        return 1.0

    def should_pause_betting(self, drawdown: Optional[float] = None) -> Tuple[bool, str]:
        """
        Check if drawdown exceeds hard stop threshold.

        Args:
            drawdown: Optional explicit drawdown value

        Returns:
            (should_pause, reason)
        """
        if drawdown is None:
            drawdown = self.get_current_drawdown()

        if drawdown >= self.config.drawdown_hard_stop:
            reason = (
                f"Drawdown {drawdown:.1%} exceeded hard stop "
                f"threshold {self.config.drawdown_hard_stop:.1%}"
            )
            logger.error(f"BETTING PAUSED: {reason}")
            return True, reason

        return False, ""

    def get_drawdown_report(self, drawdown: Optional[float] = None) -> Dict:
        """
        Get detailed drawdown analysis.

        Args:
            drawdown: Optional explicit drawdown value

        Returns:
            Dict with drawdown metrics and status
        """
        if drawdown is None:
            drawdown = self.get_current_drawdown()

        scale_factor = self.get_scale_factor(drawdown)
        should_pause, pause_reason = self.should_pause_betting(drawdown)

        # Determine status
        if should_pause:
            status = "HARD_STOP"
        elif drawdown >= self.config.drawdown_scale_25pct:
            status = "MINIMUM_SIZING"
        elif drawdown >= self.config.drawdown_scale_50pct:
            status = "REDUCED_SIZING_50"
        elif drawdown >= self.config.drawdown_scale_start:
            status = "REDUCED_SIZING"
        else:
            status = "NORMAL"

        return {
            'drawdown': drawdown,
            'drawdown_pct': drawdown * 100,
            'scale_factor': scale_factor,
            'scale_factor_pct': scale_factor * 100,
            'status': status,
            'should_pause': should_pause,
            'pause_reason': pause_reason,
            'thresholds': {
                'start': self.config.drawdown_scale_start,
                '50pct': self.config.drawdown_scale_50pct,
                '25pct': self.config.drawdown_scale_25pct,
                'hard_stop': self.config.drawdown_hard_stop,
            }
        }

    def __repr__(self) -> str:
        drawdown = self.get_current_drawdown()
        scale_factor = self.get_scale_factor(drawdown)
        return (
            f"DrawdownScaler("
            f"current_dd={drawdown:.1%}, "
            f"scale_factor={scale_factor:.1%}"
            f")"
        )
