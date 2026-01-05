"""
Correlation-Aware Position Sizer

Unified position sizing that accounts for:
1. Base Kelly calculation
2. Correlation adjustments
3. Drawdown scaling
4. Exposure limits

Integrates all risk management components.
"""

from typing import Optional
from loguru import logger

from .config import RiskConfig
from .models import BetCorrelationContext, PositionSizeResult
from .correlation_tracker import CorrelationTracker
from .drawdown_manager import DrawdownScaler
from .exposure_manager import ExposureManager


class CorrelationAwarePositionSizer:
    """
    Position sizing with full risk management stack.

    Applies adjustments in this order:
    1. Base Kelly calculation (from kelly_sizer)
    2. Correlation discount
    3. Drawdown scaling
    4. Exposure limit caps
    """

    def __init__(
        self,
        config: RiskConfig,
        kelly_sizer,
        correlation_tracker: CorrelationTracker,
        drawdown_scaler: DrawdownScaler,
        exposure_manager: ExposureManager,
    ):
        """
        Initialize position sizer.

        Args:
            config: Risk configuration
            kelly_sizer: KellyBetSizer instance for base Kelly calculation
            correlation_tracker: CorrelationTracker instance
            drawdown_scaler: DrawdownScaler instance
            exposure_manager: ExposureManager instance
        """
        self.config = config
        self.kelly_sizer = kelly_sizer
        self.correlation_tracker = correlation_tracker
        self.drawdown_scaler = drawdown_scaler
        self.exposure_manager = exposure_manager

        logger.info("CorrelationAwarePositionSizer initialized")

    def calculate_position_size(
        self,
        bankroll: float,
        win_prob: float,
        odds: float,
        context: BetCorrelationContext,
        odds_format: str = "american",
        drawdown: Optional[float] = None,
        **kwargs,
    ) -> PositionSizeResult:
        """
        Calculate final position size with all adjustments.

        Args:
            bankroll: Current bankroll
            win_prob: Model's win probability (0-1)
            odds: Betting odds
            context: Bet correlation context
            odds_format: Odds format ("american" or "decimal")
            drawdown: Optional explicit drawdown (for backtest)
            **kwargs: Additional arguments passed to kelly_sizer

        Returns:
            PositionSizeResult with final size and all adjustments
        """
        adjustments = []
        warnings = []

        # Step 1: Base Kelly calculation
        base_kelly = self.kelly_sizer.calculate_kelly(
            win_prob=win_prob,
            odds=odds,
            odds_format=odds_format,
            **kwargs
        )
        base_size = bankroll * base_kelly

        logger.debug(
            f"Base Kelly: {base_kelly:.4f} ({base_kelly*100:.2f}%) "
            f"→ ${base_size:.2f}"
        )

        # Step 2: Correlation adjustment
        corr_factor = self.correlation_tracker.get_correlation_factor(
            new_bet=context,
            bankroll=bankroll
        )

        if corr_factor < 1.0:
            adjustments.append(f"correlation_{corr_factor:.2f}x")
            logger.debug(f"Correlation factor: {corr_factor:.2f}x")

        # Step 3: Drawdown scaling
        dd_factor = self.drawdown_scaler.get_scale_factor(drawdown)

        if dd_factor < 1.0:
            adjustments.append(f"drawdown_{dd_factor:.2f}x")
            logger.debug(f"Drawdown factor: {dd_factor:.2f}x")

        if dd_factor == 0.0:
            # Hard stop
            warnings.append("Drawdown hard stop triggered")
            return PositionSizeResult(
                final_size=0.0,
                base_kelly=base_kelly,
                base_size=base_size,
                correlation_factor=corr_factor,
                drawdown_factor=dd_factor,
                adjustments=adjustments,
                warnings=warnings
            )

        # Apply correlation and drawdown adjustments
        adjusted_size = base_size * corr_factor * dd_factor

        # Step 4: Cap by exposure limits
        final_size, limiting_factors = self.exposure_manager.cap_to_limits(
            proposed_size=adjusted_size,
            bankroll=bankroll,
            context=context
        )

        if limiting_factors:
            adjustments.extend(limiting_factors)
            logger.debug(f"Exposure limits: {limiting_factors}")

        # Check correlation exposure limits (hard limits)
        is_allowed, violations = self.correlation_tracker.check_exposure_limits(
            new_bet=context,
            new_amount=final_size,
            bankroll=bankroll
        )

        if not is_allowed:
            warnings.extend(violations)
            logger.warning(f"Correlation exposure violations: {violations}")
            final_size = 0.0

        # Log final result
        logger.info(
            f"Position sizing: ${base_size:.2f} → ${final_size:.2f} | "
            f"Adjustments: {', '.join(adjustments) if adjustments else 'none'}"
        )

        return PositionSizeResult(
            final_size=final_size,
            base_kelly=base_kelly,
            base_size=base_size,
            correlation_factor=corr_factor,
            drawdown_factor=dd_factor,
            exposure_cap=final_size if limiting_factors else None,
            adjustments=adjustments,
            warnings=warnings
        )

    def evaluate_bet(
        self,
        bankroll: float,
        win_prob: float,
        odds: float,
        game_id: str,
        home_team: str,
        away_team: str,
        bet_side: str,
        bet_type: str = "spread",
        odds_format: str = "american",
        drawdown: Optional[float] = None,
        **kwargs,
    ) -> PositionSizeResult:
        """
        Convenience method to evaluate bet from game details.

        Args:
            bankroll: Current bankroll
            win_prob: Model's win probability
            odds: Betting odds
            game_id: Game identifier
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            bet_side: Bet side (HOME/AWAY/OVER/UNDER)
            bet_type: Bet type (moneyline/spread/totals)
            odds_format: Odds format
            drawdown: Optional explicit drawdown
            **kwargs: Additional arguments for kelly_sizer

        Returns:
            PositionSizeResult
        """
        # Build correlation context
        context = BetCorrelationContext.from_game(
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            bet_side=bet_side,
            bet_type=bet_type
        )

        return self.calculate_position_size(
            bankroll=bankroll,
            win_prob=win_prob,
            odds=odds,
            context=context,
            odds_format=odds_format,
            drawdown=drawdown,
            **kwargs
        )

    def __repr__(self) -> str:
        return (
            f"CorrelationAwarePositionSizer("
            f"correlation_tracker={self.correlation_tracker}, "
            f"drawdown_scaler={self.drawdown_scaler}"
            f")"
        )
