"""
Risk-Aware Backtest Integration

Integrates the advanced risk management system with rigorous backtesting.
Extends existing ConstraintManager with correlation tracking and drawdown scaling.
"""

from typing import List, Dict, Optional
from datetime import datetime
from loguru import logger

from .core import BacktestConfig, RigorousBet
from .constraints import ConstraintManager, BetConstraints
from src.risk import (
    RiskConfig,
    CorrelationTracker,
    DrawdownScaler,
    ExposureManager,
    CorrelationAwarePositionSizer,
    BetCorrelationContext,
)
from src.betting.kelly import KellyBetSizer


class RiskAwareBacktester:
    """
    Enhanced backtest engine with full risk management.

    Wraps existing ConstraintManager and adds:
    - Correlation-aware position sizing
    - Drawdown-based scaling
    - Enhanced exposure tracking
    """

    def __init__(
        self,
        backtest_config: BacktestConfig,
        risk_config: Optional[RiskConfig] = None,
    ):
        """
        Initialize risk-aware backtester.

        Args:
            backtest_config: Standard backtest configuration
            risk_config: Optional risk configuration (uses defaults if None)
        """
        self.backtest_config = backtest_config
        self.risk_config = risk_config or RiskConfig()

        # Initialize base constraint manager
        self.constraint_manager = ConstraintManager(backtest_config)

        # Initialize risk components
        self.kelly_sizer = KellyBetSizer(
            fraction=backtest_config.kelly_fraction,
            max_bet_pct=backtest_config.max_bet_fraction
        )
        self.correlation_tracker = CorrelationTracker(self.risk_config)
        self.drawdown_scaler = DrawdownScaler(self.risk_config)

        # Note: ExposureManager queries database - use in-memory tracking for backtest
        self._daily_wagered = 0.0
        self._current_date = None

        # Track peak and drawdown
        self._peak_bankroll = backtest_config.initial_bankroll
        self._current_drawdown = 0.0

        logger.info(
            f"RiskAwareBacktester initialized | "
            f"Correlation tracking: ON | "
            f"Drawdown scaling: ON"
        )

    def evaluate_bet(
        self,
        bet: RigorousBet,
        current_bankroll: float,
        game_date: datetime
    ) -> BetConstraints:
        """
        Evaluate bet through full risk management stack.

        Args:
            bet: Proposed bet
            current_bankroll: Current bankroll
            game_date: Date of game

        Returns:
            BetConstraints with adjusted size and violations
        """
        # Reset daily tracking if new day
        if self._current_date != game_date.date():
            self._current_date = game_date.date()
            self._daily_wagered = 0.0

        # Step 1: Update peak and calculate drawdown
        if current_bankroll > self._peak_bankroll:
            self._peak_bankroll = current_bankroll
        self._current_drawdown = (
            (self._peak_bankroll - current_bankroll) / self._peak_bankroll
            if self._peak_bankroll > 0 else 0.0
        )

        # Step 2: Check drawdown hard stop
        should_pause, pause_reason = self.drawdown_scaler.should_pause_betting(
            self._current_drawdown
        )
        if should_pause:
            logger.warning(f"Drawdown pause: {pause_reason}")
            return BetConstraints(
                original_size=bet.bet_size,
                adjusted_size=0.0,
                constraint_violations=[f"drawdown_pause: {pause_reason}"],
                is_allowed=False
            )

        # Step 3: Get drawdown scale factor
        dd_factor = self.drawdown_scaler.get_scale_factor(self._current_drawdown)

        # Step 4: Build correlation context
        context = BetCorrelationContext.from_game(
            game_id=bet.game_id,
            home_team=bet.game_id.split('_')[1] if '_' in bet.game_id else "UNK",  # Parse from game_id
            away_team=bet.game_id.split('_')[2] if '_' in bet.game_id else "UNK",
            bet_side=bet.bet_side,
            bet_type=bet.bet_type
        )

        # Step 5: Get correlation factor
        corr_factor = self.correlation_tracker.get_correlation_factor(
            new_bet=context,
            bankroll=current_bankroll
        )

        # Step 6: Apply risk adjustments to bet size
        violations = []
        adjusted_size = bet.bet_size * corr_factor * dd_factor

        if corr_factor < 1.0:
            violations.append(f"correlation_{corr_factor:.2f}x")
        if dd_factor < 1.0:
            violations.append(f"drawdown_{dd_factor:.2f}x")

        # Step 7: Check daily exposure limit
        daily_limit = current_bankroll * self.risk_config.max_daily_exposure
        remaining_daily = max(0, daily_limit - self._daily_wagered)
        if adjusted_size > remaining_daily:
            adjusted_size = remaining_daily
            violations.append(f"daily_limit")

        # Step 8: Apply base constraints (from original ConstraintManager)
        temp_bet = RigorousBet(
            date=bet.date,
            game_id=bet.game_id,
            bet_side=bet.bet_side,
            bet_type=bet.bet_type,
            bet_size=adjusted_size,  # Use risk-adjusted size
            odds=bet.odds,
            line=bet.line,
            model_prob=bet.model_prob,
            market_prob=bet.market_prob,
            edge=bet.edge
        )
        base_constraints = self.constraint_manager.check_and_adjust_bet(
            temp_bet,
            current_bankroll
        )

        # Merge violations
        all_violations = violations + base_constraints.constraint_violations
        final_size = base_constraints.adjusted_size
        is_allowed = base_constraints.is_allowed

        # Step 9: Check correlation exposure limits (hard limits)
        if is_allowed and final_size > 0:
            is_allowed, corr_violations = self.correlation_tracker.check_exposure_limits(
                new_bet=context,
                new_amount=final_size,
                bankroll=current_bankroll
            )
            if not is_allowed:
                all_violations.extend(corr_violations)
                final_size = 0.0

        return BetConstraints(
            original_size=bet.bet_size,
            adjusted_size=final_size,
            constraint_violations=all_violations,
            is_allowed=is_allowed and final_size >= 1.0
        )

    def record_bet(self, bet: RigorousBet, adjusted_size: float):
        """
        Record a placed bet for tracking.

        Args:
            bet: Bet details
            adjusted_size: Final bet size after adjustments
        """
        # Update daily wagered
        self._daily_wagered += adjusted_size

        # Record in base constraint manager
        bet_copy = RigorousBet(
            date=bet.date,
            game_id=bet.game_id,
            bet_side=bet.bet_side,
            bet_type=bet.bet_type,
            bet_size=adjusted_size,
            odds=bet.odds,
            line=bet.line,
            model_prob=bet.model_prob,
            market_prob=bet.market_prob,
            edge=bet.edge
        )
        self.constraint_manager.record_bet(bet_copy)

        # Add to correlation tracker
        context = BetCorrelationContext.from_game(
            game_id=bet.game_id,
            home_team=bet.game_id.split('_')[1] if '_' in bet.game_id else "UNK",
            away_team=bet.game_id.split('_')[2] if '_' in bet.game_id else "UNK",
            bet_side=bet.bet_side,
            bet_type=bet.bet_type
        )
        self.correlation_tracker.add_pending_bet(
            bet_id=f"{bet.game_id}_{bet.bet_side}",
            context=context,
            amount=adjusted_size
        )

    def settle_game(self, game_id: str):
        """
        Settle a game (clear exposure).

        Args:
            game_id: Game identifier
        """
        self.constraint_manager.settle_game(game_id)

        # Settle all bets for this game in correlation tracker
        # Note: In backtest, we settle entire games at once
        for bet_id in list(self.correlation_tracker._pending_bets.keys()):
            if bet_id.startswith(game_id):
                self.correlation_tracker.settle_bet(bet_id)

    def reset_daily(self):
        """Reset daily tracking."""
        self._daily_wagered = 0.0
        self.constraint_manager.reset_daily()

    def get_risk_summary(self, current_bankroll: float) -> Dict:
        """
        Get current risk status summary.

        Args:
            current_bankroll: Current bankroll

        Returns:
            Dict with risk metrics
        """
        snapshot = self.correlation_tracker.get_exposure_snapshot()
        dd_report = self.drawdown_scaler.get_drawdown_report(self._current_drawdown)

        return {
            'current_bankroll': current_bankroll,
            'peak_bankroll': self._peak_bankroll,
            'drawdown': self._current_drawdown,
            'drawdown_scale_factor': dd_report['scale_factor'],
            'pending_bets': self.correlation_tracker.get_pending_count(),
            'pending_exposure': snapshot.total_pending,
            'daily_wagered': self._daily_wagered,
            'status': dd_report['status']
        }


def compare_backtest_modes(
    bets: List[RigorousBet],
    backtest_config: BacktestConfig,
    risk_config: Optional[RiskConfig] = None,
    mode: str = "both"
) -> Dict:
    """
    Compare backtest performance with and without risk management.

    Args:
        bets: List of proposed bets (chronologically sorted)
        backtest_config: Backtest configuration
        risk_config: Optional risk configuration
        mode: "baseline", "risk", or "both"

    Returns:
        Dict with comparison results
    """
    results = {}

    if mode in ["baseline", "both"]:
        logger.info("Running baseline backtest (no risk management)...")
        baseline_manager = ConstraintManager(backtest_config)
        results['baseline'] = _run_backtest(bets, baseline_manager, backtest_config)

    if mode in ["risk", "both"]:
        logger.info("Running risk-aware backtest...")
        risk_manager = RiskAwareBacktester(backtest_config, risk_config)
        results['risk'] = _run_backtest_risk(bets, risk_manager, backtest_config)

    return results


def _run_backtest(bets, manager, config):
    """Helper to run baseline backtest."""
    bankroll = config.initial_bankroll
    placed_bets = []

    for bet in bets:
        constraints = manager.check_and_adjust_bet(bet, bankroll)
        if constraints.is_allowed:
            manager.record_bet(bet)
            placed_bets.append(bet)
            # Simulate outcome (would need actual outcome data)

    return {
        'total_bets': len(placed_bets),
        'final_bankroll': bankroll
    }


def _run_backtest_risk(bets, manager, config):
    """Helper to run risk-aware backtest."""
    bankroll = config.initial_bankroll
    placed_bets = []

    for bet in bets:
        constraints = manager.evaluate_bet(bet, bankroll, bet.date)
        if constraints.is_allowed:
            manager.record_bet(bet, constraints.adjusted_size)
            placed_bets.append((bet, constraints))

    return {
        'total_bets': len(placed_bets),
        'final_bankroll': bankroll,
        'risk_summary': manager.get_risk_summary(bankroll)
    }
