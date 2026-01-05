"""
Position limit and exposure management.

Manages betting constraints for realistic simulation:
1. Max bet per book (sportsbook limits)
2. Max daily exposure (% of bankroll)
3. Max per-game exposure (no correlation risk)
4. Bankroll floor (stop-loss)
"""

from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np

from .core import BacktestConfig, RigorousBet


@dataclass
class BetConstraints:
    """Constraints for a specific bet."""

    original_size: float
    adjusted_size: float
    constraint_violations: List[str] = field(default_factory=list)
    is_allowed: bool = True


@dataclass
class PortfolioState:
    """Current portfolio state for constraint checking."""

    current_bankroll: float
    daily_wagered: float = 0.0
    pending_exposure: Dict[str, float] = field(default_factory=dict)  # game_id -> exposure
    book_limits: Dict[str, float] = field(default_factory=dict)  # book_name -> remaining


class ConstraintManager:
    """
    Manages betting constraints for realistic simulation.

    Enforces:
    1. Bankroll floor (keep minimum reserve)
    2. Per-book limits
    3. Daily exposure limit
    4. Per-game exposure limit
    5. Max bet fraction
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.state = PortfolioState(current_bankroll=config.initial_bankroll)
        self.initial_bankroll = config.initial_bankroll

    def check_and_adjust_bet(
        self, bet: RigorousBet, current_bankroll: float
    ) -> BetConstraints:
        """
        Check all constraints and adjust bet size if needed.

        Order of constraints (most restrictive first):
        1. Bankroll floor
        2. Per-book limits
        3. Daily exposure limit
        4. Per-game exposure limit
        5. Max bet fraction

        Args:
            bet: Proposed bet
            current_bankroll: Current bankroll

        Returns:
            BetConstraints with adjusted size
        """
        violations = []
        adjusted_size = bet.bet_size

        # Update state
        self.state.current_bankroll = current_bankroll

        # 1. Bankroll floor (keep 10% minimum)
        floor = self.initial_bankroll * 0.1
        max_from_floor = max(0, current_bankroll - floor)
        if adjusted_size > max_from_floor:
            adjusted_size = max_from_floor
            violations.append("bankroll_floor")

        # 2. Per-book limits
        book_limit = self.config.max_bet_per_book
        if adjusted_size > book_limit:
            adjusted_size = book_limit
            violations.append("book_limit")

        # 3. Daily exposure limit
        daily_limit = current_bankroll * self.config.max_daily_exposure
        remaining_daily = max(0, daily_limit - self.state.daily_wagered)
        if adjusted_size > remaining_daily:
            adjusted_size = remaining_daily
            violations.append("daily_exposure")

        # 4. Per-game exposure
        game_limit = current_bankroll * self.config.max_per_game_exposure
        existing_exposure = self.state.pending_exposure.get(bet.game_id, 0)
        remaining_game = max(0, game_limit - existing_exposure)
        if adjusted_size > remaining_game:
            adjusted_size = remaining_game
            violations.append("game_exposure")

        # 5. Max bet fraction
        max_bet = current_bankroll * self.config.max_bet_fraction
        if adjusted_size > max_bet:
            adjusted_size = max_bet
            violations.append("max_bet_fraction")

        # Minimum bet size
        is_allowed = adjusted_size >= 1.0  # $1 minimum

        return BetConstraints(
            original_size=bet.bet_size,
            adjusted_size=max(0, adjusted_size),
            constraint_violations=violations,
            is_allowed=is_allowed,
        )

    def record_bet(self, bet: RigorousBet):
        """Record a placed bet for constraint tracking."""
        self.state.daily_wagered += bet.bet_size

        if bet.game_id in self.state.pending_exposure:
            self.state.pending_exposure[bet.game_id] += bet.bet_size
        else:
            self.state.pending_exposure[bet.game_id] = bet.bet_size

    def settle_game(self, game_id: str):
        """Clear exposure for a settled game."""
        if game_id in self.state.pending_exposure:
            del self.state.pending_exposure[game_id]

    def reset_daily(self):
        """Reset daily constraints (called at start of each day)."""
        self.state.daily_wagered = 0.0

    def get_current_exposure(self) -> float:
        """Get total current exposure."""
        return sum(self.state.pending_exposure.values())

    def get_exposure_by_game(self) -> Dict[str, float]:
        """Get exposure breakdown by game."""
        return self.state.pending_exposure.copy()
