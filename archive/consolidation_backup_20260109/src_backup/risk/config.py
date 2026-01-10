"""
Risk Management Configuration

Centralized configuration for advanced risk management features:
- Correlation limits and discounts
- Drawdown scaling thresholds
- Exposure limits (daily/weekly/pending)
"""

from dataclasses import dataclass


@dataclass
class RiskConfig:
    """
    Centralized risk management configuration.

    All exposure limits are expressed as fractions of current bankroll (0-1).
    All discounts are multipliers (0-1) applied to position sizes.
    """

    # === Correlation Limits ===
    max_same_team_exposure: float = 0.10
    """Maximum exposure on/against the same team (10% of bankroll)"""

    max_same_game_exposure: float = 0.05
    """Maximum exposure on a single game (5% of bankroll)"""

    max_same_conference_exposure: float = 0.15
    """Maximum exposure within the same conference (15% of bankroll)"""

    max_same_division_exposure: float = 0.08
    """Maximum exposure within the same division (8% of bankroll)"""

    # === Correlation Discounts ===
    same_team_discount: float = 0.7
    """Multiply bet size by 0.7 (30% reduction) when betting on/against same team"""

    same_game_discount: float = 0.5
    """Multiply bet size by 0.5 (50% reduction) for multiple bets on same game"""

    same_conference_discount: float = 0.9
    """Multiply bet size by 0.9 (10% reduction) when heavily exposed to conference"""

    same_division_discount: float = 0.85
    """Multiply bet size by 0.85 (15% reduction) when heavily exposed to division"""

    # === Drawdown Thresholds ===
    drawdown_scale_start: float = 0.10
    """Start reducing bet sizes at 10% drawdown"""

    drawdown_scale_50pct: float = 0.15
    """Scale bets to 50% size at 15% drawdown"""

    drawdown_scale_25pct: float = 0.25
    """Scale bets to 25% size at 25% drawdown"""

    drawdown_hard_stop: float = 0.30
    """Hard stop: pause all betting at 30% drawdown"""

    # === Exposure Limits ===
    max_daily_exposure: float = 0.15
    """Maximum total wagered per day (15% of bankroll)"""

    max_weekly_exposure: float = 0.40
    """Maximum total wagered per week (40% of bankroll)"""

    max_pending_exposure: float = 0.25
    """Maximum total in unsettled bets (25% of bankroll)"""

    # === Attribution Settings ===
    attribution_teams: bool = True
    """Track P&L attribution by team"""

    attribution_situations: bool = True
    """Track P&L attribution by situation (B2B, rest, home/away, edge)"""

    attribution_time_periods: bool = True
    """Track P&L attribution by time period (daily/weekly/monthly)"""

    attribution_edge_buckets: bool = True
    """Track P&L attribution by edge bucket"""

    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate exposure limits are in valid range
        for attr in ['max_same_team_exposure', 'max_same_game_exposure',
                     'max_same_conference_exposure', 'max_same_division_exposure',
                     'max_daily_exposure', 'max_weekly_exposure', 'max_pending_exposure']:
            value = getattr(self, attr)
            if not 0 < value <= 1:
                raise ValueError(f"{attr} must be between 0 and 1, got {value}")

        # Validate discounts are in valid range
        for attr in ['same_team_discount', 'same_game_discount',
                     'same_conference_discount', 'same_division_discount']:
            value = getattr(self, attr)
            if not 0 < value <= 1:
                raise ValueError(f"{attr} must be between 0 and 1, got {value}")

        # Validate drawdown thresholds are increasing
        if not (0 < self.drawdown_scale_start < self.drawdown_scale_50pct <
                self.drawdown_scale_25pct < self.drawdown_hard_stop <= 1):
            raise ValueError(
                "Drawdown thresholds must be increasing: "
                f"start({self.drawdown_scale_start}) < 50%({self.drawdown_scale_50pct}) < "
                f"25%({self.drawdown_scale_25pct}) < stop({self.drawdown_hard_stop})"
            )
