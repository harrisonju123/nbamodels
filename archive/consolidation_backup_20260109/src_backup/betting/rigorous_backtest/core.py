"""
Core classes for rigorous backtesting with statistical rigor.

This module provides:
- BacktestConfig: Configuration dataclass for backtest parameters
- RigorousBacktestResult: Result dataclass with uncertainty quantification
- RigorousBet: Enhanced bet record with transaction costs
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import numpy as np


@dataclass
class BacktestConfig:
    """Configuration for rigorous backtesting."""

    # Walk-forward settings
    initial_train_size: int = 500  # Minimum games needed to start testing
    retrain_frequency: str = "monthly"  # "monthly", "quarterly", or "N_games"
    use_expanding_window: bool = True  # True=expanding, False=rolling

    # Bet sizing
    kelly_fraction: float = 0.2  # Fraction of Kelly to bet (20% Kelly)
    max_bet_fraction: float = 0.05  # Max 5% of bankroll per bet
    min_edge_threshold: float = 0.07  # Minimum 7% edge to bet

    # Monte Carlo settings
    n_simulations: int = 1000  # Number of Monte Carlo runs
    bootstrap_samples: int = 5000  # Bootstrap samples for CI
    random_seed: int = 42

    # Initial bankroll
    initial_bankroll: float = 10000.0  # Starting bankroll

    # Transaction costs
    base_vig: float = 0.045  # 4.5% vig (standard -110 odds)
    slippage_model: str = "sqrt"  # "fixed", "linear", "sqrt"
    execution_probability: float = 0.95  # 95% of bets get executed

    # Constraints
    max_bet_per_book: float = 500.0  # Max bet at single book
    max_daily_exposure: float = 0.15  # Max 15% of bankroll per day
    max_per_game_exposure: float = 0.05  # Max 5% per game

    # Statistical settings
    confidence_level: float = 0.95  # 95% confidence intervals
    multiple_testing_method: str = "fdr_bh"  # "bonferroni", "holm", "fdr_bh"


@dataclass
class RigorousBet:
    """Enhanced bet record with full details."""

    # Basic bet info
    date: datetime
    game_id: str
    bet_side: str  # "HOME" or "AWAY"
    bet_type: str  # "SPREAD", "TOTAL", "MONEYLINE"
    bet_size: float
    odds: float  # Decimal odds
    line: float  # Spread or total line

    # Model predictions
    model_prob: float  # Model's win probability
    market_prob: float  # Implied probability from odds
    edge: float  # Expected edge

    # Outcome
    won: Optional[bool] = None
    actual_score_home: Optional[int] = None
    actual_score_away: Optional[int] = None
    pnl: float = 0.0  # Profit/loss

    # Bankroll tracking
    bankroll_before: float = 0.0
    bankroll_after: float = 0.0

    # Transaction costs
    slippage: float = 0.0  # Points of slippage
    vig_cost: float = 0.0  # Vig paid
    execution_prob: float = 1.0  # Probability bet executed
    execution_failed: bool = False

    # CLV tracking
    closing_odds: Optional[float] = None
    clv: Optional[float] = None


@dataclass
class FoldResult:
    """Results from a single walk-forward fold."""

    fold_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    # Performance
    roi: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float

    # Confidence intervals
    roi_ci: Tuple[float, float]
    win_rate_ci: Tuple[float, float]

    # Sample info
    num_bets: int
    total_wagered: float
    total_pnl: float

    # Bets in this fold
    bets: List[RigorousBet] = field(default_factory=list)


@dataclass
class RigorousBacktestResult:
    """Complete backtest results with uncertainty quantification."""

    # Configuration used
    config: BacktestConfig

    # Point estimates
    roi: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    num_bets: int
    total_wagered: float
    total_pnl: float

    # Confidence intervals (95% by default)
    roi_ci: Tuple[float, float]
    win_rate_ci: Tuple[float, float]
    sharpe_ci: Tuple[float, float]
    max_drawdown_ci: Tuple[float, float]

    # Monte Carlo distributions
    roi_distribution: np.ndarray
    sharpe_distribution: np.ndarray
    drawdown_distribution: np.ndarray

    # Percentiles (5th, 25th, 50th, 75th, 95th)
    roi_percentiles: Dict[int, float] = field(default_factory=dict)
    sharpe_percentiles: Dict[int, float] = field(default_factory=dict)
    drawdown_percentiles: Dict[int, float] = field(default_factory=dict)

    # Statistical significance
    p_value_vs_breakeven: float = 1.0
    p_value_vs_market: float = 1.0

    # Risk metrics
    risk_of_ruin: float = 0.0  # P(bankroll < 50% of initial)
    probability_profitable: float = 0.0
    expected_kelly_growth: float = 0.0

    # Per-fold results
    fold_results: List[FoldResult] = field(default_factory=list)

    # Transaction cost impact
    gross_roi: float = 0.0
    net_roi: float = 0.0
    total_slippage: float = 0.0
    total_vig_paid: float = 0.0
    bets_not_executed: int = 0

    # All bets
    bets: List[RigorousBet] = field(default_factory=list)
    bankroll_history: np.ndarray = field(default_factory=lambda: np.array([]))

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    test_seasons: List[int] = field(default_factory=list)

    def summary(self) -> str:
        """Return formatted summary string."""
        return f"""
=== Rigorous Backtest Results ===

ROI: {self.roi:.2%} (95% CI: {self.roi_ci[0]:.2%}, {self.roi_ci[1]:.2%})
Win Rate: {self.win_rate:.1%} (95% CI: {self.win_rate_ci[0]:.1%}, {self.win_rate_ci[1]:.1%})
Sharpe: {self.sharpe_ratio:.2f} (95% CI: {self.sharpe_ci[0]:.2f}, {self.sharpe_ci[1]:.2f})
Max Drawdown: {self.max_drawdown:.1%} (95% CI: {self.max_drawdown_ci[0]:.1%}, {self.max_drawdown_ci[1]:.1%})

Number of Bets: {self.num_bets}
Total Wagered: ${self.total_wagered:,.2f}
Total P&L: ${self.total_pnl:,.2f}

Statistical Significance:
  P-value vs break-even: {self.p_value_vs_breakeven:.4f}
  P-value vs market: {self.p_value_vs_market:.4f}

Risk Metrics:
  Risk of Ruin (50%+ loss): {self.risk_of_ruin:.1%}
  Probability Profitable: {self.probability_profitable:.1%}
  Expected Kelly Growth: {self.expected_kelly_growth:.2%}

Transaction Costs:
  Gross ROI: {self.gross_roi:.2%}
  Net ROI: {self.net_roi:.2%}
  Total Slippage: ${self.total_slippage:.2f}
  Total Vig Paid: ${self.total_vig_paid:.2f}
  Bets Not Executed: {self.bets_not_executed}

Walk-Forward Folds: {len(self.fold_results)}
Test Seasons: {', '.join(map(str, self.test_seasons))}
"""
