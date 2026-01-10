"""
Rigorous backtesting framework with statistical rigor.

This module provides professional-grade backtesting with:
- Walk-forward validation to prevent data leakage
- Monte Carlo simulation for variance estimation
- Bootstrap confidence intervals on all metrics
- Transaction cost modeling (slippage, vig, execution)
- Position limit constraints
- Statistical significance testing

Example usage:
```python
from src.betting.rigorous_backtest import (
    BacktestConfig,
    RigorousBacktester,
    BacktestVisualizer,
)

# Configure backtest
config = BacktestConfig(
    initial_train_size=500,
    retrain_frequency="monthly",
    kelly_fraction=0.2,
    min_edge_threshold=0.07,
    n_simulations=1000,
    initial_bankroll=10000.0,
)

# Run backtest
backtester = RigorousBacktester(config)
result = backtester.run_backtest(df, feature_cols, target_col)

# Print summary
print(result.summary())

# Visualize
viz = BacktestVisualizer()
viz.create_summary_dashboard(result, save_dir="data/backtest/")
```
"""

from .core import (
    BacktestConfig,
    RigorousBet,
    FoldResult,
    RigorousBacktestResult,
)
from .walk_forward import WalkForwardValidator, WalkForwardFold
from .statistics import StatisticalAnalyzer
from .monte_carlo import MonteCarloSimulator, MonteCarloResult
from .transaction_costs import TransactionCostModel, TransactionCosts
from .constraints import ConstraintManager, BetConstraints, PortfolioState
from .visualizations import BacktestVisualizer

__all__ = [
    # Core classes
    "BacktestConfig",
    "RigorousBet",
    "FoldResult",
    "RigorousBacktestResult",
    # Walk-forward
    "WalkForwardValidator",
    "WalkForwardFold",
    # Statistics
    "StatisticalAnalyzer",
    # Monte Carlo
    "MonteCarloSimulator",
    "MonteCarloResult",
    # Transaction costs
    "TransactionCostModel",
    "TransactionCosts",
    # Constraints
    "ConstraintManager",
    "BetConstraints",
    "PortfolioState",
    # Visualization
    "BacktestVisualizer",
]

__version__ = "1.0.0"
