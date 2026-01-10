"""
Visualization tools for backtest results.

Creates plots for:
- ROI distribution
- Walk-forward performance
- Drawdown analysis
- Bankroll evolution
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
from pathlib import Path

from .core import RigorousBacktestResult


class BacktestVisualizer:
    """Visualization tools for backtest results."""

    def __init__(self, figsize: tuple = (12, 6)):
        self.figsize = figsize
        plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")

    def plot_roi_distribution(
        self, result: RigorousBacktestResult, save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot ROI distribution from Monte Carlo simulation.

        Shows:
        - Histogram of ROI outcomes
        - Point estimate with CI
        - Percentile markers
        - Probability of profit
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)

        # ROI Distribution
        ax1.hist(
            result.roi_distribution,
            bins=50,
            density=True,
            alpha=0.7,
            color="steelblue",
            edgecolor="black",
        )
        ax1.axvline(
            result.roi, color="red", linestyle="-", linewidth=2, label=f"ROI: {result.roi:.1%}"
        )
        ax1.axvline(result.roi_ci[0], color="red", linestyle="--", linewidth=1)
        ax1.axvline(result.roi_ci[1], color="red", linestyle="--", linewidth=1)
        ax1.axvline(0, color="black", linestyle="-", linewidth=1, label="Break-even")

        ax1.set_xlabel("ROI")
        ax1.set_ylabel("Density")
        ax1.set_title(
            f"ROI Distribution (n={result.num_bets} bets)\n"
            f"P(Profitable) = {result.probability_profitable:.1%}"
        )
        ax1.legend()

        # Bankroll evolution
        if len(result.bankroll_history) > 0:
            ax2.plot(result.bankroll_history, color="steelblue", alpha=0.8)
            ax2.axhline(
                result.config.initial_bankroll, color="black", linestyle="--", label="Initial"
            )
            ax2.fill_between(
                range(len(result.bankroll_history)),
                result.bankroll_history,
                result.config.initial_bankroll,
                alpha=0.3,
            )
            ax2.set_xlabel("Bet Number")
            ax2.set_ylabel("Bankroll ($)")
            ax2.set_title("Bankroll Evolution")
            ax2.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_walk_forward_performance(
        self, result: RigorousBacktestResult, save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot walk-forward validation results by fold.

        Shows:
        - ROI by fold with CI
        - Win rate by fold
        - Consistency across time
        """
        if not result.fold_results:
            print("No fold results to plot")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))

        folds = result.fold_results
        n_folds = len(folds)
        x = range(n_folds)

        # ROI by fold
        ax = axes[0, 0]
        rois = [f.roi for f in folds]
        roi_cis = [f.roi_ci for f in folds]

        ax.bar(x, rois, color="steelblue", alpha=0.7)
        ax.errorbar(
            x,
            rois,
            yerr=[[r - ci[0] for r, ci in zip(rois, roi_cis)], [ci[1] - r for r, ci in zip(rois, roi_cis)]],
            fmt="none",
            color="black",
            capsize=3,
        )
        ax.axhline(0, color="red", linestyle="--")
        ax.set_xlabel("Fold")
        ax.set_ylabel("ROI")
        ax.set_title("ROI by Walk-Forward Fold")

        # Win rate by fold
        ax = axes[0, 1]
        win_rates = [f.win_rate for f in folds]
        ax.bar(x, win_rates, color="seagreen", alpha=0.7)
        ax.axhline(0.524, color="red", linestyle="--", label="Break-even (52.4%)")
        ax.set_xlabel("Fold")
        ax.set_ylabel("Win Rate")
        ax.set_title("Win Rate by Fold")
        ax.legend()

        # Sample size by fold
        ax = axes[1, 0]
        sizes = [f.num_bets for f in folds]
        ax.bar(x, sizes, color="coral", alpha=0.7)
        ax.set_xlabel("Fold")
        ax.set_ylabel("Number of Bets")
        ax.set_title("Sample Size by Fold")

        # Cumulative P&L
        ax = axes[1, 1]
        cum_pnl = np.cumsum([f.total_pnl for f in folds])
        ax.plot(x, cum_pnl, marker="o", color="steelblue")
        ax.fill_between(x, 0, cum_pnl, alpha=0.3)
        ax.axhline(0, color="black", linestyle="--")
        ax.set_xlabel("Fold")
        ax.set_ylabel("Cumulative P&L ($)")
        ax.set_title("Cumulative Performance Across Folds")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_drawdown_distribution(
        self, result: RigorousBacktestResult, save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot max drawdown distribution from Monte Carlo."""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(
            result.drawdown_distribution,
            bins=50,
            density=True,
            alpha=0.7,
            color="indianred",
            edgecolor="black",
        )
        ax.axvline(
            result.max_drawdown,
            color="darkred",
            linestyle="-",
            linewidth=2,
            label=f"Actual: {result.max_drawdown:.1%}",
        )
        ax.axvline(result.max_drawdown_ci[0], color="darkred", linestyle="--")
        ax.axvline(result.max_drawdown_ci[1], color="darkred", linestyle="--")

        ax.set_xlabel("Max Drawdown")
        ax.set_ylabel("Density")
        ax.set_title(
            f"Max Drawdown Distribution\n"
            f"Risk of Ruin (50%+ loss): {result.risk_of_ruin:.1%}"
        )
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def create_summary_dashboard(
        self, result: RigorousBacktestResult, save_dir: Optional[str] = None
    ):
        """
        Create comprehensive dashboard with all plots.

        Args:
            result: Backtest result
            save_dir: Directory to save plots (optional)
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        # Create all plots
        self.plot_roi_distribution(
            result, save_path=save_dir / "roi_distribution.png" if save_dir else None
        )
        self.plot_walk_forward_performance(
            result, save_path=save_dir / "walk_forward.png" if save_dir else None
        )
        self.plot_drawdown_distribution(
            result, save_path=save_dir / "drawdown.png" if save_dir else None
        )

        print(f"Dashboard saved to {save_dir}")

        plt.show()
