"""
Monte Carlo simulation for betting outcomes.

Two approaches:
1. Bootstrap resampling: Resample from actual bet outcomes
2. Parametric simulation: Simulate from estimated distribution
"""

from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np

from .core import BacktestConfig, RigorousBet


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""

    roi_distribution: np.ndarray
    sharpe_distribution: np.ndarray
    drawdown_distribution: np.ndarray
    final_bankroll_distribution: np.ndarray

    # Percentiles
    percentiles: Dict[str, Dict[int, float]] = field(default_factory=dict)

    # Summary stats
    mean_roi: float = 0.0
    std_roi: float = 0.0
    probability_profitable: float = 0.0
    risk_of_ruin: float = 0.0  # P(bankroll < 50% of initial)


class MonteCarloSimulator:
    """
    Monte Carlo simulation for betting outcomes.

    Bootstrap is preferred when sample size is sufficient (>100 bets).
    Parametric is useful for small samples or projections.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

    def run_bootstrap_simulation(
        self,
        bets: List[RigorousBet],
        n_simulations: int = None,
    ) -> MonteCarloResult:
        """
        Bootstrap resampling of bet outcomes.

        Process:
        1. For each simulation:
           a. Resample bets with replacement
           b. Replay the sequence calculating bankroll
           c. Record final metrics
        2. Build distribution of outcomes

        This preserves:
        - Actual bet size distribution
        - Correlation between edge and outcome
        - Real transaction costs

        Args:
            bets: List of bet results
            n_simulations: Number of simulations

        Returns:
            MonteCarloResult with distributions
        """
        n_simulations = n_simulations or self.config.n_simulations
        n_bets = len(bets)

        if n_bets < 30:
            print(f"Warning: Only {n_bets} bets - bootstrap may be unreliable")

        # Extract bet outcomes and sizes
        outcomes = np.array([1 if b.won else 0 for b in bets])
        bet_sizes = np.array([b.bet_size for b in bets])
        payouts = np.array([b.potential_profit if hasattr(b, 'potential_profit')
                           else b.pnl if b.won else -b.bet_size for b in bets])

        # Run simulations
        roi_results = np.zeros(n_simulations)
        sharpe_results = np.zeros(n_simulations)
        drawdown_results = np.zeros(n_simulations)
        final_bankroll_results = np.zeros(n_simulations)

        initial_bankroll = self.config.initial_bankroll

        for sim in range(n_simulations):
            # Resample bet indices with replacement
            indices = self.rng.choice(n_bets, size=n_bets, replace=True)

            # Replay sequence
            bankroll = initial_bankroll
            peak = initial_bankroll
            max_dd = 0
            returns = []
            total_wagered = 0

            for idx in indices:
                bet_size = bet_sizes[idx]

                # Scale bet size to current bankroll
                # (maintains fractional Kelly logic)
                scaled_bet = bet_size * (bankroll / initial_bankroll)
                scaled_bet = min(scaled_bet, bankroll * self.config.max_bet_fraction)
                scaled_bet = min(scaled_bet, bankroll - 0.1 * initial_bankroll)  # Floor

                if scaled_bet < 1.0:
                    # Can't bet less than $1
                    continue

                if outcomes[idx] == 1:
                    # Win - calculate profit
                    profit_ratio = payouts[idx] / bet_sizes[idx] if bet_sizes[idx] > 0 else 0
                    profit = scaled_bet * profit_ratio
                else:
                    # Loss
                    profit = -scaled_bet

                bankroll += profit
                total_wagered += scaled_bet

                if bankroll > peak:
                    peak = bankroll

                dd = (peak - bankroll) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)

                if bankroll > 0:
                    returns.append(profit / bankroll)

                # Stop if bankrupt
                if bankroll <= 0:
                    bankroll = 0
                    break

            # Record metrics
            roi_results[sim] = (
                (bankroll - initial_bankroll) / total_wagered
                if total_wagered > 0 else -1
            )
            drawdown_results[sim] = max_dd
            final_bankroll_results[sim] = bankroll

            if len(returns) > 1 and np.std(returns) > 0:
                sharpe_results[sim] = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe_results[sim] = 0

        return MonteCarloResult(
            roi_distribution=roi_results,
            sharpe_distribution=sharpe_results,
            drawdown_distribution=drawdown_results,
            final_bankroll_distribution=final_bankroll_results,
            percentiles=self._calculate_percentiles(
                roi_results, sharpe_results, drawdown_results
            ),
            mean_roi=np.mean(roi_results),
            std_roi=np.std(roi_results),
            probability_profitable=np.mean(roi_results > 0),
            risk_of_ruin=np.mean(final_bankroll_results < initial_bankroll * 0.5),
        )

    def run_parametric_simulation(
        self,
        win_rate: float,
        avg_odds: float,
        avg_bet_size: float,
        n_bets: int,
        n_simulations: int = None,
    ) -> MonteCarloResult:
        """
        Parametric simulation using estimated parameters.

        Useful when:
        - Sample size is too small for bootstrap
        - Want to project future performance
        - Testing sensitivity to parameters

        Args:
            win_rate: Expected win rate
            avg_odds: Average decimal odds
            avg_bet_size: Average bet size
            n_bets: Number of bets to simulate
            n_simulations: Number of simulations

        Returns:
            MonteCarloResult
        """
        n_simulations = n_simulations or self.config.n_simulations

        roi_results = np.zeros(n_simulations)
        sharpe_results = np.zeros(n_simulations)
        drawdown_results = np.zeros(n_simulations)
        final_bankroll_results = np.zeros(n_simulations)

        initial_bankroll = self.config.initial_bankroll

        for sim in range(n_simulations):
            # Simulate bet sequence
            outcomes = self.rng.binomial(1, win_rate, n_bets)

            bankroll = initial_bankroll
            peak = initial_bankroll
            max_dd = 0
            returns = []
            total_wagered = 0

            for won in outcomes:
                # Scale bet size to current bankroll
                bet_size = min(
                    avg_bet_size * (bankroll / initial_bankroll),
                    bankroll * self.config.max_bet_fraction,
                )

                if bet_size < 1.0:
                    continue

                if won:
                    profit = bet_size * (avg_odds - 1)
                else:
                    profit = -bet_size

                bankroll += profit
                total_wagered += bet_size

                if bankroll > peak:
                    peak = bankroll

                dd = (peak - bankroll) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)

                if bankroll > 0:
                    returns.append(profit / bankroll)

                if bankroll <= 0:
                    bankroll = 0
                    break

            roi_results[sim] = (
                (bankroll - initial_bankroll) / total_wagered
                if total_wagered > 0 else -1
            )
            drawdown_results[sim] = max_dd
            final_bankroll_results[sim] = bankroll

            if len(returns) > 1 and np.std(returns) > 0:
                sharpe_results[sim] = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe_results[sim] = 0

        return MonteCarloResult(
            roi_distribution=roi_results,
            sharpe_distribution=sharpe_results,
            drawdown_distribution=drawdown_results,
            final_bankroll_distribution=final_bankroll_results,
            percentiles=self._calculate_percentiles(
                roi_results, sharpe_results, drawdown_results
            ),
            mean_roi=np.mean(roi_results),
            std_roi=np.std(roi_results),
            probability_profitable=np.mean(roi_results > 0),
            risk_of_ruin=np.mean(final_bankroll_results < initial_bankroll * 0.5),
        )

    def _calculate_percentiles(
        self,
        roi: np.ndarray,
        sharpe: np.ndarray,
        drawdown: np.ndarray,
    ) -> Dict[str, Dict[int, float]]:
        """Calculate percentile distributions."""
        percentile_levels = [5, 25, 50, 75, 95]

        return {
            "roi": {p: float(np.percentile(roi, p)) for p in percentile_levels},
            "sharpe": {p: float(np.percentile(sharpe, p)) for p in percentile_levels},
            "max_drawdown": {p: float(np.percentile(drawdown, p)) for p in percentile_levels},
        }
