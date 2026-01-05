"""
Statistical analysis with proper inference.

Provides:
- Bootstrap confidence intervals (BCa method)
- P-values for strategy performance
- Multiple testing correction
- Effect size calculations
"""

from typing import Tuple, List, Callable
import numpy as np
from scipy import stats

from .core import BacktestConfig, RigorousBet


class StatisticalAnalyzer:
    """
    Statistical analysis with proper inference.

    Key methods:
    - Bootstrap confidence intervals
    - P-values for strategy performance
    - Multiple testing correction
    - Effect size calculations
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

    def bootstrap_ci(
        self,
        data: np.ndarray,
        statistic: Callable,
        n_bootstrap: int = None,
        confidence_level: float = None,
        method: str = "percentile",
    ) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval.

        Methods:
        - percentile: Simple percentile method
        - bca: Bias-corrected and accelerated (more accurate)
        - basic: Basic bootstrap

        Args:
            data: Sample data
            statistic: Function to compute statistic (e.g., np.mean)
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (e.g., 0.95)
            method: Bootstrap method

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        n_bootstrap = n_bootstrap or self.config.bootstrap_samples
        confidence_level = confidence_level or self.config.confidence_level

        n = len(data)
        if n < 10:
            # Not enough data for reliable bootstrap
            return (np.nan, np.nan)

        boot_stats = np.zeros(n_bootstrap)

        for i in range(n_bootstrap):
            boot_sample = self.rng.choice(data, size=n, replace=True)
            boot_stats[i] = statistic(boot_sample)

        alpha = 1 - confidence_level

        if method == "percentile":
            lower = np.percentile(boot_stats, 100 * alpha / 2)
            upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))

        elif method == "bca":
            # Bias-corrected and accelerated
            lower, upper = self._bca_interval(data, boot_stats, statistic, alpha)

        elif method == "basic":
            theta_hat = statistic(data)
            lower = 2 * theta_hat - np.percentile(boot_stats, 100 * (1 - alpha / 2))
            upper = 2 * theta_hat - np.percentile(boot_stats, 100 * alpha / 2)
        else:
            raise ValueError(f"Unknown method: {method}")

        return (lower, upper)

    def _bca_interval(
        self,
        data: np.ndarray,
        boot_stats: np.ndarray,
        statistic: Callable,
        alpha: float,
    ) -> Tuple[float, float]:
        """
        Calculate BCa (bias-corrected and accelerated) interval.

        More accurate than simple percentile, especially for skewed distributions.
        """
        theta_hat = statistic(data)
        n = len(data)

        # Bias correction
        z0 = stats.norm.ppf(np.mean(boot_stats < theta_hat))

        # Acceleration (using jackknife)
        jackknife_stats = np.zeros(n)
        for i in range(n):
            jack_sample = np.delete(data, i)
            jackknife_stats[i] = statistic(jack_sample)

        jack_mean = np.mean(jackknife_stats)
        num = np.sum((jack_mean - jackknife_stats) ** 3)
        denom = 6 * (np.sum((jack_mean - jackknife_stats) ** 2) ** 1.5)
        a = num / denom if denom != 0 else 0

        # Adjusted percentiles
        z_alpha_low = stats.norm.ppf(alpha / 2)
        z_alpha_high = stats.norm.ppf(1 - alpha / 2)

        def adjusted_percentile(z_alpha):
            return stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))

        lower_pct = 100 * adjusted_percentile(z_alpha_low)
        upper_pct = 100 * adjusted_percentile(z_alpha_high)

        # Clip to valid percentile range
        lower_pct = np.clip(lower_pct, 0, 100)
        upper_pct = np.clip(upper_pct, 0, 100)

        lower = np.percentile(boot_stats, lower_pct)
        upper = np.percentile(boot_stats, upper_pct)

        return (lower, upper)

    def calculate_roi_ci(
        self,
        bets: List[RigorousBet],
        method: str = "bca",
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Calculate ROI with confidence interval.

        Args:
            bets: List of bet results
            method: Bootstrap method

        Returns:
            Tuple of (point_estimate, (lower, upper))
        """
        if not bets:
            return (0.0, (0.0, 0.0))

        returns = np.array([b.pnl / b.bet_size if b.bet_size > 0 else 0 for b in bets])

        roi = np.mean(returns)
        ci = self.bootstrap_ci(returns, np.mean, method=method)

        return roi, ci

    def calculate_win_rate_ci(
        self,
        bets: List[RigorousBet],
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Calculate win rate with confidence interval.

        Uses Wilson score interval for better small-sample properties.
        """
        if not bets:
            return (0.0, (0.0, 0.0))

        n = len(bets)
        wins = sum(1 for b in bets if b.won)
        p_hat = wins / n

        # Wilson score interval
        z = stats.norm.ppf(1 - (1 - self.config.confidence_level) / 2)

        denominator = 1 + z**2 / n
        center = (p_hat + z**2 / (2 * n)) / denominator
        spread = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denominator

        lower = center - spread
        upper = center + spread

        return p_hat, (max(0, lower), min(1, upper))

    def calculate_sharpe_ci(
        self,
        bets: List[RigorousBet],
        method: str = "percentile",
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Calculate Sharpe ratio with confidence interval.

        Uses bootstrap since Sharpe distribution is complex.
        """
        if not bets or len(bets) < 2:
            return (0.0, (0.0, 0.0))

        returns = np.array([b.pnl / b.bet_size if b.bet_size > 0 else 0 for b in bets])

        def sharpe_statistic(r):
            if len(r) < 2 or np.std(r, ddof=1) == 0:
                return 0
            return np.mean(r) / np.std(r, ddof=1) * np.sqrt(252)  # Annualized

        sharpe = sharpe_statistic(returns)
        ci = self.bootstrap_ci(returns, sharpe_statistic, method=method)

        return sharpe, ci

    def calculate_p_value(
        self,
        bets: List[RigorousBet],
        null_roi: float = 0.0,
        method: str = "permutation",
    ) -> float:
        """
        Calculate p-value for ROI vs null hypothesis.

        Uses permutation test for robustness.

        Args:
            bets: List of bet results
            null_roi: Null hypothesis ROI (default 0 = break-even)
            method: "permutation" or "bootstrap"

        Returns:
            One-sided p-value (P(ROI >= observed | H0))
        """
        if not bets:
            return 1.0

        returns = np.array([b.pnl / b.bet_size if b.bet_size > 0 else 0 for b in bets])
        observed_roi = np.mean(returns)

        if method == "permutation":
            # Shift returns to have mean = null_roi
            shifted_returns = returns - np.mean(returns) + null_roi

            n_permutations = 10000
            count_more_extreme = 0

            for _ in range(n_permutations):
                perm_sample = self.rng.choice(
                    shifted_returns, size=len(returns), replace=True
                )
                perm_roi = np.mean(perm_sample)
                if perm_roi >= observed_roi:
                    count_more_extreme += 1

            return count_more_extreme / n_permutations

        elif method == "t_test":
            # One-sample t-test
            if len(returns) < 2:
                return 1.0

            t_stat = (np.mean(returns) - null_roi) / (np.std(returns, ddof=1) / np.sqrt(len(returns)))
            p_value = 1 - stats.t.cdf(t_stat, df=len(returns) - 1)

            return p_value

        else:
            raise ValueError(f"Unknown method: {method}")

    def multiple_testing_correction(
        self,
        p_values: List[float],
        method: str = None,
    ) -> Tuple[List[float], List[bool]]:
        """
        Apply multiple testing correction.

        Methods:
        - bonferroni: Conservative, controls FWER
        - holm: Less conservative, controls FWER
        - fdr_bh: Benjamini-Hochberg, controls FDR

        Args:
            p_values: List of uncorrected p-values
            method: Correction method

        Returns:
            Tuple of (adjusted_p_values, significant_flags)
        """
        method = method or self.config.multiple_testing_method
        n = len(p_values)
        p_array = np.array(p_values)

        if method == "bonferroni":
            adjusted = np.minimum(p_array * n, 1.0)

        elif method == "holm":
            sorted_idx = np.argsort(p_array)
            adjusted = np.zeros(n)
            for i, idx in enumerate(sorted_idx):
                adjusted[idx] = min(p_array[idx] * (n - i), 1.0)

            # Ensure monotonicity
            for i in range(1, n):
                adjusted[sorted_idx[i]] = max(
                    adjusted[sorted_idx[i]], adjusted[sorted_idx[i - 1]]
                )

        elif method == "fdr_bh":
            # Benjamini-Hochberg FDR control
            sorted_idx = np.argsort(p_array)
            adjusted = np.zeros(n)

            for i, idx in enumerate(sorted_idx):
                adjusted[idx] = p_array[idx] * n / (i + 1)

            # Ensure monotonicity (reverse direction)
            for i in range(n - 2, -1, -1):
                adjusted[sorted_idx[i]] = min(
                    adjusted[sorted_idx[i]], adjusted[sorted_idx[i + 1]]
                )

            adjusted = np.minimum(adjusted, 1.0)

        else:
            raise ValueError(f"Unknown method: {method}")

        alpha = 1 - self.config.confidence_level
        significant = adjusted < alpha

        return adjusted.tolist(), significant.tolist()

    def calculate_effect_size(
        self,
        bets: List[RigorousBet],
        null_roi: float = 0.0,
    ) -> float:
        """
        Calculate Cohen's d effect size for ROI.

        Measures how many standard deviations ROI is from null hypothesis.

        Args:
            bets: List of bet results
            null_roi: Null hypothesis ROI

        Returns:
            Cohen's d effect size
        """
        if not bets or len(bets) < 2:
            return 0.0

        returns = np.array([b.pnl / b.bet_size if b.bet_size > 0 else 0 for b in bets])

        mean_roi = np.mean(returns)
        std_roi = np.std(returns, ddof=1)

        if std_roi == 0:
            return 0.0

        cohen_d = (mean_roi - null_roi) / std_roi

        return cohen_d
