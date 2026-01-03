"""
Signal Validation Framework

Validates new market microstructure signals improve ROI through backtesting.

Requirements for a signal to pass validation:
1. Minimum 100 qualifying bets in backtest
2. Positive ROI (>= 1%)
3. Statistical significance (p < 0.05)
4. Positive CLV on average
5. Out-of-sample validation
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from scipy import stats
from loguru import logger


@dataclass
class SignalValidationResult:
    """Results from signal validation."""
    signal_name: str
    is_valid: bool
    in_sample_roi: float
    out_sample_roi: float
    in_sample_win_rate: float
    out_sample_win_rate: float
    p_value: float
    avg_clv: float
    num_bets: int
    recommendation: str
    notes: str

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'signal_name': self.signal_name,
            'is_valid': self.is_valid,
            'in_sample_roi': self.in_sample_roi,
            'out_sample_roi': self.out_sample_roi,
            'in_sample_win_rate': self.in_sample_win_rate,
            'out_sample_win_rate': self.out_sample_win_rate,
            'p_value': self.p_value,
            'avg_clv': self.avg_clv,
            'num_bets': self.num_bets,
            'recommendation': self.recommendation,
            'notes': self.notes
        }


class SignalValidator:
    """
    Validates new signals improve ROI in backtest.

    Requirements:
    1. Minimum 100 qualifying bets in backtest
    2. Positive ROI (>= 1%)
    3. Statistical significance (p < 0.05)
    4. Positive CLV on average
    5. Out-of-sample validation
    """

    MIN_SAMPLE_SIZE = 100
    MIN_ROI = 0.01  # 1%
    P_VALUE_THRESHOLD = 0.05
    MIN_CLV = 0.0

    def __init__(
        self,
        min_sample_size: int = 100,
        min_roi: float = 0.01,
        p_value_threshold: float = 0.05
    ):
        """Initialize validator with custom thresholds."""
        self.MIN_SAMPLE_SIZE = min_sample_size
        self.MIN_ROI = min_roi
        self.P_VALUE_THRESHOLD = p_value_threshold

    def validate_signal(
        self,
        signal_name: str,
        signal_bets_df: pd.DataFrame,
        baseline_bets_df: pd.DataFrame
    ) -> SignalValidationResult:
        """
        Validate a signal against baseline strategy.

        Args:
            signal_name: Name of the signal
            signal_bets_df: Bets with signal applied (must have columns: outcome, profit, clv)
            baseline_bets_df: Baseline bets without signal

        Returns:
            SignalValidationResult with validation outcome
        """
        logger.info(f"Validating signal: {signal_name}")

        # Validate required columns
        required_cols = ['outcome', 'profit']
        missing = [col for col in required_cols if col not in signal_bets_df.columns]
        if missing:
            return SignalValidationResult(
                signal_name=signal_name,
                is_valid=False,
                in_sample_roi=0.0,
                out_sample_roi=0.0,
                in_sample_win_rate=0.0,
                out_sample_win_rate=0.0,
                p_value=1.0,
                avg_clv=0.0,
                num_bets=len(signal_bets_df),
                recommendation="MISSING_DATA",
                notes=f"Missing required columns: {missing}"
            )

        # Check sample size
        if len(signal_bets_df) < self.MIN_SAMPLE_SIZE:
            return SignalValidationResult(
                signal_name=signal_name,
                is_valid=False,
                in_sample_roi=0.0,
                out_sample_roi=0.0,
                in_sample_win_rate=0.0,
                out_sample_win_rate=0.0,
                p_value=1.0,
                avg_clv=0.0,
                num_bets=len(signal_bets_df),
                recommendation="INSUFFICIENT_DATA",
                notes=f"Only {len(signal_bets_df)} bets, need {self.MIN_SAMPLE_SIZE}"
            )

        # Ensure chronological order before splitting
        if 'logged_at' in signal_bets_df.columns:
            signal_bets_df = signal_bets_df.sort_values('logged_at').reset_index(drop=True)

        # Split into in-sample and out-of-sample (80/20 split chronologically)
        split_idx = int(len(signal_bets_df) * 0.8)
        in_sample = signal_bets_df.iloc[:split_idx]
        out_sample = signal_bets_df.iloc[split_idx:]

        # Calculate in-sample metrics
        in_sample_roi = self._calculate_roi(in_sample)
        in_sample_win_rate = self._calculate_win_rate(in_sample)

        # Calculate out-of-sample metrics
        out_sample_roi = self._calculate_roi(out_sample)
        out_sample_win_rate = self._calculate_win_rate(out_sample)

        # Calculate CLV
        avg_clv = signal_bets_df['clv'].mean() if 'clv' in signal_bets_df.columns else 0.0

        # Statistical significance test (compare signal vs baseline)
        p_value = self._calculate_significance(signal_bets_df, baseline_bets_df)

        # Validation criteria
        is_valid = (
            out_sample_roi >= self.MIN_ROI and
            p_value < self.P_VALUE_THRESHOLD and
            avg_clv > self.MIN_CLV and
            len(signal_bets_df) >= self.MIN_SAMPLE_SIZE
        )

        # Recommendation
        if is_valid:
            recommendation = "APPROVED"
            notes = f"Signal validated. Out-sample ROI: {out_sample_roi:.2%}, p={p_value:.4f}"
        elif out_sample_roi < 0:
            recommendation = "REJECT"
            notes = f"Negative out-sample ROI: {out_sample_roi:.2%}"
        elif p_value >= self.P_VALUE_THRESHOLD:
            recommendation = "NOT_SIGNIFICANT"
            notes = f"No statistical significance: p={p_value:.4f}"
        elif avg_clv <= 0:
            recommendation = "NEGATIVE_CLV"
            notes = f"Negative CLV: {avg_clv:.3f}"
        else:
            recommendation = "MARGINAL"
            notes = "Marginal improvement, consider more data"

        return SignalValidationResult(
            signal_name=signal_name,
            is_valid=is_valid,
            in_sample_roi=in_sample_roi,
            out_sample_roi=out_sample_roi,
            in_sample_win_rate=in_sample_win_rate,
            out_sample_win_rate=out_sample_win_rate,
            p_value=p_value,
            avg_clv=avg_clv,
            num_bets=len(signal_bets_df),
            recommendation=recommendation,
            notes=notes
        )

    def _calculate_roi(self, bets_df: pd.DataFrame) -> float:
        """Calculate ROI from bets DataFrame."""
        if len(bets_df) == 0 or 'profit' not in bets_df.columns:
            return 0.0

        total_profit = bets_df['profit'].sum()
        total_wagered = len(bets_df)  # Assuming unit stakes

        if total_wagered == 0:
            return 0.0

        return total_profit / total_wagered

    def _calculate_win_rate(self, bets_df: pd.DataFrame) -> float:
        """Calculate win rate from bets DataFrame."""
        if len(bets_df) == 0 or 'outcome' not in bets_df.columns:
            return 0.0

        wins = (bets_df['outcome'] == 'win').sum()
        total = len(bets_df)

        return wins / total if total > 0 else 0.0

    def _calculate_significance(
        self,
        signal_bets: pd.DataFrame,
        baseline_bets: pd.DataFrame
    ) -> float:
        """
        Calculate statistical significance using Mann-Whitney U test.

        Uses non-parametric test since betting profits are not normally distributed.
        Tests if signal bets have significantly better ROI than baseline.
        """
        if 'profit' not in signal_bets.columns or 'profit' not in baseline_bets.columns:
            return 1.0

        signal_profits = signal_bets['profit'].values
        baseline_profits = baseline_bets['profit'].values

        # Mann-Whitney U test (non-parametric, robust to non-normal distributions)
        try:
            # alternative='greater' tests if signal > baseline
            statistic, p_value = stats.mannwhitneyu(
                signal_profits,
                baseline_profits,
                alternative='greater'
            )
            return p_value
        except Exception as e:
            logger.warning(f"Error calculating significance: {e}")
            return 1.0

    def run_walk_forward_validation(
        self,
        signal_name: str,
        bets_df: pd.DataFrame,
        n_splits: int = 5
    ) -> Dict:
        """
        Walk-forward cross-validation.

        Splits data chronologically, trains on past, tests on future.
        More realistic than random CV for time series.

        Args:
            signal_name: Name of signal
            bets_df: Historical bets with signal
            n_splits: Number of validation folds

        Returns:
            Dict with validation results per fold
        """
        logger.info(f"Walk-forward validation for {signal_name} with {n_splits} splits")

        if len(bets_df) < n_splits * self.MIN_SAMPLE_SIZE:
            return {
                'error': f"Not enough data for {n_splits} splits",
                'num_bets': len(bets_df)
            }

        # Split into n equal chunks chronologically
        chunk_size = len(bets_df) // (n_splits + 1)
        results = []

        for i in range(n_splits):
            # Train on all data up to split i
            train_end = chunk_size * (i + 1)
            test_start = train_end
            test_end = train_end + chunk_size

            train_df = bets_df.iloc[:train_end]
            test_df = bets_df.iloc[test_start:test_end]

            if len(test_df) < 10:
                continue

            # Calculate metrics on test fold
            test_roi = self._calculate_roi(test_df)
            test_win_rate = self._calculate_win_rate(test_df)
            test_clv = test_df['clv'].mean() if 'clv' in test_df.columns else 0.0

            results.append({
                'fold': i + 1,
                'train_size': len(train_df),
                'test_size': len(test_df),
                'test_roi': test_roi,
                'test_win_rate': test_win_rate,
                'test_clv': test_clv
            })

        # Aggregate results
        if not results:
            return {'error': 'No valid folds'}

        avg_roi = np.mean([r['test_roi'] for r in results])
        avg_win_rate = np.mean([r['test_win_rate'] for r in results])
        avg_clv = np.mean([r['test_clv'] for r in results])
        roi_std = np.std([r['test_roi'] for r in results])

        return {
            'signal_name': signal_name,
            'n_splits': len(results),
            'avg_test_roi': avg_roi,
            'avg_test_win_rate': avg_win_rate,
            'avg_test_clv': avg_clv,
            'roi_std': roi_std,
            'folds': results,
            'is_consistent': roi_std < 0.05  # Less than 5% std deviation
        }


# ========== Signal-Specific Backtesting Functions ==========


def backtest_steam_follow_strategy(
    historical_games: pd.DataFrame,
    steam_signals: pd.DataFrame,
    baseline_bets: pd.DataFrame,
    min_steam_confidence: float = 0.7
) -> pd.DataFrame:
    """
    Backtest: Bet when model edge aligns with steam move.

    Hypothesis: Steam moves indicate sharp money.
    Betting when model + steam agree should have higher CLV.

    Args:
        historical_games: Historical game results
        steam_signals: Detected steam moves
        baseline_bets: Baseline strategy bets
        min_steam_confidence: Minimum steam confidence to follow

    Returns:
        DataFrame of bets with steam filter applied
    """
    logger.info("Backtesting steam follow strategy...")

    # Filter steam signals by confidence
    strong_steam = steam_signals[steam_signals['confidence'] >= min_steam_confidence]

    # Merge with baseline bets
    steam_bets = baseline_bets.merge(
        strong_steam[['game_id', 'bet_type', 'direction', 'confidence']],
        left_on=['game_id', 'bet_type', 'bet_side'],
        right_on=['game_id', 'bet_type', 'direction'],
        how='inner'
    )

    logger.info(f"Steam filter: {len(baseline_bets)} → {len(steam_bets)} bets")

    return steam_bets


def backtest_rlm_strategy(
    historical_games: pd.DataFrame,
    rlm_signals: pd.DataFrame,
    baseline_bets: pd.DataFrame,
    require_model_alignment: bool = True
) -> pd.DataFrame:
    """
    Backtest: Bet with RLM signals.

    Hypothesis: RLM indicates sharp money.
    When model edge + RLM agree, higher win rate expected.

    Args:
        historical_games: Historical game results
        rlm_signals: Detected RLM signals
        baseline_bets: Baseline strategy bets
        require_model_alignment: Only bet when model agrees with RLM

    Returns:
        DataFrame of bets with RLM filter applied
    """
    logger.info("Backtesting RLM strategy...")

    # Merge with baseline bets
    rlm_bets = baseline_bets.merge(
        rlm_signals[['game_id', 'bet_type', 'sharp_side', 'confidence']],
        left_on=['game_id', 'bet_type', 'bet_side'],
        right_on=['game_id', 'bet_type', 'sharp_side'],
        how='inner'
    )

    logger.info(f"RLM filter: {len(baseline_bets)} → {len(rlm_bets)} bets")

    return rlm_bets


if __name__ == "__main__":
    print("=== Signal Validator ===\n")

    validator = SignalValidator(
        min_sample_size=100,
        min_roi=0.01,
        p_value_threshold=0.05
    )

    print(f"Validation criteria:")
    print(f"  Min sample size: {validator.MIN_SAMPLE_SIZE}")
    print(f"  Min ROI: {validator.MIN_ROI:.1%}")
    print(f"  P-value threshold: {validator.P_VALUE_THRESHOLD}")
    print("\nValidator initialized successfully")
