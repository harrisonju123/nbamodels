"""
Walk-forward validation engine to prevent data leakage.

Key principle: Never use future data for training.
Models only see data available at prediction time.
"""

from typing import Generator, Tuple, List
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np

from .core import BacktestConfig


@dataclass
class WalkForwardFold:
    """Single fold in walk-forward validation."""

    fold_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_size: int
    test_size: int


class WalkForwardValidator:
    """
    Walk-forward validation with expanding or rolling windows.

    Two modes:
    1. Expanding window: Train on all data up to current point
       - More stable estimates
       - Captures long-term patterns

    2. Rolling window: Train on fixed-size recent window
       - Adapts to regime changes
       - May miss long-term patterns
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.folds: List[WalkForwardFold] = []

    def generate_folds(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, WalkForwardFold], None, None]:
        """
        Generate train/test splits chronologically.

        Args:
            df: DataFrame with game data
            date_col: Name of date column

        Yields:
            Tuple of (train_df, test_df, fold_info) for each fold
        """
        df = df.copy()
        df = df.sort_values(date_col).reset_index(drop=True)

        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])

        if self.config.retrain_frequency == "monthly":
            yield from self._generate_monthly_folds(df, date_col)
        elif self.config.retrain_frequency == "quarterly":
            yield from self._generate_quarterly_folds(df, date_col)
        else:
            # N games format (e.g., "50_games")
            n_games = int(self.config.retrain_frequency.replace("_games", ""))
            yield from self._generate_n_games_folds(df, n_games, date_col)

    def _generate_monthly_folds(
        self,
        df: pd.DataFrame,
        date_col: str,
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, WalkForwardFold], None, None]:
        """Generate monthly retraining folds."""
        df['year_month'] = pd.to_datetime(df[date_col]).dt.to_period('M')
        months = df['year_month'].unique()

        # Need minimum training data
        min_train_months = self.config.initial_train_size // 150  # ~150 games/month
        min_train_months = max(3, min_train_months)  # At least 3 months

        fold_id = 0
        for i, test_month in enumerate(months[min_train_months:], start=min_train_months):
            if self.config.use_expanding_window:
                # Expanding: all data up to test month
                train_mask = df['year_month'] < test_month
            else:
                # Rolling: fixed window (12 months)
                window_months = 12
                start_month = months[max(0, i - window_months)]
                train_mask = (df['year_month'] >= start_month) & (df['year_month'] < test_month)

            test_mask = df['year_month'] == test_month

            train_df = df[train_mask].drop('year_month', axis=1).copy()
            test_df = df[test_mask].drop('year_month', axis=1).copy()

            if len(train_df) >= self.config.initial_train_size and len(test_df) > 0:
                # Validate no leakage
                self.validate_no_leakage(train_df, test_df, date_col)

                # Create fold info
                fold_info = WalkForwardFold(
                    fold_id=fold_id,
                    train_start=train_df[date_col].min(),
                    train_end=train_df[date_col].max(),
                    test_start=test_df[date_col].min(),
                    test_end=test_df[date_col].max(),
                    train_size=len(train_df),
                    test_size=len(test_df),
                )
                self.folds.append(fold_info)

                yield train_df, test_df, fold_info
                fold_id += 1

    def _generate_quarterly_folds(
        self,
        df: pd.DataFrame,
        date_col: str,
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, WalkForwardFold], None, None]:
        """Generate quarterly retraining folds."""
        df['year_quarter'] = pd.to_datetime(df[date_col]).dt.to_period('Q')
        quarters = df['year_quarter'].unique()

        # Need minimum training data
        min_train_quarters = self.config.initial_train_size // 450  # ~450 games/quarter (NBA)
        min_train_quarters = max(2, min_train_quarters)  # At least 2 quarters

        fold_id = 0
        for i, test_quarter in enumerate(quarters[min_train_quarters:], start=min_train_quarters):
            if self.config.use_expanding_window:
                train_mask = df['year_quarter'] < test_quarter
            else:
                # Rolling: fixed window (4 quarters = 1 year)
                window_quarters = 4
                start_quarter = quarters[max(0, i - window_quarters)]
                train_mask = (df['year_quarter'] >= start_quarter) & (df['year_quarter'] < test_quarter)

            test_mask = df['year_quarter'] == test_quarter

            train_df = df[train_mask].drop('year_quarter', axis=1).copy()
            test_df = df[test_mask].drop('year_quarter', axis=1).copy()

            if len(train_df) >= self.config.initial_train_size and len(test_df) > 0:
                self.validate_no_leakage(train_df, test_df, date_col)

                fold_info = WalkForwardFold(
                    fold_id=fold_id,
                    train_start=train_df[date_col].min(),
                    train_end=train_df[date_col].max(),
                    test_start=test_df[date_col].min(),
                    test_end=test_df[date_col].max(),
                    train_size=len(train_df),
                    test_size=len(test_df),
                )
                self.folds.append(fold_info)

                yield train_df, test_df, fold_info
                fold_id += 1

    def _generate_n_games_folds(
        self,
        df: pd.DataFrame,
        n_games: int,
        date_col: str,
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, WalkForwardFold], None, None]:
        """Generate folds by retraining every N games."""
        total_games = len(df)
        fold_id = 0

        # Start with initial training size
        current_idx = self.config.initial_train_size

        while current_idx < total_games:
            # Test on next N games
            test_end_idx = min(current_idx + n_games, total_games)

            if self.config.use_expanding_window:
                train_df = df.iloc[:current_idx].copy()
            else:
                # Rolling window (use last N games for training)
                train_start_idx = max(0, current_idx - self.config.initial_train_size)
                train_df = df.iloc[train_start_idx:current_idx].copy()

            test_df = df.iloc[current_idx:test_end_idx].copy()

            if len(train_df) >= self.config.initial_train_size and len(test_df) > 0:
                self.validate_no_leakage(train_df, test_df, date_col)

                fold_info = WalkForwardFold(
                    fold_id=fold_id,
                    train_start=train_df[date_col].min(),
                    train_end=train_df[date_col].max(),
                    test_start=test_df[date_col].min(),
                    test_end=test_df[date_col].max(),
                    train_size=len(train_df),
                    test_size=len(test_df),
                )
                self.folds.append(fold_info)

                yield train_df, test_df, fold_info
                fold_id += 1

            current_idx = test_end_idx

    def validate_no_leakage(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        date_col: str = "date",
    ) -> bool:
        """
        Verify no data leakage between train and test.

        Checks:
        1. All train dates < all test dates
        2. No overlapping game_ids
        3. Train and test are disjoint

        Args:
            train_df: Training data
            test_df: Test data
            date_col: Name of date column

        Returns:
            True if validation passes

        Raises:
            ValueError: If data leakage detected
        """
        # Ensure date columns are datetime
        train_dates = pd.to_datetime(train_df[date_col])
        test_dates = pd.to_datetime(test_df[date_col])

        train_max_date = train_dates.max()
        test_min_date = test_dates.min()

        # Check temporal ordering
        if train_max_date >= test_min_date:
            raise ValueError(
                f"Data leakage detected! Train max date {train_max_date} >= "
                f"Test min date {test_min_date}"
            )

        # Check for overlapping game IDs if column exists
        if 'game_id' in train_df.columns and 'game_id' in test_df.columns:
            train_ids = set(train_df['game_id'])
            test_ids = set(test_df['game_id'])
            overlap = train_ids & test_ids

            if overlap:
                raise ValueError(
                    f"Data leakage: {len(overlap)} overlapping game_ids found"
                )

        # Ensure no row overlap
        train_indices = set(train_df.index)
        test_indices = set(test_df.index)
        index_overlap = train_indices & test_indices

        if index_overlap:
            raise ValueError(
                f"Data leakage: {len(index_overlap)} overlapping indices found"
            )

        return True

    def get_fold_summary(self) -> pd.DataFrame:
        """
        Get summary of all folds.

        Returns:
            DataFrame with fold information
        """
        if not self.folds:
            return pd.DataFrame()

        records = []
        for fold in self.folds:
            records.append({
                'fold_id': fold.fold_id,
                'train_start': fold.train_start,
                'train_end': fold.train_end,
                'test_start': fold.test_start,
                'test_end': fold.test_end,
                'train_size': fold.train_size,
                'test_size': fold.test_size,
                'train_months': (fold.train_end - fold.train_start).days / 30,
                'test_months': (fold.test_end - fold.test_start).days / 30,
            })

        return pd.DataFrame(records)
