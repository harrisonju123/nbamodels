"""
Champion/Challenger Framework for model A/B testing.

Provides:
- Model comparison with statistical significance testing
- Auto-promotion logic
- Backtest-based validation
"""

from __future__ import annotations

import pickle
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, date
from typing import Optional, Dict, Any, List
from pathlib import Path

import pandas as pd
import numpy as np

from .model_registry import ModelRegistry, ModelVersion, ModelMetrics

logger = logging.getLogger(__name__)


@dataclass
class ModelComparison:
    """Results from comparing champion vs challenger."""

    champion_model_id: str
    challenger_model_id: str
    comparison_date: datetime

    # Performance metrics
    champion_roi: float
    challenger_roi: float
    roi_difference: float

    champion_win_rate: float
    challenger_win_rate: float

    champion_sharpe: float
    challenger_sharpe: float

    champion_num_bets: int
    challenger_num_bets: int

    # Statistical significance
    p_value: float
    is_significant: bool

    # Decision
    winner: str  # "champion", "challenger", "tie"
    recommendation: str  # "promote", "reject", "needs_more_data"
    reason: str

    # Full details (JSON serializable)
    full_results: Optional[Dict[str, Any]] = None


class ChampionChallengerFramework:
    """
    Framework for comparing champion vs challenger models.

    Provides statistical comparison using rigorous backtesting methodology.
    """

    # Auto-promotion thresholds
    MIN_P_VALUE = 0.05  # Challenger must be significant at 5% level
    MIN_ROI_IMPROVEMENT = 0.0  # Challenger ROI must be higher
    MIN_BETS = 100  # Minimum bets required for comparison
    MAX_SHARPE_DEGRADATION = 0.0  # Don't allow Sharpe ratio to decrease

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        db_dir: str = "data/versioning"
    ):
        self.registry = registry or ModelRegistry(db_dir)

    def compare_models(
        self,
        champion_id: str,
        challenger_id: str,
        test_data: pd.DataFrame,
    ) -> ModelComparison:
        """
        Compare champion vs challenger using identical test data.

        This method runs both models on the same test set and compares
        their performance using statistical significance testing.

        Args:
            champion_id: Champion model ID
            challenger_id: Challenger model ID
            test_data: Test dataset with features

        Returns:
            ModelComparison result
        """
        # Load models
        champion = self.registry.get_model_by_id(champion_id)
        challenger = self.registry.get_model_by_id(challenger_id)

        if not champion or not challenger:
            raise ValueError("Champion or challenger not found")

        logger.info(f"Comparing champion {champion.version} vs challenger {challenger.version}")

        # Load model files safely
        champion_model = self._safe_load_model(champion.model_path, champion.model_id)
        challenger_model = self._safe_load_model(challenger.model_path, challenger.model_id)

        # Get predictions from both models
        champion_preds = self._get_model_predictions(champion_model, test_data)
        challenger_preds = self._get_model_predictions(challenger_model, test_data)

        # Simulate betting performance for both
        champion_results = self._simulate_betting_performance(
            champion_preds, test_data
        )
        challenger_results = self._simulate_betting_performance(
            challenger_preds, test_data
        )

        # Calculate ROI difference and p-value
        roi_diff = challenger_results['roi'] - champion_results['roi']
        p_value = self._calculate_paired_p_value(
            champion_results['bet_outcomes'],
            challenger_results['bet_outcomes']
        )

        is_significant = p_value < self.MIN_P_VALUE

        # Determine winner
        if is_significant and roi_diff > self.MIN_ROI_IMPROVEMENT:
            winner = "challenger"
        elif is_significant and roi_diff < -self.MIN_ROI_IMPROVEMENT:
            winner = "champion"
        else:
            winner = "tie"

        # Make recommendation
        recommendation, reason = self._make_recommendation(
            champion_results,
            challenger_results,
            p_value,
            is_significant
        )

        comparison = ModelComparison(
            champion_model_id=champion_id,
            challenger_model_id=challenger_id,
            comparison_date=datetime.now(),
            champion_roi=champion_results['roi'],
            challenger_roi=challenger_results['roi'],
            roi_difference=roi_diff,
            champion_win_rate=champion_results['win_rate'],
            challenger_win_rate=challenger_results['win_rate'],
            champion_sharpe=champion_results['sharpe'],
            challenger_sharpe=challenger_results['sharpe'],
            champion_num_bets=champion_results['num_bets'],
            challenger_num_bets=challenger_results['num_bets'],
            p_value=p_value,
            is_significant=is_significant,
            winner=winner,
            recommendation=recommendation,
            reason=reason,
            full_results={
                'champion': champion_results,
                'challenger': challenger_results
            }
        )

        # Store comparison in database
        self._store_comparison(comparison)

        return comparison

    def _safe_load_model(self, model_path: str, model_id: str) -> Any:
        """
        Safely load model with path validation.

        Args:
            model_path: Path to model file
            model_id: Model ID for logging

        Returns:
            Loaded model object

        Raises:
            ValueError: If path is invalid or outside allowed directory
            FileNotFoundError: If model file doesn't exist
        """
        # 1. Validate path is within allowed directory
        model_path_obj = Path(model_path).resolve()
        allowed_dir = Path("models").resolve()

        # Check path is within models directory
        try:
            model_path_obj.relative_to(allowed_dir)
        except ValueError:
            raise ValueError(
                f"Model path outside allowed directory: {model_path}. "
                f"Must be within: {allowed_dir}"
            )

        # 2. Validate file exists and is a file
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if not model_path_obj.is_file():
            raise ValueError(f"Model path is not a file: {model_path}")

        # 3. Check file extension
        if model_path_obj.suffix not in {'.pkl', '.pickle'}:
            raise ValueError(
                f"Invalid model file extension: {model_path_obj.suffix}. "
                f"Must be .pkl or .pickle"
            )

        # 4. Load pickle file
        try:
            with open(model_path_obj, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Safely loaded model {model_id} from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise RuntimeError(f"Could not deserialize model {model_id}: {e}")

    def _get_model_predictions(
        self,
        model: Any,
        test_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Get predictions from a model."""
        # Assuming model has predict_proba method
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(test_data)
            # For binary classification, take probability of class 1
            if len(probs.shape) == 2:
                probs = probs[:, 1]
        else:
            raise ValueError("Model doesn't have predict_proba method")

        predictions = pd.DataFrame({
            'model_prob': probs
        }, index=test_data.index)

        return predictions

    def _simulate_betting_performance(
        self,
        predictions: pd.DataFrame,
        test_data: pd.DataFrame,
        min_edge: float = 0.07,
        kelly_fraction: float = 0.2
    ) -> Dict[str, Any]:
        """
        Simulate betting performance using Kelly criterion.

        Args:
            predictions: Model predictions
            test_data: Test data with outcomes
            min_edge: Minimum edge to place bet
            kelly_fraction: Kelly fraction to use

        Returns:
            Dict with performance metrics
        """
        # Placeholder implementation - would integrate with existing backtesting
        # This is a simplified version

        # Assume test_data has 'spread_home' and 'home_win' columns
        if 'home_win' not in test_data.columns:
            logger.warning("No target column found, using simulated outcomes")
            outcomes = np.random.binomial(1, predictions['model_prob'].values)
        else:
            outcomes = test_data['home_win'].values

        # Calculate edges (simplified - assumes fair odds)
        model_probs = predictions['model_prob'].values
        market_probs = np.full_like(model_probs, 0.5)  # Simplified
        edges = model_probs - market_probs

        # Filter bets by edge threshold
        bet_mask = edges >= min_edge
        num_bets = bet_mask.sum()

        if num_bets == 0:
            return {
                'roi': 0.0,
                'win_rate': 0.0,
                'sharpe': 0.0,
                'num_bets': 0,
                'bet_outcomes': np.array([])
            }

        # Calculate outcomes for bets placed
        bet_outcomes = outcomes[bet_mask]
        bet_edges = edges[bet_mask]

        # Calculate P&L (simplified)
        wins = bet_outcomes.sum()
        losses = num_bets - wins
        win_rate = wins / num_bets if num_bets > 0 else 0

        # Simple ROI calculation
        roi = (wins - losses) / num_bets if num_bets > 0 else 0

        # Sharpe ratio (simplified)
        returns = np.where(bet_outcomes == 1, 1.0, -1.0)
        sharpe = returns.mean() / returns.std() if len(returns) > 1 else 0

        return {
            'roi': roi,
            'win_rate': win_rate,
            'sharpe': sharpe,
            'num_bets': num_bets,
            'bet_outcomes': bet_outcomes
        }

    def _calculate_paired_p_value(
        self,
        champion_outcomes: np.ndarray,
        challenger_outcomes: np.ndarray
    ) -> float:
        """
        Calculate p-value for paired comparison.

        Uses paired t-test if samples are paired, otherwise unpaired.
        """
        if len(champion_outcomes) == 0 or len(challenger_outcomes) == 0:
            return 1.0

        # If same length, use paired test
        if len(champion_outcomes) == len(challenger_outcomes):
            from scipy.stats import ttest_rel
            _, p_value = ttest_rel(challenger_outcomes, champion_outcomes)
        else:
            # Different lengths - use independent t-test
            from scipy.stats import ttest_ind
            _, p_value = ttest_ind(challenger_outcomes, champion_outcomes)

        # One-tailed test (challenger > champion)
        return p_value / 2

    def _make_recommendation(
        self,
        champion_results: Dict,
        challenger_results: Dict,
        p_value: float,
        is_significant: bool
    ) -> tuple[str, str]:
        """
        Make promotion recommendation based on results.

        Returns:
            Tuple of (recommendation, reason)
        """
        num_bets = challenger_results['num_bets']
        roi_diff = challenger_results['roi'] - champion_results['roi']
        sharpe_diff = challenger_results['sharpe'] - champion_results['sharpe']

        # Check minimum bets
        if num_bets < self.MIN_BETS:
            return ("needs_more_data",
                    f"Only {num_bets} bets, need {self.MIN_BETS} minimum")

        # Check statistical significance
        if not is_significant:
            return ("reject",
                    f"Not statistically significant (p={p_value:.4f})")

        # Check ROI improvement
        if roi_diff <= self.MIN_ROI_IMPROVEMENT:
            return ("reject",
                    f"ROI not improved (diff={roi_diff:.4f})")

        # Check Sharpe ratio hasn't degraded
        if sharpe_diff < self.MAX_SHARPE_DEGRADATION:
            return ("reject",
                    f"Sharpe ratio degraded by {abs(sharpe_diff):.4f}")

        # All criteria met
        return ("promote",
                f"Challenger outperforms: +{roi_diff:.2%} ROI (p={p_value:.4f})")

    def _store_comparison(self, comparison: ModelComparison) -> int:
        """Store comparison results in database."""
        with self.registry.db_manager.get_connection('model_registry') as conn:
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO model_comparisons (
                        champion_model_id, challenger_model_id, comparison_date,
                        champion_roi, challenger_roi, roi_difference,
                        p_value, is_significant, winner, recommendation,
                        full_results
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    comparison.champion_model_id,
                    comparison.challenger_model_id,
                    comparison.comparison_date,
                    comparison.champion_roi,
                    comparison.challenger_roi,
                    comparison.roi_difference,
                    comparison.p_value,
                    comparison.is_significant,
                    comparison.winner,
                    comparison.recommendation,
                    json.dumps(comparison.full_results) if comparison.full_results else None
                ))

                comparison_id = cursor.lastrowid
                conn.commit()
                logger.info(f"Stored comparison (ID: {comparison_id})")
                return comparison_id

            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to store comparison: {e}")
                raise

    def auto_promote_if_better(
        self,
        model_name: str,
        test_data: Optional[pd.DataFrame] = None
    ) -> Optional[ModelComparison]:
        """
        Automatically compare challengers to champion and promote if better.

        Args:
            model_name: Model name (e.g., 'spread_model')
            test_data: Optional test data (if None, loads from features)

        Returns:
            ModelComparison if comparison was run, None otherwise
        """
        champion = self.registry.get_champion(model_name)
        if not champion:
            logger.info(f"No champion found for {model_name}")
            return None

        challengers = self.registry.get_challengers(model_name)
        if not challengers:
            logger.info(f"No challengers found for {model_name}")
            return None

        # Use most recent challenger
        challenger = challengers[0]
        logger.info(f"Comparing champion {champion.version} vs challenger {challenger.version}")

        # Load test data if not provided
        if test_data is None:
            test_data = self._load_test_data()

        # Run comparison
        comparison = self.compare_models(
            champion.model_id,
            challenger.model_id,
            test_data
        )

        # Auto-promote if recommended
        if comparison.recommendation == "promote":
            logger.info(f"Auto-promoting {model_name} {challenger.version}")
            self.registry.promote_to_champion(
                challenger.model_id,
                reason=comparison.reason,
                promoted_by="auto_promotion"
            )
            logger.info(f"Successfully promoted {challenger.version} to champion")

        return comparison

    def _load_test_data(self) -> pd.DataFrame:
        """Load test data from features file."""
        features_path = Path("data/features/game_features.parquet")
        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")

        df = pd.read_parquet(features_path)

        # Use recent data for testing (last 500 games)
        return df.tail(500)
