"""
Feature Experiment Engine for automated feature backtesting.

Provides:
- Automatic feature testing (with/without comparison)
- Statistical significance calculation
- Feature importance analysis
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

import pandas as pd
import numpy as np

from src.versioning import ModelRegistry
from .feature_registry import FeatureRegistry, FeatureExperimentResult

logger = logging.getLogger(__name__)


class FeatureExperiment:
    """
    Automated feature experiment engine.

    Tests features by comparing model performance with/without the feature.
    """

    # Auto-accept thresholds
    MIN_P_VALUE = 0.05
    MIN_ROI_LIFT = 0.005  # 0.5% ROI improvement
    MIN_BETS = 100

    def __init__(
        self,
        feature_registry: Optional[FeatureRegistry] = None,
        model_registry: Optional[ModelRegistry] = None,
        db_dir: str = "data/versioning"
    ):
        self.feature_registry = feature_registry or FeatureRegistry(db_dir)
        self.model_registry = model_registry or ModelRegistry(db_dir)

    def test_feature(
        self,
        feature_name: str,
        baseline_model_id: Optional[str] = None,
        category: str = "custom",
        hypothesis: str = ""
    ) -> FeatureExperimentResult:
        """
        Test a single feature's impact.

        Steps:
        1. Get or propose feature in registry
        2. Get baseline model (champion or specified)
        3. Train new model with feature added
        4. Run identical backtests on both
        5. Calculate lift and significance
        6. Record results and make decision

        Args:
            feature_name: Name of feature to test
            baseline_model_id: Optional baseline model ID (uses champion if None)
            category: Feature category
            hypothesis: Why this feature might help

        Returns:
            FeatureExperimentResult
        """
        logger.info(f"Testing feature: {feature_name}")

        # Step 1: Get or propose feature
        feature = self.feature_registry.get_feature_by_name(feature_name)
        if not feature:
            logger.info(f"Proposing new feature: {feature_name}")
            feature_id = self.feature_registry.propose_feature(
                name=feature_name,
                category=category,
                hypothesis=hypothesis
            )
        else:
            feature_id = feature.feature_id
            logger.info(f"Using existing feature (ID: {feature_id})")

        # Step 2: Get baseline model
        if baseline_model_id:
            baseline_model = self.model_registry.get_model_by_id(baseline_model_id)
        else:
            # Use current champion as baseline
            baseline_model = self.model_registry.get_champion("spread_model")

        if not baseline_model:
            raise ValueError("No baseline model found")

        logger.info(f"Baseline model: {baseline_model.version}")

        # Step 3-4: Train and test (simplified - in real implementation,
        # would integrate with existing training pipeline)
        baseline_results, experiment_results = self._run_comparison(
            baseline_model,
            feature_name
        )

        # Step 5: Calculate statistics
        p_value = self._calculate_p_value(
            baseline_results['roi'],
            experiment_results['roi'],
            baseline_results['n_bets'],
            experiment_results['n_bets']
        )

        is_significant = p_value < self.MIN_P_VALUE
        effect_size = self._calculate_effect_size(
            baseline_results['roi'],
            experiment_results['roi']
        )

        roi_lift = experiment_results['roi'] - baseline_results['roi']
        accuracy_lift = experiment_results['accuracy'] - baseline_results['accuracy']

        # Step 6: Record results
        experiment_id = self.feature_registry.record_experiment(
            feature_id=feature_id,
            baseline_model_id=baseline_model.model_id,
            experiment_model_id=None,  # Would be populated in real implementation
            baseline_accuracy=baseline_results['accuracy'],
            experiment_accuracy=experiment_results['accuracy'],
            baseline_roi=baseline_results['roi'],
            experiment_roi=experiment_results['roi'],
            p_value=p_value,
            is_significant=is_significant,
            effect_size=effect_size,
            feature_importance_rank=experiment_results.get('importance_rank'),
            feature_importance_gain=experiment_results.get('importance_gain')
        )

        # Make decision
        decision, reason = self._make_decision(
            roi_lift, p_value, is_significant,
            experiment_results['n_bets']
        )

        if decision == "accept":
            self.feature_registry.accept_feature(feature_id, reason)
        elif decision == "reject":
            self.feature_registry.reject_feature(feature_id, reason)

        result = FeatureExperimentResult(
            experiment_id=experiment_id,
            feature_id=feature_id,
            feature_name=feature_name,
            experiment_date=pd.Timestamp.now(),
            baseline_accuracy=baseline_results['accuracy'],
            experiment_accuracy=experiment_results['accuracy'],
            accuracy_lift=accuracy_lift,
            baseline_roi=baseline_results['roi'],
            experiment_roi=experiment_results['roi'],
            roi_lift=roi_lift,
            p_value=p_value,
            is_significant=is_significant,
            effect_size=effect_size,
            feature_importance_rank=experiment_results.get('importance_rank'),
            feature_importance_gain=experiment_results.get('importance_gain'),
            decision=decision,
            decision_reason=reason
        )

        logger.info(f"Feature test complete: {decision} - {reason}")
        return result

    def batch_test_features(
        self,
        feature_names: Optional[List[str]] = None
    ) -> List[FeatureExperimentResult]:
        """
        Test multiple features.

        Args:
            feature_names: List of feature names (if None, tests all pending)

        Returns:
            List of FeatureExperimentResult
        """
        if feature_names is None:
            # Get pending features from registry
            pending = self.feature_registry.get_pending_features()
            feature_names = [f.feature_name for f in pending]

        logger.info(f"Batch testing {len(feature_names)} features")

        results = []
        for feature_name in feature_names:
            try:
                result = self.test_feature(feature_name)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to test feature {feature_name}: {e}")

        return results

    def _run_comparison(
        self,
        baseline_model,
        feature_name: str
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Run comparison between baseline and experiment.

        This is a simplified placeholder. In real implementation, would:
        1. Load baseline model
        2. Train new model with feature added to GameFeatureBuilder
        3. Run rigorous backtests on both
        4. Return detailed metrics

        Returns:
            Tuple of (baseline_results, experiment_results)
        """
        # Simplified placeholder
        logger.info("Running comparison (simplified placeholder)")

        # In real implementation, would actually train models and run backtests
        baseline_results = {
            'accuracy': 0.63,
            'roi': 0.05,
            'n_bets': 150,
            'sharpe': 0.8
        }

        # Simulate small improvement
        experiment_results = {
            'accuracy': 0.635,
            'roi': 0.058,
            'n_bets': 150,
            'sharpe': 0.85,
            'importance_rank': 15,
            'importance_gain': 0.025
        }

        return baseline_results, experiment_results

    def _calculate_p_value(
        self,
        baseline_roi: float,
        experiment_roi: float,
        n_baseline: int,
        n_experiment: int
    ) -> float:
        """Calculate p-value for ROI difference."""
        from scipy.stats import ttest_ind_from_stats

        # Simplified - assumes normal distribution
        # In real implementation, would use bootstrap or permutation test

        # Estimate standard deviations (simplified)
        baseline_std = 0.3  # Typical betting outcome std
        experiment_std = 0.3

        if n_baseline < 10 or n_experiment < 10:
            return 1.0

        try:
            _, p_value = ttest_ind_from_stats(
                mean1=experiment_roi,
                std1=experiment_std,
                nobs1=n_experiment,
                mean2=baseline_roi,
                std2=baseline_std,
                nobs2=n_baseline
            )
            return p_value / 2  # One-tailed test
        except:
            return 1.0

    def _calculate_effect_size(
        self,
        baseline_roi: float,
        experiment_roi: float
    ) -> float:
        """Calculate Cohen's d effect size."""
        pooled_std = 0.3  # Simplified
        return (experiment_roi - baseline_roi) / pooled_std

    def _make_decision(
        self,
        roi_lift: float,
        p_value: float,
        is_significant: bool,
        n_bets: int
    ) -> tuple[str, str]:
        """
        Make acceptance decision.

        Returns:
            Tuple of (decision, reason)
        """
        if n_bets < self.MIN_BETS:
            return ("needs_more_testing",
                    f"Only {n_bets} bets, need {self.MIN_BETS} minimum")

        if not is_significant:
            return ("reject",
                    f"Not statistically significant (p={p_value:.4f})")

        if roi_lift < self.MIN_ROI_LIFT:
            return ("reject",
                    f"ROI lift too small ({roi_lift:.2%} < {self.MIN_ROI_LIFT:.2%})")

        return ("accept",
                f"Significant improvement: +{roi_lift:.2%} ROI (p={p_value:.4f})")
