"""
Feature Drift Detection

Monitors changes in feature distributions and importance over time.
Uses Population Stability Index (PSI) and importance comparison.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class DriftResult:
    """Result of drift detection for a single feature."""
    feature: str
    psi: float
    drift_level: str  # 'none', 'moderate', 'significant'
    baseline_mean: float
    current_mean: float
    baseline_std: float
    current_std: float


class FeatureDriftMonitor:
    """
    Monitor changes in feature distributions and model importance.

    Uses Population Stability Index (PSI) to detect distribution shifts:
    - PSI < 0.1: No significant change
    - PSI 0.1 - 0.2: Moderate change, monitor closely
    - PSI > 0.2: Significant change, investigate

    Also tracks feature importance drift to detect when the model's
    reliance on features changes.
    """

    PSI_THRESHOLDS = {
        'none': 0.1,
        'moderate': 0.2,
    }

    def __init__(
        self,
        baseline_path: str = None,
        n_bins: int = 10,
    ):
        """
        Initialize FeatureDriftMonitor.

        Args:
            baseline_path: Path to saved baseline distributions
            n_bins: Number of bins for PSI calculation
        """
        self.baseline_path = baseline_path or "data/monitoring/feature_baseline.json"
        self.n_bins = n_bins
        self.baseline_distributions: Dict[str, Dict] = {}
        self.baseline_importance: Optional[pd.DataFrame] = None
        self._load_baseline()

    def _load_baseline(self):
        """Load saved baseline if available."""
        if os.path.exists(self.baseline_path):
            try:
                with open(self.baseline_path, 'r') as f:
                    data = json.load(f)
                self.baseline_distributions = data.get('distributions', {})
                if 'importance' in data:
                    self.baseline_importance = pd.DataFrame(data['importance'])
                logger.info(f"Loaded baseline from {self.baseline_path}")
            except Exception as e:
                logger.warning(f"Could not load baseline: {e}")

    def save_baseline(
        self,
        features_df: pd.DataFrame,
        importance_df: pd.DataFrame = None,
    ):
        """
        Save current feature distributions as baseline.

        Args:
            features_df: DataFrame with feature values
            importance_df: DataFrame with feature importance scores
        """
        # Validate input
        if features_df is None or features_df.empty:
            logger.error("Cannot save baseline: features_df is empty or None")
            return

        distributions = {}

        for col in features_df.select_dtypes(include=[np.number]).columns:
            values = features_df[col].dropna()
            if len(values) < 10:
                continue

            # Store distribution statistics
            distributions[col] = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'quantiles': {
                    str(q): float(values.quantile(q))
                    for q in [0.1, 0.25, 0.5, 0.75, 0.9]
                },
                'n_samples': len(values),
                'histogram': self._compute_histogram(values),
            }

        data = {
            'distributions': distributions,
            'saved_at': datetime.now().isoformat(),
            'n_features': len(distributions),
        }

        if importance_df is not None:
            data['importance'] = importance_df.to_dict('records')

        os.makedirs(os.path.dirname(self.baseline_path), exist_ok=True)
        with open(self.baseline_path, 'w') as f:
            json.dump(data, f, indent=2)

        self.baseline_distributions = distributions
        self.baseline_importance = importance_df

        logger.info(f"Saved baseline for {len(distributions)} features to {self.baseline_path}")

    def _compute_histogram(self, values: pd.Series) -> Dict:
        """Compute histogram for PSI calculation."""
        counts, bin_edges = np.histogram(values, bins=self.n_bins)
        proportions = counts / counts.sum()

        return {
            'counts': counts.tolist(),
            'bin_edges': bin_edges.tolist(),
            'proportions': proportions.tolist(),
        }

    def calculate_psi(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
    ) -> float:
        """
        Calculate Population Stability Index.

        PSI = Σ (current_i - baseline_i) * ln(current_i / baseline_i)

        Args:
            baseline: Baseline distribution (proportions)
            current: Current distribution (proportions)

        Returns:
            PSI value
        """
        # Avoid division by zero
        epsilon = 1e-10
        baseline = np.array(baseline) + epsilon
        current = np.array(current) + epsilon

        # Normalize
        baseline = baseline / baseline.sum()
        current = current / current.sum()

        # Calculate PSI
        psi = np.sum((current - baseline) * np.log(current / baseline))

        return psi

    def detect_distribution_drift(
        self,
        current_df: pd.DataFrame,
        features: List[str] = None,
    ) -> Dict[str, DriftResult]:
        """
        Detect distribution drift for features.

        Args:
            current_df: Current feature values
            features: List of features to check (default: all)

        Returns:
            Dictionary mapping feature names to DriftResult
        """
        if not self.baseline_distributions:
            logger.warning("No baseline set. Call save_baseline() first.")
            return {}

        features = features or list(self.baseline_distributions.keys())
        results = {}

        for feature in features:
            if feature not in self.baseline_distributions:
                continue
            if feature not in current_df.columns:
                continue

            baseline = self.baseline_distributions[feature]
            current_values = current_df[feature].dropna()

            if len(current_values) < 10:
                continue

            # Compute PSI using baseline bins
            bin_edges = baseline['histogram']['bin_edges']
            baseline_props = baseline['histogram']['proportions']

            # Bin current values using baseline edges
            current_counts, _ = np.histogram(current_values, bins=bin_edges)

            # Check for division by zero - if all counts are zero, skip this feature
            current_sum = current_counts.sum()
            if current_sum == 0:
                logger.warning(f"Feature '{feature}' has zero counts in histogram, skipping PSI calculation")
                continue

            current_props = current_counts / current_sum

            psi = self.calculate_psi(baseline_props, current_props)

            # Determine drift level
            if psi < self.PSI_THRESHOLDS['none']:
                drift_level = 'none'
            elif psi < self.PSI_THRESHOLDS['moderate']:
                drift_level = 'moderate'
            else:
                drift_level = 'significant'

            results[feature] = DriftResult(
                feature=feature,
                psi=psi,
                drift_level=drift_level,
                baseline_mean=baseline['mean'],
                current_mean=float(current_values.mean()),
                baseline_std=baseline['std'],
                current_std=float(current_values.std()),
            )

        return results

    def detect_importance_drift(
        self,
        current_importance: pd.DataFrame,
        threshold: float = 0.3,
    ) -> List[Dict]:
        """
        Detect changes in feature importance.

        Args:
            current_importance: DataFrame with current importance
                Must have 'feature' and 'importance' columns
            threshold: Relative change threshold to flag (0.3 = 30%)

        Returns:
            List of features with significant importance changes
        """
        if self.baseline_importance is None:
            logger.warning("No baseline importance set.")
            return []

        # Normalize importance to sum to 1
        baseline = self.baseline_importance.copy()
        baseline['importance_norm'] = baseline['importance'] / baseline['importance'].sum()

        current = current_importance.copy()
        current['importance_norm'] = current['importance'] / current['importance'].sum()

        # Merge
        merged = baseline.merge(
            current,
            on='feature',
            suffixes=('_baseline', '_current')
        )

        alerts = []
        for _, row in merged.iterrows():
            baseline_imp = row['importance_norm_baseline']
            current_imp = row['importance_norm_current']

            # Use threshold to avoid near-zero division issues
            if baseline_imp < 1e-6:
                continue

            relative_change = (current_imp - baseline_imp) / baseline_imp

            if abs(relative_change) > threshold:
                alerts.append({
                    'feature': row['feature'],
                    'baseline_importance': baseline_imp,
                    'current_importance': current_imp,
                    'relative_change': relative_change,
                    'direction': 'increased' if relative_change > 0 else 'decreased',
                })

        # Sort by absolute change
        alerts.sort(key=lambda x: abs(x['relative_change']), reverse=True)

        return alerts

    def get_drift_summary(
        self,
        current_df: pd.DataFrame,
        current_importance: pd.DataFrame = None,
    ) -> Dict:
        """
        Generate comprehensive drift summary.

        Returns:
            Dictionary with drift analysis results
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'distribution_drift': {},
            'importance_drift': [],
            'alerts': [],
        }

        # Distribution drift
        dist_results = self.detect_distribution_drift(current_df)

        significant_count = 0
        moderate_count = 0

        for feature, result in dist_results.items():
            summary['distribution_drift'][feature] = {
                'psi': result.psi,
                'drift_level': result.drift_level,
                'mean_shift': result.current_mean - result.baseline_mean,
            }

            if result.drift_level == 'significant':
                significant_count += 1
                summary['alerts'].append({
                    'type': 'distribution_drift',
                    'severity': 'warning',
                    'feature': feature,
                    'psi': result.psi,
                    'message': f"Feature '{feature}' has significant distribution drift (PSI={result.psi:.3f})"
                })
            elif result.drift_level == 'moderate':
                moderate_count += 1

        summary['distribution_summary'] = {
            'features_checked': len(dist_results),
            'significant_drift': significant_count,
            'moderate_drift': moderate_count,
            'no_drift': len(dist_results) - significant_count - moderate_count,
        }

        # Importance drift
        if current_importance is not None:
            imp_drift = self.detect_importance_drift(current_importance)
            summary['importance_drift'] = imp_drift

            for item in imp_drift[:5]:  # Top 5 changes
                summary['alerts'].append({
                    'type': 'importance_drift',
                    'severity': 'info',
                    'feature': item['feature'],
                    'change': item['relative_change'],
                    'message': f"Feature '{item['feature']}' importance {item['direction']} by {abs(item['relative_change']):.1%}"
                })

        return summary

    def get_drifted_features(
        self,
        current_df: pd.DataFrame,
        min_level: str = 'moderate',
    ) -> List[str]:
        """
        Get list of features with drift above specified level.

        Args:
            current_df: Current feature values
            min_level: Minimum drift level ('moderate' or 'significant')

        Returns:
            List of feature names with drift
        """
        results = self.detect_distribution_drift(current_df)

        levels = ['moderate', 'significant'] if min_level == 'moderate' else ['significant']

        return [
            feature
            for feature, result in results.items()
            if result.drift_level in levels
        ]


def compute_psi_for_features(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: List[str] = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Convenience function to compute PSI for multiple features.

    Args:
        baseline_df: Baseline feature values
        current_df: Current feature values
        features: Features to check (default: all numeric)
        n_bins: Number of bins for PSI

    Returns:
        DataFrame with PSI for each feature
    """
    monitor = FeatureDriftMonitor(n_bins=n_bins)

    # Temporarily set baseline from provided data
    for col in baseline_df.select_dtypes(include=[np.number]).columns:
        values = baseline_df[col].dropna()
        if len(values) >= 10:
            monitor.baseline_distributions[col] = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'histogram': monitor._compute_histogram(values),
            }

    results = monitor.detect_distribution_drift(current_df, features)

    return pd.DataFrame([
        {
            'feature': r.feature,
            'psi': r.psi,
            'drift_level': r.drift_level,
            'baseline_mean': r.baseline_mean,
            'current_mean': r.current_mean,
        }
        for r in results.values()
    ]).sort_values('psi', ascending=False)
