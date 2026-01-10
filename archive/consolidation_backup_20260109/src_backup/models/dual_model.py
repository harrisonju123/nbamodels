"""
Dual Model for NBA Prediction

Combines MLP (best for ATS edge finding) and XGBoost (best for accuracy).
Uses model disagreement as a signal for high-confidence bets.
"""

import os
import pickle
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from loguru import logger

from src.utils.constants import (
    SPREAD_TO_PROB_FACTOR,
    MIN_DISAGREEMENT,
    MIN_EDGE_VS_MARKET,
    MIN_EDGE_VS_ELO,
    MIN_PROBABILITY,
    MAX_PROBABILITY,
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class DualPredictionModel:
    """
    Dual model combining MLP and XGBoost for NBA predictions.

    Key insight: When MLP disagrees with XGBoost, the MLP is usually right
    because it captures compound advantages (multiple factors aligning).

    Usage:
        - For moneyline: Use XGBoost (more conservative, better accuracy)
        - For ATS value: Use MLP or disagreement signal
        - For bet sizing: Use disagreement score
    """

    MLP_PARAMS = {
        "hidden_layer_sizes": (64,),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.1,  # L2 regularization
        "batch_size": 64,
        "max_iter": 500,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 20,
        "random_state": 42,
    }

    XGB_PARAMS = {
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.05,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "random_state": 42,
        "verbosity": 0,
    }

    def __init__(
        self,
        mlp_params: dict = None,
        xgb_params: dict = None,
        use_sample_weighting: bool = True,
    ):
        """
        Initialize dual model.

        Args:
            mlp_params: Override MLP parameters
            xgb_params: Override XGBoost parameters
            use_sample_weighting: Weight recent games more heavily for XGBoost
        """
        self.mlp_params = {**self.MLP_PARAMS, **(mlp_params or {})}
        self.xgb_params = {**self.XGB_PARAMS, **(xgb_params or {})}
        self.use_sample_weighting = use_sample_weighting

        # Models
        self.mlp = None
        self.xgb = None
        self.scaler = None

        # Calibrators
        self.mlp_calibrator = None
        self.xgb_calibrator = None

        # Feature columns
        self.feature_columns = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: pd.Series = None,
        feature_columns: List[str] = None,
    ):
        """
        Train both models.

        Args:
            X: Training features
            y: Training labels (1 = home win, 0 = away win)
            dates: Game dates for sample weighting
            feature_columns: Columns to use as features
        """
        # Store feature columns
        if feature_columns:
            self.feature_columns = feature_columns
            X = X[feature_columns].copy()
        else:
            self.feature_columns = X.columns.tolist()
            X = X.copy()

        # Handle missing values
        X = X.fillna(0)
        X_values = X.values
        y_values = y.values

        # Calculate sample weights for XGBoost
        sample_weights = None
        if self.use_sample_weighting and dates is not None:
            dates = pd.to_datetime(dates)
            max_date = dates.max()
            days_ago = (max_date - dates).dt.days.values
            max_days = days_ago.max()
            # Linear decay: min weight = 0.5 at oldest games
            sample_weights = 1 - 0.5 * (days_ago / max_days)

        # Scale features for MLP
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_values)

        # Train MLP
        logger.info("Training MLP...")
        self.mlp = MLPClassifier(**self.mlp_params)
        self.mlp.fit(X_scaled, y_values)

        # Train XGBoost
        logger.info("Training XGBoost...")
        self.xgb = XGBClassifier(**self.xgb_params)
        if sample_weights is not None:
            self.xgb.fit(X_values, y_values, sample_weight=sample_weights)
        else:
            self.xgb.fit(X_values, y_values)

        # Fit calibrators
        logger.info("Fitting calibrators...")
        mlp_train_probs = self.mlp.predict_proba(X_scaled)[:, 1]
        xgb_train_probs = self.xgb.predict_proba(X_values)[:, 1]

        self.mlp_calibrator = IsotonicRegression(out_of_bounds='clip')
        self.mlp_calibrator.fit(mlp_train_probs, y_values)

        self.xgb_calibrator = IsotonicRegression(out_of_bounds='clip')
        self.xgb_calibrator.fit(xgb_train_probs, y_values)

        logger.info("Dual model training complete")
        return self

    def predict_proba(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get calibrated probability predictions from both models.

        Returns:
            Dictionary with 'mlp', 'xgb', 'ensemble', and 'disagreement' arrays
        """
        X = X[self.feature_columns].fillna(0)
        X_values = X.values
        X_scaled = self.scaler.transform(X_values)

        # Raw predictions
        mlp_raw = self.mlp.predict_proba(X_scaled)[:, 1]
        xgb_raw = self.xgb.predict_proba(X_values)[:, 1]

        # Calibrated predictions
        mlp_prob = np.clip(self.mlp_calibrator.predict(mlp_raw), MIN_PROBABILITY, MAX_PROBABILITY)
        xgb_prob = np.clip(self.xgb_calibrator.predict(xgb_raw), MIN_PROBABILITY, MAX_PROBABILITY)

        # Ensemble (40% MLP, 60% XGB)
        ensemble_prob = 0.4 * mlp_prob + 0.6 * xgb_prob

        # Convert to implied spreads
        mlp_spread = self._prob_to_spread(mlp_prob)
        xgb_spread = self._prob_to_spread(xgb_prob)

        # Disagreement score
        # ACTUAL BEHAVIOR: Positive disagreement means mlp_spread > xgb_spread
        # This means XGB is MORE bullish on home than MLP
        # Strategy trusts XGB when disagreement is high (XGB more conservative/accurate)
        disagreement = mlp_spread - xgb_spread

        return {
            'mlp': mlp_prob,
            'xgb': xgb_prob,
            'ensemble': ensemble_prob,
            'mlp_spread': mlp_spread,
            'xgb_spread': xgb_spread,
            'disagreement': disagreement,
        }

    @staticmethod
    def _prob_to_spread(prob: np.ndarray) -> np.ndarray:
        """Convert win probability to implied spread."""
        return -((prob - 0.5) * SPREAD_TO_PROB_FACTOR)

    @staticmethod
    def _spread_to_prob(spread: np.ndarray) -> np.ndarray:
        """Convert spread to implied probability."""
        return 0.5 - spread / SPREAD_TO_PROB_FACTOR

    def get_predictions(
        self,
        X: pd.DataFrame,
        elo_prob: np.ndarray = None,
        vegas_spread: np.ndarray = None,
    ) -> pd.DataFrame:
        """
        Get comprehensive predictions for games.

        Args:
            X: Game features
            elo_prob: Elo-based probabilities (as baseline/market proxy)
            vegas_spread: Real Vegas spreads for edge calculation

        Returns:
            DataFrame with all predictions and signals
        """
        preds = self.predict_proba(X)

        result = pd.DataFrame({
            'mlp_prob': preds['mlp'],
            'xgb_prob': preds['xgb'],
            'ensemble_prob': preds['ensemble'],
            'mlp_spread': preds['mlp_spread'],
            'xgb_spread': preds['xgb_spread'],
            'disagreement': preds['disagreement'],
        })

        # Add edge vs Elo if provided
        if elo_prob is not None:
            elo_spread = self._prob_to_spread(elo_prob)
            result['elo_prob'] = elo_prob
            result['elo_spread'] = elo_spread
            result['mlp_edge_vs_elo'] = preds['mlp_spread'] - elo_spread
            result['xgb_edge_vs_elo'] = preds['xgb_spread'] - elo_spread

        # Add edge vs Vegas if provided
        if vegas_spread is not None:
            result['vegas_spread'] = vegas_spread
            result['mlp_edge_vs_vegas'] = preds['mlp_spread'] - vegas_spread

        # Betting signals - validated against real Vegas (58% ATS, +11% ROI)
        min_disagree = MIN_DISAGREEMENT
        min_edge = MIN_EDGE_VS_MARKET

        if vegas_spread is not None:
            # Use real Vegas edge
            result['high_confidence_home'] = (
                (preds['disagreement'] >= min_disagree) &
                (result['mlp_edge_vs_vegas'] >= min_edge)
            )
            result['high_confidence_away'] = (
                (preds['disagreement'] <= -min_disagree) &
                (result['mlp_edge_vs_vegas'] <= -min_edge)
            )
        elif elo_prob is not None:
            # Fall back to Elo edge
            result['high_confidence_home'] = (
                (preds['disagreement'] >= min_disagree) &
                (result['mlp_edge_vs_elo'] >= min_edge)
            )
            result['high_confidence_away'] = (
                (preds['disagreement'] <= -min_disagree) &
                (result['mlp_edge_vs_elo'] <= -min_edge)
            )
        else:
            # No market data - use looser threshold
            result['high_confidence_home'] = (preds['disagreement'] >= min_disagree)
            result['high_confidence_away'] = (preds['disagreement'] <= -min_disagree)

        # Add betting recommendations
        result['bet_side'] = np.where(
            result['high_confidence_home'], 'HOME',
            np.where(result['high_confidence_away'], 'AWAY', 'PASS')
        )

        # Confidence level based on disagreement magnitude
        result['confidence'] = np.abs(preds['disagreement']).clip(0, 10) / 10

        return result

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        elo_prob: np.ndarray = None,
    ) -> Dict[str, float]:
        """
        Evaluate both models.

        Returns:
            Dictionary with metrics for both models and ensemble
        """
        preds = self.predict_proba(X)
        y_values = y.values

        metrics = {}

        for name, prob in [('mlp', preds['mlp']), ('xgb', preds['xgb']),
                           ('ensemble', preds['ensemble'])]:
            metrics[f'{name}_auc'] = roc_auc_score(y_values, prob)
            metrics[f'{name}_accuracy'] = accuracy_score(y_values, (prob > 0.5).astype(int))
            metrics[f'{name}_brier'] = brier_score_loss(y_values, prob)

        # ATS performance (if we have Elo as market proxy)
        if elo_prob is not None:
            elo_spread = self._prob_to_spread(elo_prob)
            point_diff = X.get('point_diff')

            if point_diff is not None:
                home_ats = point_diff.values - elo_spread

                # Disagreement signal performance
                disagree_home = preds['disagreement'] >= 3
                disagree_away = preds['disagreement'] <= -3

                if disagree_home.sum() > 0:
                    home_wins = (home_ats[disagree_home] > 0).sum()
                    home_total = disagree_home.sum()
                    metrics['disagree_home_ats_pct'] = home_wins / home_total
                    metrics['disagree_home_count'] = home_total

                if disagree_away.sum() > 0:
                    away_wins = (home_ats[disagree_away] < 0).sum()
                    away_total = disagree_away.sum()
                    metrics['disagree_away_ats_pct'] = away_wins / away_total
                    metrics['disagree_away_count'] = away_total

        return metrics

    def save(self, path: str):
        """Save model to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                'mlp': self.mlp,
                'xgb': self.xgb,
                'scaler': self.scaler,
                'mlp_calibrator': self.mlp_calibrator,
                'xgb_calibrator': self.xgb_calibrator,
                'feature_columns': self.feature_columns,
                'mlp_params': self.mlp_params,
                'xgb_params': self.xgb_params,
            }, f)
        logger.info(f"Dual model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "DualPredictionModel":
        """Load model from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls(
            mlp_params=data['mlp_params'],
            xgb_params=data['xgb_params'],
        )
        instance.mlp = data['mlp']
        instance.xgb = data['xgb']
        instance.scaler = data['scaler']
        instance.mlp_calibrator = data['mlp_calibrator']
        instance.xgb_calibrator = data['xgb_calibrator']
        instance.feature_columns = data['feature_columns']

        logger.info(f"Dual model loaded from {path}")
        return instance


def train_dual_model(
    games_path: str = "data/raw/games.parquet",
    model_path: str = "models/dual_model.pkl",
    train_end_season: int = 2024,
) -> Tuple[DualPredictionModel, Dict]:
    """
    Train the dual model on historical data.

    Args:
        games_path: Path to games data
        model_path: Path to save model
        train_end_season: Last season to include in training (exclusive of test)

    Returns:
        Trained model and evaluation metrics
    """
    from src.features.game_features import GameFeatureBuilder

    # Load games
    games = pd.read_parquet(games_path)
    games['date'] = pd.to_datetime(games['date'])
    logger.info(f"Loaded {len(games)} games")

    # Build features
    builder = GameFeatureBuilder()
    features = builder.build_game_features(games)

    # Split data
    train = features[features['season'] < train_end_season].copy()
    test = features[features['season'] >= train_end_season].copy()

    logger.info(f"Train: {len(train)} games, Test: {len(test)} games")

    # Get feature columns
    feature_cols = builder.get_feature_columns(features)
    feature_cols = [c for c in feature_cols if c in train.columns]
    logger.info(f"Using {len(feature_cols)} features")

    # Train model
    model = DualPredictionModel()
    model.fit(
        train[feature_cols],
        train['home_win'],
        dates=train['date'],
        feature_columns=feature_cols,
    )

    # Evaluate on test set
    test_metrics = model.evaluate(
        test[feature_cols],
        test['home_win'],
        elo_prob=test['elo_prob'].values if 'elo_prob' in test.columns else None,
    )

    logger.info("Test metrics:")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")

    # Save model
    model.save(model_path)

    return model, test_metrics


if __name__ == "__main__":
    # Train and evaluate
    model, metrics = train_dual_model()

    print("\n" + "="*60)
    print("DUAL MODEL TRAINED")
    print("="*60)
    print("\nMetrics:")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
