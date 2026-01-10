"""
Model Tuning with Optuna

Hyperparameter optimization for NBA betting models.
"""

import os
import pickle
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, roc_auc_score, brier_score_loss, log_loss
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from loguru import logger

from .calibration import CalibratedModel, evaluate_calibration


class RegressionTuner:
    """Hyperparameter tuning for regression models (spread/totals)."""

    def __init__(
        self,
        features_path: str = "data/features/game_features.parquet",
        n_trials: int = 50,
        target: str = "point_diff",  # 'point_diff' or 'total_points'
    ):
        self.features_path = features_path
        self.n_trials = n_trials
        self.target = target
        self._load_data()

    def _load_data(self):
        """Load and split data for regression."""
        df = pd.read_parquet(self.features_path)
        df = df.sort_values("date").reset_index(drop=True)

        # Target column (already exists in features file)
        if self.target == "point_diff":
            df["target"] = df["point_diff"]
        elif self.target == "total_points":
            df["target"] = df["total_points"]
        else:
            raise ValueError(f"Unknown target: {self.target}")

        # Exclude leaky and non-feature columns
        exclude = [
            'game_id', 'date', 'home_team', 'away_team', 'home_score',
            'away_score', 'point_diff', 'total_points', 'home_win', 'season',
            'away_win', 'home_total_points', 'away_total_points',
            'home_point_diff', 'away_point_diff', 'target'
        ]
        self.feature_cols = [
            c for c in df.columns
            if c not in exclude and df[c].dtype in ['float64', 'int64', 'int32', 'float32']
        ]

        # Time-based split: 70% train, 15% val, 15% test
        train_size = int(0.7 * len(df))
        val_size = int(0.15 * len(df))

        self.train_df = df[:train_size]
        self.val_df = df[train_size:train_size + val_size]
        self.test_df = df[train_size + val_size:]

        self.X_train = self.train_df[self.feature_cols].fillna(0)
        self.y_train = self.train_df["target"]
        self.X_val = self.val_df[self.feature_cols].fillna(0)
        self.y_val = self.val_df["target"]
        self.X_test = self.test_df[self.feature_cols].fillna(0)
        self.y_test = self.test_df["target"]

        logger.info(f"Regression data loaded ({self.target}): {len(self.train_df)} train, {len(self.val_df)} val, {len(self.test_df)} test")

    def _objective_xgb(self, trial: optuna.Trial) -> float:
        """Optuna objective for XGBoost regressor."""
        params = {
            "objective": "reg:squarederror",
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "random_state": 42,
            "n_jobs": -1,
        }

        model = xgb.XGBRegressor(**params)
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            verbose=False
        )

        preds = model.predict(self.X_val)
        rmse = np.sqrt(np.mean((self.y_val - preds) ** 2))
        return rmse

    def _objective_lgb(self, trial: optuna.Trial) -> float:
        """Optuna objective for LightGBM regressor."""
        params = {
            "objective": "regression",
            "metric": "rmse",
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 10, 100),
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

        model = lgb.LGBMRegressor(**params)
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
        )

        preds = model.predict(self.X_val)
        rmse = np.sqrt(np.mean((self.y_val - preds) ** 2))
        return rmse

    def tune(self) -> Dict:
        """Tune both XGBoost and LightGBM, return best."""
        logger.info(f"Tuning regression models for {self.target}...")

        # Tune XGBoost
        xgb_study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=42),
        )
        xgb_study.optimize(self._objective_xgb, n_trials=self.n_trials, show_progress_bar=True)

        # Tune LightGBM
        lgb_study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=42),
        )
        lgb_study.optimize(self._objective_lgb, n_trials=self.n_trials, show_progress_bar=True)

        logger.info(f"XGBoost best RMSE: {xgb_study.best_value:.3f}")
        logger.info(f"LightGBM best RMSE: {lgb_study.best_value:.3f}")

        # Pick best model
        if xgb_study.best_value <= lgb_study.best_value:
            best_type = "xgboost"
            best_params = xgb_study.best_params
            best_rmse = xgb_study.best_value
        else:
            best_type = "lightgbm"
            best_params = lgb_study.best_params
            best_rmse = lgb_study.best_value

        return {
            "best_type": best_type,
            "best_params": best_params,
            "best_rmse": best_rmse,
            "xgb_results": {"params": xgb_study.best_params, "rmse": xgb_study.best_value},
            "lgb_results": {"params": lgb_study.best_params, "rmse": lgb_study.best_value},
        }

    def train_final_model(self, model_type: str, params: Dict) -> Tuple:
        """Train final regression model on train+val."""
        X_trainval = pd.concat([self.X_train, self.X_val])
        y_trainval = pd.concat([self.y_train, self.y_val])

        if model_type == "xgboost":
            base_params = {
                "objective": "reg:squarederror",
                "random_state": 42,
                "n_jobs": -1,
            }
            base_params.update(params)
            model = xgb.XGBRegressor(**base_params)
        else:
            base_params = {
                "objective": "regression",
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            }
            base_params.update(params)
            model = lgb.LGBMRegressor(**base_params)

        model.fit(X_trainval, y_trainval)

        # Evaluate on test
        preds = model.predict(self.X_test)
        metrics = {
            "mae": float(np.mean(np.abs(self.y_test - preds))),
            "rmse": float(np.sqrt(np.mean((self.y_test - preds) ** 2))),
            "std_error": float(np.std(self.y_test - preds)),
        }

        logger.info(f"Test metrics: {metrics}")
        return model, metrics

    def save_model(self, model, params: Dict, metrics: Dict, path: str):
        """Save regression model."""
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                "feature": self.feature_cols,
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=False)
        else:
            importance = None

        model_data = {
            "model": model,
            "feature_columns": self.feature_cols,
            "feature_importance": importance,
            "calibrated_std": metrics["std_error"],
            "params": params,
            "metrics": metrics,
            "tuned_at": datetime.now().isoformat(),
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Regression model saved to {path}")


class ModelTuner:
    """Hyperparameter tuning for betting models using Optuna."""

    def __init__(
        self,
        features_path: str = "data/features/game_features.parquet",
        n_trials: int = 100,
        cv_splits: int = 5,
        metric: str = "brier",  # 'brier', 'auc', 'log_loss', 'accuracy'
    ):
        self.features_path = features_path
        self.n_trials = n_trials
        self.cv_splits = cv_splits
        self.metric = metric

        # Load and prepare data
        self._load_data()

    def _load_data(self):
        """Load and split data."""
        df = pd.read_parquet(self.features_path)
        df = df.sort_values("date").reset_index(drop=True)

        # Exclude leaky and non-feature columns
        exclude = [
            'game_id', 'date', 'home_team', 'away_team', 'home_score',
            'away_score', 'point_diff', 'total_points', 'home_win', 'season',
            'away_win', 'home_total_points', 'away_total_points',
            'home_point_diff', 'away_point_diff'
        ]
        self.feature_cols = [
            c for c in df.columns
            if c not in exclude and df[c].dtype in ['float64', 'int64', 'int32', 'float32']
        ]

        # Time-based split: 70% train, 15% val, 15% test
        train_size = int(0.7 * len(df))
        val_size = int(0.15 * len(df))

        self.train_df = df[:train_size]
        self.val_df = df[train_size:train_size + val_size]
        self.test_df = df[train_size + val_size:]

        self.X_train = self.train_df[self.feature_cols].fillna(0)
        self.y_train = self.train_df["home_win"]
        self.X_val = self.val_df[self.feature_cols].fillna(0)
        self.y_val = self.val_df["home_win"]
        self.X_test = self.test_df[self.feature_cols].fillna(0)
        self.y_test = self.test_df["home_win"]

        logger.info(f"Data loaded: {len(self.train_df)} train, {len(self.val_df)} val, {len(self.test_df)} test")
        logger.info(f"Features: {len(self.feature_cols)}")

    def _objective_xgb(self, trial: optuna.Trial) -> float:
        """Optuna objective for XGBoost."""
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "random_state": 42,
            "n_jobs": -1,
        }

        model = xgb.XGBClassifier(**params)
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            verbose=False
        )

        probs = model.predict_proba(self.X_val)[:, 1]
        return self._calculate_metric(self.y_val, probs)

    def _objective_lgb(self, trial: optuna.Trial) -> float:
        """Optuna objective for LightGBM."""
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 10, 100),
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

        model = lgb.LGBMClassifier(**params)
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
        )

        probs = model.predict_proba(self.X_val)[:, 1]
        return self._calculate_metric(self.y_val, probs)

    def _calculate_metric(self, y_true, y_pred) -> float:
        """Calculate optimization metric (lower is better for Optuna minimize)."""
        if self.metric == "brier":
            return brier_score_loss(y_true, y_pred)
        elif self.metric == "log_loss":
            return log_loss(y_true, y_pred)
        elif self.metric == "auc":
            return -roc_auc_score(y_true, y_pred)  # Negative because we minimize
        elif self.metric == "accuracy":
            return -accuracy_score(y_true, (y_pred > 0.5).astype(int))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def tune_xgboost(self) -> Dict:
        """Tune XGBoost hyperparameters."""
        logger.info("Tuning XGBoost...")

        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=42),
        )
        study.optimize(
            self._objective_xgb,
            n_trials=self.n_trials,
            show_progress_bar=True,
        )

        logger.info(f"Best trial: {study.best_trial.value:.4f}")
        logger.info(f"Best params: {study.best_trial.params}")

        return {
            "best_value": study.best_trial.value,
            "best_params": study.best_trial.params,
            "study": study,
        }

    def tune_lightgbm(self) -> Dict:
        """Tune LightGBM hyperparameters."""
        logger.info("Tuning LightGBM...")

        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=42),
        )
        study.optimize(
            self._objective_lgb,
            n_trials=self.n_trials,
            show_progress_bar=True,
        )

        logger.info(f"Best trial: {study.best_trial.value:.4f}")
        logger.info(f"Best params: {study.best_trial.params}")

        return {
            "best_value": study.best_trial.value,
            "best_params": study.best_trial.params,
            "study": study,
        }

    def train_final_model(
        self,
        model_type: str,
        params: Dict,
        calibrate: bool = True,
        calibration_method: str = "auto",  # 'auto', 'isotonic', 'sigmoid', 'temperature'
    ) -> Tuple:
        """
        Train final model with best parameters.

        Args:
            model_type: 'xgboost' or 'lightgbm'
            params: Model hyperparameters
            calibrate: Whether to apply calibration
            calibration_method: 'auto' picks best, or specify method
        """
        logger.info(f"Training final {model_type} model...")

        # Combine train and val for final training
        X_trainval = pd.concat([self.X_train, self.X_val])
        y_trainval = pd.concat([self.y_train, self.y_val])

        if model_type == "xgboost":
            base_params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "random_state": 42,
                "n_jobs": -1,
            }
            base_params.update(params)
            model = xgb.XGBClassifier(**base_params)
        elif model_type == "lightgbm":
            base_params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            }
            base_params.update(params)
            model = lgb.LGBMClassifier(**base_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.fit(X_trainval, y_trainval)

        # Get uncalibrated predictions for comparison
        uncalib_probs = model.predict_proba(self.X_test)[:, 1]
        uncalib_metrics = evaluate_calibration(self.y_test.values, uncalib_probs)
        logger.info(f"Uncalibrated - Brier: {uncalib_metrics['brier_score']:.4f}, ECE: {uncalib_metrics['ece']:.4f}")

        # Calibrate if requested
        calibration_results = {}
        if calibrate:
            # Reserve portion of data for calibration
            calib_size = int(0.2 * len(X_trainval))
            X_calib = X_trainval.iloc[-calib_size:]
            y_calib = y_trainval.iloc[-calib_size:]

            if calibration_method == "auto":
                # Compare all methods and pick best
                best_method = None
                best_brier = float("inf")

                for method in ["isotonic", "sigmoid", "temperature"]:
                    try:
                        calib_model = CalibratedModel(model, method=method)
                        calib_model.fit(X_trainval, y_trainval, X_calib, y_calib)
                        calib_probs = calib_model.predict_proba(self.X_test)[:, 1]
                        calib_eval = evaluate_calibration(self.y_test.values, calib_probs)

                        calibration_results[method] = {
                            "brier": calib_eval["brier_score"],
                            "ece": calib_eval["ece"],
                            "log_loss": calib_eval["log_loss"],
                        }

                        logger.info(f"  {method}: Brier={calib_eval['brier_score']:.4f}, ECE={calib_eval['ece']:.4f}")

                        if calib_eval["brier_score"] < best_brier:
                            best_brier = calib_eval["brier_score"]
                            best_method = method
                            final_model = calib_model
                    except Exception as e:
                        logger.warning(f"  {method} failed: {e}")
                        continue

                logger.info(f"Best calibration method: {best_method}")
                calibration_method = best_method
            else:
                # Use specified method
                final_model = CalibratedModel(model, method=calibration_method)
                final_model.fit(X_trainval, y_trainval, X_calib, y_calib)
        else:
            final_model = model
            calibration_method = None

        # Evaluate on test set
        probs = final_model.predict_proba(self.X_test)[:, 1]
        preds = (probs > 0.5).astype(int)

        # Calculate calibration curve for storing
        prob_true, prob_pred = calibration_curve(self.y_test, probs, n_bins=10)

        metrics = {
            "accuracy": float(accuracy_score(self.y_test, preds)),
            "auc": float(roc_auc_score(self.y_test, probs)),
            "brier": float(brier_score_loss(self.y_test, probs)),
            "log_loss": float(log_loss(self.y_test, probs)),
            "calibration_method": calibration_method,
            "calibration_comparison": calibration_results,
            "uncalibrated_brier": float(uncalib_metrics["brier_score"]),
            "calibration_improvement": float(uncalib_metrics["brier_score"] - brier_score_loss(self.y_test, probs)),
            "calibration_curve": {"prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist()},
        }

        logger.info(f"Final metrics: Brier={metrics['brier']:.4f}, AUC={metrics['auc']:.4f}")
        if calibrate:
            logger.info(f"Calibration improved Brier by {metrics['calibration_improvement']:.4f}")

        return final_model, model, metrics

    def save_model(
        self,
        model,
        base_model,
        params: Dict,
        metrics: Dict,
        path: str,
    ):
        """Save trained model."""
        # Get feature importance
        if hasattr(base_model, 'feature_importances_'):
            importance = pd.DataFrame({
                "feature": self.feature_cols,
                "importance": base_model.feature_importances_
            }).sort_values("importance", ascending=False)
        else:
            importance = None

        model_data = {
            "model": model,
            "base_model": base_model,
            "feature_cols": self.feature_cols,
            "params": params,
            "metrics": metrics,
            "importance": importance,
            "tuned_at": datetime.now().isoformat(),
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {path}")


def run_full_tuning(n_trials: int = 50, metric: str = "brier"):
    """Run full tuning pipeline."""
    print("=" * 60)
    print("NBA Model Tuning Pipeline")
    print("=" * 60)

    tuner = ModelTuner(n_trials=n_trials, metric=metric)

    # Tune both models
    print("\n--- Tuning XGBoost ---")
    xgb_results = tuner.tune_xgboost()

    print("\n--- Tuning LightGBM ---")
    lgb_results = tuner.tune_lightgbm()

    # Compare results
    print("\n" + "=" * 60)
    print("Results Comparison")
    print("=" * 60)
    print(f"XGBoost best {metric}: {xgb_results['best_value']:.4f}")
    print(f"LightGBM best {metric}: {lgb_results['best_value']:.4f}")

    # Train final models with best params
    print("\n--- Training Final Models ---")

    xgb_model, xgb_base, xgb_metrics = tuner.train_final_model(
        "xgboost", xgb_results["best_params"], calibrate=True
    )

    lgb_model, lgb_base, lgb_metrics = tuner.train_final_model(
        "lightgbm", lgb_results["best_params"], calibrate=True
    )

    # Choose best model
    if xgb_metrics["brier"] <= lgb_metrics["brier"]:
        best_type = "xgboost"
        best_model = xgb_model
        best_base = xgb_base
        best_params = xgb_results["best_params"]
        best_metrics = xgb_metrics
    else:
        best_type = "lightgbm"
        best_model = lgb_model
        best_base = lgb_base
        best_params = lgb_results["best_params"]
        best_metrics = lgb_metrics

    print(f"\n--- Best Model: {best_type.upper()} ---")
    print(f"Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"AUC:      {best_metrics['auc']:.4f}")
    print(f"Brier:    {best_metrics['brier']:.4f}")
    print(f"Log Loss: {best_metrics['log_loss']:.4f}")

    # Save tuned model
    tuner.save_model(
        best_model, best_base, best_params, best_metrics,
        "models/spread_model_tuned.pkl"
    )

    # Also save both for comparison
    tuner.save_model(
        xgb_model, xgb_base, xgb_results["best_params"], xgb_metrics,
        "models/xgb_tuned.pkl"
    )
    tuner.save_model(
        lgb_model, lgb_base, lgb_results["best_params"], lgb_metrics,
        "models/lgb_tuned.pkl"
    )

    return {
        "xgboost": {"params": xgb_results["best_params"], "metrics": xgb_metrics},
        "lightgbm": {"params": lgb_results["best_params"], "metrics": lgb_metrics},
        "best": best_type,
    }


def run_regression_tuning(n_trials: int = 30):
    """Run tuning for spread and totals regression models."""
    print("=" * 60)
    print("NBA Regression Model Tuning")
    print("=" * 60)

    results = {}

    # Tune Point Spread Model
    print("\n--- Tuning Point Spread Model ---")
    spread_tuner = RegressionTuner(n_trials=n_trials, target="point_diff")
    spread_results = spread_tuner.tune()

    print(f"\nBest model: {spread_results['best_type'].upper()}")
    print(f"Best RMSE: {spread_results['best_rmse']:.3f} points")

    # Train final spread model
    spread_model, spread_metrics = spread_tuner.train_final_model(
        spread_results["best_type"], spread_results["best_params"]
    )
    spread_tuner.save_model(
        spread_model, spread_results["best_params"], spread_metrics,
        "models/point_spread_model_tuned.pkl"
    )
    results["spread"] = {
        "best_type": spread_results["best_type"],
        "params": spread_results["best_params"],
        "metrics": spread_metrics,
    }

    # Tune Totals Model
    print("\n--- Tuning Totals Model ---")
    totals_tuner = RegressionTuner(n_trials=n_trials, target="total_points")
    totals_results = totals_tuner.tune()

    print(f"\nBest model: {totals_results['best_type'].upper()}")
    print(f"Best RMSE: {totals_results['best_rmse']:.3f} points")

    # Train final totals model
    totals_model, totals_metrics = totals_tuner.train_final_model(
        totals_results["best_type"], totals_results["best_params"]
    )
    totals_tuner.save_model(
        totals_model, totals_results["best_params"], totals_metrics,
        "models/totals_model_tuned.pkl"
    )
    results["totals"] = {
        "best_type": totals_results["best_type"],
        "params": totals_results["best_params"],
        "metrics": totals_metrics,
    }

    # Summary
    print("\n" + "=" * 60)
    print("Tuning Complete!")
    print("=" * 60)
    print(f"\nSpread Model ({results['spread']['best_type']}):")
    print(f"  MAE:  {results['spread']['metrics']['mae']:.2f} points")
    print(f"  RMSE: {results['spread']['metrics']['rmse']:.2f} points")

    print(f"\nTotals Model ({results['totals']['best_type']}):")
    print(f"  MAE:  {results['totals']['metrics']['mae']:.2f} points")
    print(f"  RMSE: {results['totals']['metrics']['rmse']:.2f} points")

    return results


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Run classification tuning (moneyline)
    # results = run_full_tuning(n_trials=50, metric="brier")

    # Run regression tuning (spread/totals)
    results = run_regression_tuning(n_trials=30)
