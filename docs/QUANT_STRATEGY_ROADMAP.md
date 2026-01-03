# Quant-Style NBA Betting Model: Complete Strategy Roadmap

## Executive Summary

This document outlines a comprehensive plan to transform the current NBA betting model into a sophisticated, quant-like system focused on **long-term edge sustainability**. The approach draws from quantitative finance principles: model diversity, uncertainty quantification, regime detection, and systematic alpha monitoring.

**Core Philosophy**: Sports betting markets are less efficient than financial markets, but they're getting smarter. Sustainable edge requires systems that adapt and detect when edges decay.

---

## Current System Assessment

### What We Have
| Component | Implementation | Performance |
|-----------|---------------|-------------|
| **Models** | XGBoost for spreads/totals/ML | 55-58% win rate |
| **Features** | Team rolling stats, Elo, rest, travel, altitude | ~40 features |
| **Strategy** | 5+ point edge, no B2B, team exclusions | 4-10% ROI |
| **Sizing** | 1/5 Kelly criterion | Max 5% per bet |
| **Tracking** | SQLite bet tracker with CLV | Basic analytics |

### Key Files
- `src/models/spread_model.py` - Primary XGBoost model
- `src/models/dual_model.py` - MLP + XGBoost ensemble
- `src/features/team_features.py` - Rolling team statistics
- `src/betting/kelly.py` - Kelly criterion sizing
- `src/betting/edge_strategy.py` - Bet selection filters
- `src/bet_tracker.py` - Performance tracking with CLV

### Identified Gaps
1. **Single model type** - All XGBoost variations, correlated errors
2. **No uncertainty quantification** - Can't distinguish confident vs uncertain predictions
3. **No regime detection** - Can't detect when market conditions shift
4. **No alpha monitoring** - Edge decay not systematically tracked
5. **Team-level only** - Missing player-level granularity

---

## Phase 1: Alpha Monitoring Infrastructure

> **Priority**: CRITICAL
> **Rationale**: Can't improve what you can't measure. Early warning of edge decay.

### 1.1 Core Monitoring Module

**New File**: `src/monitoring/alpha_monitor.py`

```python
class AlphaMonitor:
    """
    Central monitoring system for betting edge health.
    Tracks rolling performance, detects decay, and generates alerts.
    """

    def __init__(self, bet_tracker, lookback_windows=[20, 50, 100]):
        self.tracker = bet_tracker
        self.windows = lookback_windows
        self.baseline_metrics = None

    def get_rolling_metrics(self, window: int = 50) -> Dict:
        """
        Calculate rolling performance metrics.

        Returns:
            - rolling_clv: Average CLV over last N bets
            - rolling_win_rate: Win rate over last N bets
            - rolling_roi: ROI over last N bets
            - clv_trend: Slope of CLV over time (positive = improving)
            - sharpe_ratio: Risk-adjusted returns
        """

    def detect_performance_decay(self, threshold_clv=-0.005) -> List[Alert]:
        """
        Generate alerts when performance degrades.

        Alert conditions:
        - CLV drops below threshold for sustained period
        - Win rate below break-even (52.4% for -110 odds)
        - Sharpe ratio turns negative
        - Sudden drop in any metric (>2 std dev)
        """

    def get_signal_health(self, feature_name: str) -> Dict:
        """
        Track predictive power of individual features over time.

        Returns:
        - current_correlation: Feature correlation with outcomes
        - historical_correlation: Rolling 100-bet correlation
        - trend: Is signal strengthening or weakening?
        - contribution: Feature importance in current model
        """

    def save_baseline(self, path: str):
        """Save current metrics as baseline for future comparison."""

    def compare_to_baseline(self) -> Dict:
        """Compare current performance to saved baseline."""
```

### 1.2 Feature Drift Detection

**New File**: `src/monitoring/feature_drift.py`

```python
class FeatureDriftMonitor:
    """
    Detect changes in feature distributions and importance.
    Uses Population Stability Index (PSI) and importance comparison.
    """

    def calculate_psi(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        Population Stability Index.
        PSI < 0.1: No significant change
        PSI 0.1-0.2: Moderate change, monitor
        PSI > 0.2: Significant change, investigate
        """

    def detect_distribution_drift(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame,
        threshold: float = 0.2
    ) -> Dict[str, float]:
        """Return PSI for each feature."""

    def detect_importance_drift(
        self,
        baseline_importance: pd.DataFrame,
        current_importance: pd.DataFrame,
        threshold: float = 0.3  # 30% relative change
    ) -> List[str]:
        """Return features with significant importance changes."""

    def generate_drift_report(self) -> Dict:
        """
        Comprehensive drift analysis.
        - Features with distribution drift
        - Features with importance drift
        - Recommended actions
        """
```

### 1.3 Enhanced Bet Tracker Analytics

**Modify**: `src/bet_tracker.py`

Add new methods:

```python
def get_rolling_clv(self, window: int = 50) -> pd.DataFrame:
    """
    Rolling CLV with confidence intervals.

    Returns DataFrame with:
    - date
    - rolling_clv (mean)
    - clv_std (standard deviation)
    - clv_lower (95% CI lower)
    - clv_upper (95% CI upper)
    - n_bets (count in window)
    """

def get_performance_by_edge_bucket(self) -> pd.DataFrame:
    """
    Break down performance by edge level.

    Buckets: 3-5%, 5-7%, 7-10%, 10%+
    For each: win_rate, roi, clv, n_bets, p_value
    """

def get_clv_trend(self, window: int = 100) -> Dict:
    """
    Analyze CLV trend over time.

    Returns:
    - slope: Linear regression slope
    - r_squared: Fit quality
    - direction: 'improving', 'stable', 'declining'
    - days_to_zero: If declining, extrapolated days until CLV = 0
    """

def get_rolling_sharpe(self, window: int = 50, risk_free_rate: float = 0) -> pd.Series:
    """Rolling Sharpe ratio of betting returns."""

def get_performance_by_regime(self, regime_labels: pd.Series) -> pd.DataFrame:
    """Performance breakdown by market regime."""
```

### 1.4 Alert System

**New File**: `src/monitoring/alerts.py`

```python
@dataclass
class Alert:
    severity: str  # 'info', 'warning', 'critical'
    category: str  # 'performance', 'drift', 'regime'
    message: str
    metrics: Dict
    timestamp: datetime
    recommended_action: str

class AlertSystem:
    """
    Centralized alert management.
    """

    def __init__(self, config: Dict):
        self.thresholds = config
        self.active_alerts = []

    def check_all(self, alpha_monitor, drift_monitor, regime_detector) -> List[Alert]:
        """Run all alert checks and return active alerts."""

    def get_daily_summary(self) -> str:
        """Generate daily health report."""
```

---

## Phase 2: Model Diversity & Ensemble Framework

> **Priority**: HIGH
> **Rationale**: Orthogonal models provide uncorrelated errors, more robust to market regime changes.

### 2.1 Base Model Interface

**New File**: `src/models/base_model.py`

```python
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Abstract base class for all prediction models.
    Ensures consistent interface for ensemble integration.
    """

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None
    ) -> 'BaseModel':
        """Train the model."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability predictions."""
        pass

    @abstractmethod
    def get_uncertainty(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return prediction uncertainty/confidence.
        Higher values = more uncertain.
        """
        pass

    @abstractmethod
    def feature_importance(self) -> pd.DataFrame:
        """Return feature importance scores."""
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Binary predictions (default threshold 0.5)."""
        return (self.predict_proba(X) > 0.5).astype(int)

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'BaseModel':
        pass
```

### 2.2 LightGBM Implementation

**New File**: `src/models/lgbm_model.py`

```python
import lightgbm as lgb

class LGBMSpreadModel(BaseModel):
    """
    LightGBM-based spread prediction model.

    Advantages over XGBoost:
    - Faster training on large datasets
    - Native categorical feature support
    - Different regularization approach (leaf-wise vs level-wise)
    """

    DEFAULT_PARAMS = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 1.0,
        'min_child_samples': 20,
        'verbose': -1,
    }

    def __init__(self, params: dict = None, n_estimators: int = 200):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.n_estimators = n_estimators
        self.model = None
        self.feature_columns = None

    def fit(self, X, y, X_val=None, y_val=None):
        self.feature_columns = X.columns.tolist()
        train_data = lgb.Dataset(X, label=y)
        valid_sets = [train_data]

        if X_val is not None:
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(valid_data)

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            callbacks=[lgb.early_stopping(20, verbose=False)]
        )
        return self

    def predict_proba(self, X):
        return self.model.predict(X[self.feature_columns])

    def get_uncertainty(self, X):
        # Use prediction variance from leaf predictions
        # LightGBM doesn't have built-in uncertainty, use entropy proxy
        probs = self.predict_proba(X)
        entropy = -probs * np.log(probs + 1e-10) - (1-probs) * np.log(1-probs + 1e-10)
        return entropy / np.log(2)  # Normalize to [0, 1]

    def feature_importance(self):
        return pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
```

### 2.3 CatBoost Implementation

**New File**: `src/models/catboost_model.py`

```python
from catboost import CatBoostClassifier

class CatBoostSpreadModel(BaseModel):
    """
    CatBoost-based spread prediction model.

    Advantages:
    - Superior handling of categorical features
    - Ordered boosting reduces prediction shift
    - Built-in GPU support
    - Robust to overfitting
    """

    DEFAULT_PARAMS = {
        'iterations': 200,
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3.0,
        'random_strength': 1.0,
        'bagging_temperature': 1.0,
        'border_count': 128,
        'loss_function': 'Logloss',
        'verbose': False,
    }

    def __init__(self, params: dict = None, cat_features: List[str] = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.cat_features = cat_features or []
        self.model = None

    def fit(self, X, y, X_val=None, y_val=None):
        self.feature_columns = X.columns.tolist()

        self.model = CatBoostClassifier(**self.params)

        eval_set = None
        if X_val is not None:
            eval_set = (X_val, y_val)

        self.model.fit(
            X, y,
            eval_set=eval_set,
            early_stopping_rounds=20,
            cat_features=self.cat_features,
        )
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def get_uncertainty(self, X):
        # CatBoost supports uncertainty via virtual ensembles
        # For now, use entropy-based proxy
        probs = self.predict_proba(X)
        entropy = -probs * np.log(probs + 1e-10) - (1-probs) * np.log(1-probs + 1e-10)
        return entropy / np.log(2)

    def feature_importance(self):
        return pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
```

### 2.4 Neural Network with MC Dropout

**New File**: `src/models/neural_model.py`

```python
import torch
import torch.nn as nn

class MCDropoutMLP(nn.Module):
    """
    MLP with dropout that remains active during inference
    for Monte Carlo uncertainty estimation.
    """

    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class NeuralSpreadModel(BaseModel):
    """
    Neural network spread model with MC Dropout uncertainty.

    Advantages:
    - Captures non-linear interactions
    - Built-in uncertainty quantification
    - Different inductive bias than tree models
    """

    def __init__(
        self,
        hidden_dims=[64, 32],
        dropout=0.2,
        learning_rate=0.001,
        epochs=100,
        batch_size=64,
        mc_samples=100
    ):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.mc_samples = mc_samples
        self.model = None
        self.scaler = None

    def fit(self, X, y, X_val=None, y_val=None):
        from sklearn.preprocessing import StandardScaler

        self.feature_columns = X.columns.tolist()
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = MCDropoutMLP(
            input_dim=X.shape[1],
            hidden_dims=self.hidden_dims,
            dropout=self.dropout
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y.values).unsqueeze(1)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X[self.feature_columns])
        X_tensor = torch.FloatTensor(X_scaled)

        self.model.eval()
        with torch.no_grad():
            return self.model(X_tensor).numpy().flatten()

    def get_uncertainty(self, X):
        """
        MC Dropout uncertainty estimation.
        Run forward pass multiple times with dropout enabled.
        """
        X_scaled = self.scaler.transform(X[self.feature_columns])
        X_tensor = torch.FloatTensor(X_scaled)

        self.model.train()  # Enable dropout
        predictions = []

        with torch.no_grad():
            for _ in range(self.mc_samples):
                pred = self.model(X_tensor).numpy().flatten()
                predictions.append(pred)

        predictions = np.array(predictions)
        return predictions.std(axis=0)  # Epistemic uncertainty

    def predict_with_uncertainty(self, X):
        """Return both prediction and uncertainty."""
        X_scaled = self.scaler.transform(X[self.feature_columns])
        X_tensor = torch.FloatTensor(X_scaled)

        self.model.train()
        predictions = []

        with torch.no_grad():
            for _ in range(self.mc_samples):
                pred = self.model(X_tensor).numpy().flatten()
                predictions.append(pred)

        predictions = np.array(predictions)
        return predictions.mean(axis=0), predictions.std(axis=0)

    def feature_importance(self):
        # Use gradient-based importance
        # Simplified: return uniform importance
        return pd.DataFrame({
            'feature': self.feature_columns,
            'importance': [1.0 / len(self.feature_columns)] * len(self.feature_columns)
        })
```

### 2.5 Ensemble Orchestrator

**New File**: `src/models/ensemble.py`

```python
class EnsembleModel:
    """
    Dynamic ensemble combining multiple models.

    Weighting strategies:
    - 'equal': Simple average
    - 'optimized': Optimize weights on validation set
    - 'dynamic': Adjust based on recent CLV performance
    - 'stacking': Meta-learner on model predictions
    """

    def __init__(
        self,
        models: Dict[str, BaseModel],
        weighting: str = 'dynamic',
        clv_window: int = 50
    ):
        self.models = models
        self.weighting = weighting
        self.clv_window = clv_window
        self.weights = {name: 1.0 / len(models) for name in models}
        self.meta_model = None

    def fit(self, X, y, X_val=None, y_val=None):
        """Train all constituent models."""
        for name, model in self.models.items():
            model.fit(X, y, X_val, y_val)

        if self.weighting == 'optimized' and X_val is not None:
            self._optimize_weights(X_val, y_val)
        elif self.weighting == 'stacking' and X_val is not None:
            self._fit_stacking(X_val, y_val)

        return self

    def _optimize_weights(self, X_val, y_val):
        """Optimize weights to minimize log loss on validation."""
        from scipy.optimize import minimize

        predictions = {name: model.predict_proba(X_val)
                       for name, model in self.models.items()}

        def loss(weights):
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize

            ensemble_pred = sum(
                w * predictions[name]
                for w, name in zip(weights, self.models.keys())
            )

            # Log loss
            eps = 1e-10
            return -np.mean(
                y_val * np.log(ensemble_pred + eps) +
                (1 - y_val) * np.log(1 - ensemble_pred + eps)
            )

        n_models = len(self.models)
        result = minimize(
            loss,
            x0=[1.0 / n_models] * n_models,
            bounds=[(0, 1)] * n_models,
            constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1}
        )

        self.weights = dict(zip(self.models.keys(), result.x))

    def _fit_stacking(self, X_val, y_val):
        """Train meta-learner on model predictions."""
        from sklearn.linear_model import LogisticRegression

        meta_features = np.column_stack([
            model.predict_proba(X_val) for model in self.models.values()
        ])

        self.meta_model = LogisticRegression()
        self.meta_model.fit(meta_features, y_val)

    def predict_proba(self, X):
        """Weighted ensemble prediction."""
        if self.weighting == 'stacking' and self.meta_model is not None:
            meta_features = np.column_stack([
                model.predict_proba(X) for model in self.models.values()
            ])
            return self.meta_model.predict_proba(meta_features)[:, 1]

        predictions = {
            name: model.predict_proba(X)
            for name, model in self.models.items()
        }

        return sum(
            self.weights[name] * predictions[name]
            for name in self.models.keys()
        )

    def get_uncertainty(self, X):
        """
        Ensemble uncertainty from model disagreement.
        High disagreement = high uncertainty.
        """
        predictions = [model.predict_proba(X) for model in self.models.values()]
        predictions = np.array(predictions)

        # Disagreement-based uncertainty
        return predictions.std(axis=0)

    def get_model_predictions(self, X) -> Dict[str, np.ndarray]:
        """Get individual model predictions for analysis."""
        return {
            name: model.predict_proba(X)
            for name, model in self.models.items()
        }

    def update_weights_from_clv(self, clv_by_model: Dict[str, float]):
        """
        Adjust weights based on recent CLV performance.
        Models with positive CLV get higher weights.
        """
        if self.weighting != 'dynamic':
            return

        # Softmax weighting based on CLV
        clv_values = np.array([clv_by_model.get(name, 0) for name in self.models.keys()])

        # Temperature parameter controls sharpness
        temperature = 0.1
        exp_clv = np.exp(clv_values / temperature)
        new_weights = exp_clv / exp_clv.sum()

        self.weights = dict(zip(self.models.keys(), new_weights))

    def get_model_correlations(self, X) -> pd.DataFrame:
        """
        Calculate correlation between model predictions.
        Low correlation = more diversity = better ensemble.
        """
        predictions = self.get_model_predictions(X)
        pred_df = pd.DataFrame(predictions)
        return pred_df.corr()
```

---

## Phase 3: Uncertainty Quantification

> **Priority**: HIGH
> **Rationale**: Know when to bet big vs when to pass.

### 3.1 Conformal Prediction

**New File**: `src/models/conformal.py`

```python
class ConformalPredictor:
    """
    Conformal prediction wrapper for any point predictor.
    Provides distribution-free prediction intervals with coverage guarantees.

    Coverage guarantee: P(Y ∈ interval) ≥ 1 - α
    """

    def __init__(self, base_model: BaseModel, alpha: float = 0.1):
        self.base_model = base_model
        self.alpha = alpha
        self.calibration_scores = None

    def calibrate(self, X_cal: pd.DataFrame, y_cal: pd.Series):
        """
        Calibrate on held-out calibration set.
        Computes nonconformity scores for prediction intervals.
        """
        predictions = self.base_model.predict_proba(X_cal)

        # Nonconformity score: how wrong was the prediction?
        # For binary classification: 1 - predicted probability of true class
        self.calibration_scores = np.where(
            y_cal == 1,
            1 - predictions,  # Score when true class is 1
            predictions       # Score when true class is 0
        )

    def predict_interval(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (lower, upper) prediction intervals.

        For probability predictions, returns confidence bounds.
        """
        predictions = self.base_model.predict_proba(X)

        # Quantile of calibration scores
        q = np.quantile(self.calibration_scores, 1 - self.alpha)

        lower = np.maximum(0, predictions - q)
        upper = np.minimum(1, predictions + q)

        return lower, upper

    def predict_with_coverage(self, X: pd.DataFrame) -> Dict:
        """
        Return predictions with coverage information.
        """
        predictions = self.base_model.predict_proba(X)
        lower, upper = self.predict_interval(X)

        return {
            'prediction': predictions,
            'lower': lower,
            'upper': upper,
            'interval_width': upper - lower,
            'coverage_level': 1 - self.alpha,
        }


class SplitConformalClassifier:
    """
    Split conformal prediction for classification.
    Uses adaptive prediction sets.
    """

    def __init__(self, base_model: BaseModel, alpha: float = 0.1):
        self.base_model = base_model
        self.alpha = alpha
        self.qhat = None

    def calibrate(self, X_cal, y_cal):
        """Compute conformal quantile."""
        scores = self.base_model.predict_proba(X_cal)

        # Conformity score: probability of true class
        conformity = np.where(y_cal == 1, scores, 1 - scores)

        n = len(y_cal)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.qhat = np.quantile(conformity, q_level)

    def predict_set(self, X) -> List[Set]:
        """
        Return prediction sets that contain true label with probability ≥ 1-α.
        """
        probs = self.base_model.predict_proba(X)

        prediction_sets = []
        for p in probs:
            pset = set()
            if p >= self.qhat:
                pset.add(1)
            if (1 - p) >= self.qhat:
                pset.add(0)
            prediction_sets.append(pset)

        return prediction_sets
```

### 3.2 Uncertainty-Adjusted Kelly

**Modify**: `src/betting/kelly.py`

```python
def calculate_uncertainty_adjusted_kelly(
    self,
    win_prob: float,
    prob_uncertainty: float,
    odds: float,
    odds_format: str = "american",
    uncertainty_penalty: float = 2.0,
) -> float:
    """
    Kelly criterion adjusted for prediction uncertainty.

    When uncertainty is high, reduce position size to account for
    model confidence. Uses a conservative approach that penalizes
    uncertain predictions.

    Args:
        win_prob: Model's estimated win probability
        prob_uncertainty: Standard deviation of probability estimate
        odds: Betting odds
        odds_format: "american", "decimal", or "implied"
        uncertainty_penalty: How much to penalize uncertainty (default 2.0)

    Returns:
        Adjusted Kelly fraction (always ≤ standard Kelly)
    """
    # Standard Kelly
    standard_kelly = self.calculate_kelly(win_prob, odds, odds_format)

    if standard_kelly <= 0:
        return 0

    # Uncertainty adjustment factor
    # When uncertainty is 0, factor = 1 (full Kelly)
    # When uncertainty is high, factor approaches 0
    max_uncertainty = 0.25  # Maximum possible for binary classification
    normalized_uncertainty = min(prob_uncertainty / max_uncertainty, 1.0)

    adjustment_factor = np.exp(-uncertainty_penalty * normalized_uncertainty)

    adjusted_kelly = standard_kelly * adjustment_factor

    # Never exceed max bet percentage
    return min(adjusted_kelly, self.max_bet_pct)


def calculate_robust_kelly(
    self,
    win_prob_samples: np.ndarray,
    odds: float,
    odds_format: str = "american",
) -> float:
    """
    Robust Kelly using probability distribution.

    Instead of point estimate, uses samples from probability distribution
    (e.g., from MC Dropout) to compute expected Kelly.

    More conservative when probability distribution is wide.
    """
    kellys = [
        self.calculate_kelly(p, odds, odds_format)
        for p in win_prob_samples
    ]

    # Use lower percentile for robustness
    return np.percentile(kellys, 25)
```

---

## Phase 4: Regime Detection & Adaptive Learning

> **Priority**: HIGH
> **Rationale**: Markets change. Adapt or die.

### 4.1 Market Regime Detection

**New File**: `src/monitoring/regime_detection.py`

```python
from hmmlearn import hmm
import ruptures as rpt

class MarketRegimeDetector:
    """
    Detect changes in market behavior using Hidden Markov Models
    and change point detection.

    Regimes:
    - 'normal': Stable edge, standard market efficiency
    - 'volatile': High variance in outcomes, uncertain
    - 'efficient': Market has adapted, reduced edge
    - 'opportunity': Temporary inefficiency, increased edge
    """

    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.hmm_model = None
        self.regime_labels = ['efficient', 'normal', 'opportunity']

    def fit(self, market_features: np.ndarray):
        """
        Fit HMM on historical market features.

        market_features should include:
        - CLV distribution
        - Line movement magnitude
        - Model error
        - Market volume indicators
        """
        self.hmm_model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type='full',
            n_iter=100
        )
        self.hmm_model.fit(market_features)

    def predict_regime(self, recent_features: np.ndarray) -> str:
        """Predict current market regime."""
        regime_idx = self.hmm_model.predict(recent_features)[-1]
        return self.regime_labels[regime_idx]

    def get_regime_probabilities(self, recent_features: np.ndarray) -> Dict[str, float]:
        """Get probability distribution over regimes."""
        probs = self.hmm_model.predict_proba(recent_features)[-1]
        return dict(zip(self.regime_labels, probs))


class ChangePointDetector:
    """
    Detect structural breaks in performance metrics.
    Uses PELT algorithm for efficient change point detection.
    """

    def __init__(self, model: str = "rbf", min_size: int = 20):
        self.model = model
        self.min_size = min_size

    def detect_changes(
        self,
        metric_series: np.ndarray,
        penalty: float = 10.0
    ) -> List[int]:
        """
        Detect change points in metric series.

        Returns indices where structural breaks occurred.
        """
        algo = rpt.Pelt(model=self.model, min_size=self.min_size)
        algo.fit(metric_series)
        return algo.predict(pen=penalty)

    def detect_recent_change(
        self,
        metric_series: np.ndarray,
        lookback: int = 100
    ) -> Optional[int]:
        """
        Check if there's a recent change point.
        Returns index of change if found, None otherwise.
        """
        if len(metric_series) < lookback:
            return None

        recent = metric_series[-lookback:]
        changes = self.detect_changes(recent)

        if len(changes) > 1:  # Last element is always end of series
            return len(metric_series) - lookback + changes[-2]
        return None


class AdaptiveStrategy:
    """
    Adjust betting strategy based on detected regime.
    """

    REGIME_ADJUSTMENTS = {
        'efficient': {
            'edge_threshold': 0.08,  # Higher threshold needed
            'kelly_fraction': 0.1,   # More conservative
            'confidence_min': 0.6,   # Only high confidence bets
        },
        'normal': {
            'edge_threshold': 0.05,
            'kelly_fraction': 0.2,
            'confidence_min': 0.55,
        },
        'opportunity': {
            'edge_threshold': 0.03,  # Lower threshold OK
            'kelly_fraction': 0.25,  # More aggressive
            'confidence_min': 0.52,
        },
    }

    def __init__(self, regime_detector: MarketRegimeDetector):
        self.detector = regime_detector
        self.current_regime = 'normal'

    def update_regime(self, recent_features: np.ndarray):
        """Update current regime based on recent data."""
        self.current_regime = self.detector.predict_regime(recent_features)

    def get_current_parameters(self) -> Dict:
        """Get betting parameters for current regime."""
        return self.REGIME_ADJUSTMENTS[self.current_regime]
```

### 4.2 Seasonal Features

**Modify**: `src/features/team_features.py`

```python
def add_season_phase_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features capturing seasonal patterns.

    Market efficiency varies by season phase:
    - Early season: More uncertainty, potentially more edge
    - Pre All-Star: Teams settling into patterns
    - Post All-Star: Playoff push begins, motivation varies
    - Playoff push: High motivation for bubble teams
    - End of season: Tanking teams, resting stars
    """
    df = df.copy()

    # Parse date
    df['game_date'] = pd.to_datetime(df['date'])
    df['month'] = df['game_date'].dt.month
    df['day_of_week'] = df['game_date'].dt.dayofweek

    # Season phases (approximate for NBA)
    # Oct-Nov: Early season
    # Dec-Feb: Pre All-Star
    # Feb-Mar: Post All-Star
    # Apr: Playoff push

    def get_season_phase(month):
        if month in [10, 11]:
            return 'early_season'
        elif month in [12, 1]:
            return 'mid_season_1'
        elif month == 2:
            return 'all_star_break'
        elif month == 3:
            return 'mid_season_2'
        elif month == 4:
            return 'playoff_push'
        else:
            return 'playoffs'

    df['season_phase'] = df['month'].apply(get_season_phase)

    # One-hot encode
    phase_dummies = pd.get_dummies(df['season_phase'], prefix='phase')
    df = pd.concat([df, phase_dummies], axis=1)

    # Days since season start (proxy for sample size/reliability)
    # Assume season starts Oct 15
    season_start = pd.Timestamp(year=df['game_date'].dt.year.iloc[0], month=10, day=15)
    df['days_since_season_start'] = (df['game_date'] - season_start).dt.days

    # Weekend indicator (different betting patterns)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    return df


def add_broadcast_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features for nationally televised games.

    National TV games tend to have:
    - More betting volume
    - Sharper lines
    - Potentially less edge
    """
    # This would require external data source
    # Placeholder for structure
    df['is_national_tv'] = 0  # Default
    df['is_primetime'] = 0    # Games after 8pm ET

    return df
```

### 4.3 Online Learning

**New File**: `src/models/online_learning.py`

```python
class OnlineModelUpdater:
    """
    Incremental model updates without full retraining.
    Useful for adapting to recent performance changes.
    """

    def __init__(
        self,
        base_model: BaseModel,
        update_frequency: int = 50,  # Games between updates
        min_new_samples: int = 20,
        performance_threshold: float = -0.01  # CLV threshold
    ):
        self.base_model = base_model
        self.update_frequency = update_frequency
        self.min_new_samples = min_new_samples
        self.performance_threshold = performance_threshold
        self.games_since_update = 0
        self.new_samples = []

    def add_sample(self, X: pd.DataFrame, y: pd.Series):
        """Add new sample for potential update."""
        self.new_samples.append((X, y))
        self.games_since_update += 1

    def should_update(self, recent_clv: float) -> bool:
        """
        Determine if model should be updated.

        Triggers:
        - Regular interval reached
        - Performance dropped below threshold
        - Enough new samples accumulated
        """
        if len(self.new_samples) < self.min_new_samples:
            return False

        if self.games_since_update >= self.update_frequency:
            return True

        if recent_clv < self.performance_threshold:
            return True

        return False

    def update(self, full_X: pd.DataFrame = None, full_y: pd.Series = None):
        """
        Perform model update.

        Options:
        - Fine-tune on new samples only
        - Retrain on all data with recent samples weighted higher
        """
        if full_X is not None:
            # Full retrain with sample weighting
            sample_weights = self._calculate_sample_weights(full_X)
            # Implementation depends on model type
            pass
        else:
            # Fine-tune on new samples
            new_X = pd.concat([s[0] for s in self.new_samples])
            new_y = pd.concat([s[1] for s in self.new_samples])
            # Warm start / incremental fit
            pass

        self.games_since_update = 0
        self.new_samples = []

    def _calculate_sample_weights(self, X: pd.DataFrame) -> np.ndarray:
        """
        Weight recent samples higher.
        Exponential decay based on recency.
        """
        n = len(X)
        decay = 0.99
        weights = decay ** np.arange(n)[::-1]
        return weights / weights.sum() * n


class RetrainingScheduler:
    """
    Manage model retraining schedule.
    """

    def __init__(
        self,
        regular_interval_days: int = 30,
        min_new_games: int = 100,
        performance_trigger_clv: float = -0.02,
        drift_trigger_psi: float = 0.2
    ):
        self.regular_interval = regular_interval_days
        self.min_new_games = min_new_games
        self.clv_trigger = performance_trigger_clv
        self.psi_trigger = drift_trigger_psi
        self.last_retrain_date = None

    def should_retrain(
        self,
        current_date: datetime,
        new_games: int,
        rolling_clv: float,
        max_psi: float
    ) -> Tuple[bool, str]:
        """
        Check if retraining is needed.

        Returns (should_retrain, reason)
        """
        if self.last_retrain_date is None:
            return True, "initial_training"

        days_since = (current_date - self.last_retrain_date).days

        if days_since >= self.regular_interval and new_games >= self.min_new_games:
            return True, "regular_schedule"

        if rolling_clv < self.clv_trigger:
            return True, "performance_decay"

        if max_psi > self.psi_trigger:
            return True, "feature_drift"

        return False, "no_trigger"

    def record_retrain(self, date: datetime):
        """Record that retraining occurred."""
        self.last_retrain_date = date
```

---

## Phase 5: Player-Level Modeling

> **Priority**: MEDIUM
> **Rationale**: More granular signals are harder for market to price.

### 5.1 Player Impact Model (RAPM-Style)

**New File**: `src/features/player_impact.py`

```python
from sklearn.linear_model import Ridge

class PlayerImpactModel:
    """
    Regularized Adjusted Plus-Minus (RAPM) style player valuation.

    Estimates each player's contribution to team point differential
    per 100 possessions, controlling for teammates and opponents.
    """

    def __init__(self, regularization: float = 2000):
        self.regularization = regularization
        self.model = None
        self.player_impacts = {}
        self.player_to_idx = {}

    def fit(self, stints_df: pd.DataFrame):
        """
        Fit player impact model on stint-level data.

        stints_df columns:
        - stint_id: Unique identifier
        - home_players: List of 5 player IDs on home team
        - away_players: List of 5 player IDs on away team
        - point_diff_per_100: Point differential per 100 possessions
        - possessions: Number of possessions in stint
        """
        # Build player index
        all_players = set()
        for _, row in stints_df.iterrows():
            all_players.update(row['home_players'])
            all_players.update(row['away_players'])

        self.player_to_idx = {p: i for i, p in enumerate(sorted(all_players))}
        n_players = len(self.player_to_idx)

        # Build design matrix
        # Each row: +1 for home players, -1 for away players
        n_stints = len(stints_df)
        X = np.zeros((n_stints, n_players))
        y = np.zeros(n_stints)
        weights = np.zeros(n_stints)

        for i, (_, row) in enumerate(stints_df.iterrows()):
            for player in row['home_players']:
                X[i, self.player_to_idx[player]] = 1
            for player in row['away_players']:
                X[i, self.player_to_idx[player]] = -1
            y[i] = row['point_diff_per_100']
            weights[i] = row['possessions']

        # Fit ridge regression
        self.model = Ridge(alpha=self.regularization)
        self.model.fit(X, y, sample_weight=weights)

        # Extract player impacts
        self.player_impacts = {
            player: self.model.coef_[idx]
            for player, idx in self.player_to_idx.items()
        }

    def get_player_impact(self, player_id: str) -> float:
        """Get impact for single player (points per 100 possessions)."""
        return self.player_impacts.get(player_id, 0.0)

    def get_lineup_impact(self, players: List[str]) -> float:
        """Get total impact for a lineup."""
        return sum(self.get_player_impact(p) for p in players)

    def get_game_prediction(
        self,
        home_players: List[str],
        away_players: List[str],
        expected_pace: float = 100
    ) -> float:
        """
        Predict point differential for a game.

        Returns expected home margin.
        """
        home_impact = self.get_lineup_impact(home_players)
        away_impact = self.get_lineup_impact(away_players)

        # Impact is per 100 possessions, adjust for game pace
        return (home_impact - away_impact) * (expected_pace / 100)

    def get_top_players(self, n: int = 20) -> pd.DataFrame:
        """Get top N players by impact."""
        sorted_players = sorted(
            self.player_impacts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return pd.DataFrame(
            sorted_players[:n],
            columns=['player_id', 'impact']
        )


class InjuryImpactModel:
    """
    Model impact of injuries using historical data.

    Learns how team performance changes when specific players are out,
    rather than using heuristic point estimates.
    """

    def __init__(self, player_impact_model: PlayerImpactModel):
        self.player_model = player_impact_model
        self.injury_adjustments = {}

    def fit(self, games_df: pd.DataFrame, injuries_df: pd.DataFrame):
        """
        Learn injury impact from historical games.

        For each star player:
        - Compare team performance with/without them
        - Learn adjustment factor
        """
        # Group games by team and whether star was playing
        # Compare point differential
        # Fit adjustment model
        pass

    def predict_injury_impact(
        self,
        team: str,
        missing_players: List[str]
    ) -> float:
        """
        Predict point differential impact of injuries.

        Returns expected change in team's point differential.
        """
        total_impact = 0
        for player in missing_players:
            player_impact = self.player_model.get_player_impact(player)
            # Replacement level is approximately -2.0
            replacement_impact = -2.0
            total_impact += (replacement_impact - player_impact)

        return total_impact
```

### 5.2 Lineup Features

**New File**: `src/features/lineup_features.py`

```python
class LineupFeatureBuilder:
    """
    Build features for specific lineup combinations.
    """

    def __init__(self, player_impact_model: PlayerImpactModel):
        self.impact_model = player_impact_model
        self.lineup_history = {}

    def build_lineup_features(
        self,
        home_lineup: List[str],
        away_lineup: List[str],
        player_stats: pd.DataFrame = None
    ) -> Dict[str, float]:
        """
        Build features for a matchup between two lineups.
        """
        features = {}

        # Impact-based features
        home_impact = self.impact_model.get_lineup_impact(home_lineup)
        away_impact = self.impact_model.get_lineup_impact(away_lineup)

        features['home_lineup_impact'] = home_impact
        features['away_lineup_impact'] = away_impact
        features['lineup_impact_diff'] = home_impact - away_impact

        # Chemistry features (how often have these players played together?)
        features['home_lineup_familiarity'] = self._calculate_familiarity(home_lineup)
        features['away_lineup_familiarity'] = self._calculate_familiarity(away_lineup)

        # Positional balance
        if player_stats is not None:
            features.update(self._positional_features(home_lineup, away_lineup, player_stats))

        return features

    def _calculate_familiarity(self, lineup: List[str]) -> float:
        """
        Calculate how familiar a lineup is with each other.
        Based on historical minutes played together.
        """
        lineup_key = tuple(sorted(lineup))
        return self.lineup_history.get(lineup_key, 0.0)

    def _positional_features(
        self,
        home_lineup: List[str],
        away_lineup: List[str],
        player_stats: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate positional matchup features.
        """
        features = {}

        # Height advantage by position
        # Speed/pace preferences
        # Three-point shooting balance
        # Defensive rating matchups

        return features


class MatchupAnalyzer:
    """
    Analyze specific player-vs-player matchups.
    """

    def __init__(self):
        self.historical_matchups = {}

    def get_matchup_adjustment(
        self,
        home_players: List[Dict],
        away_players: List[Dict]
    ) -> float:
        """
        Calculate adjustment based on specific matchups.

        Factors:
        - Historical head-to-head performance
        - Size mismatches
        - Speed mismatches
        - Defensive assignments
        """
        adjustment = 0.0

        # Implementation would analyze position-by-position matchups

        return adjustment
```

---

## Dependencies Summary

```txt
# requirements_quant.txt

# Core ML
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2
torch>=2.0.0
scikit-learn>=1.3.0

# Uncertainty & Conformal
mapie>=0.7.0

# Regime Detection
hmmlearn>=0.3.0
ruptures>=1.1.7

# Monitoring (optional)
evidently>=0.4.0

# Data Processing
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
```

---

## File Structure Overview

```
src/
├── monitoring/                    # Phase 1: Alpha Monitoring
│   ├── __init__.py
│   ├── alpha_monitor.py          # Rolling metrics, decay detection
│   ├── feature_drift.py          # PSI, importance drift
│   ├── alerts.py                 # Alert system
│   └── regime_detection.py       # Phase 4: HMM, change points
├── models/
│   ├── base_model.py             # Phase 2: Abstract interface
│   ├── lgbm_model.py             # Phase 2: LightGBM
│   ├── catboost_model.py         # Phase 2: CatBoost
│   ├── neural_model.py           # Phase 2: MLP with MC Dropout
│   ├── ensemble.py               # Phase 2: Dynamic ensemble
│   ├── conformal.py              # Phase 3: Uncertainty quantification
│   ├── online_learning.py        # Phase 4: Incremental updates
│   ├── spread_model.py           # EXISTING: Modify to inherit BaseModel
│   ├── calibration.py            # EXISTING: Keep as-is
│   └── injury_adjustment.py      # Phase 5: Historical injury model
├── features/
│   ├── player_impact.py          # Phase 5: RAPM model
│   ├── lineup_features.py        # Phase 5: Lineup-specific
│   ├── team_features.py          # Phase 4: Add seasonal features
│   ├── game_features.py          # EXISTING
│   └── elo.py                    # EXISTING
├── betting/
│   ├── kelly.py                  # Phase 3: Uncertainty-adjusted sizing
│   ├── edge_strategy.py          # Phase 4: Regime-adaptive strategy
│   └── backtest.py               # EXISTING
├── bet_tracker.py                # Phase 1: Enhanced CLV analytics
└── data/
    └── (existing data modules)

scripts/
├── test_quant_system.py          # Integration testing
├── train_ensemble.py             # Ensemble training script
├── backtest_regimes.py           # Regime-aware backtesting
└── generate_monitoring_report.py # Daily health report
```

---

## Success Metrics

| Metric | Current | Target | How to Measure |
|--------|---------|--------|----------------|
| Average CLV | ~0% | > +0.5% | Rolling 100-bet average |
| Win Rate Stability | High variance | < 5% std dev | Rolling 50-bet std |
| Max Drawdown | 80%+ | < 50% | Backtest simulation |
| Edge Detection Speed | N/A | < 100 bets | Time to alert on decay |
| Model Correlation | 1.0 (single model) | < 0.7 | Ensemble prediction correlation |
| Uncertainty Calibration | N/A | Coverage ≥ 90% | Conformal prediction coverage |

---

## Phased Implementation Timeline

```
Phase 1: Alpha Monitoring     ████████░░░░░░░░░░░░  Foundation
Phase 2: Ensemble Framework   ░░░░████████░░░░░░░░  Model Diversity
Phase 3: Uncertainty          ░░░░░░░░████░░░░░░░░  Better Sizing
Phase 4: Regime Detection     ░░░░░░░░░░░░████░░░░  Adaptation
Phase 5: Player Modeling      ░░░░░░░░░░░░░░░░████  Granularity
```

Each phase builds on previous phases. Phases 1-3 are foundational and should be completed first. Phases 4-5 add sophistication once the foundation is solid.

---

## Risk Considerations

1. **Overfitting Risk**: More models = more degrees of freedom. Rigorous walk-forward validation required.

2. **Complexity Cost**: Each new component adds maintenance burden. Start simple, add complexity only when validated.

3. **Data Requirements**: Player-level modeling requires collecting stint/lineup data not currently captured.

4. **Market Adaptation**: Even with all enhancements, edge may still decay as markets adapt. Monitoring is key.

5. **Implementation Time**: Full implementation is substantial. Prioritize phases by expected impact.

---

*Document Version: 1.0*
*Created: January 2025*
*Status: Planning Document - Not for Implementation*
