# Make imports optional to avoid loading issues

# Base model interface
try:
    from .base_model import BaseModel
except (ImportError, Exception):
    BaseModel = None

# Original models
try:
    from .spread_model import SpreadPredictionModel
except (ImportError, Exception):
    SpreadPredictionModel = None

try:
    from .calibration import CalibratedModel
except (ImportError, Exception):
    CalibratedModel = None

# Archived models (moved to archive/unused_models/ on 2026-01-09)
# - point_spread.py -> PointSpreadModel (superseded by SpreadPredictionModel)
# - totals.py -> TotalsModel (disabled strategy, +0.9% ROI not profitable)
# - dual_model.py -> DualPredictionModel (failed validation, -4.0% ROI)

from .injury_adjustment import InjuryAdjuster

# Trained injury model (Phase 5)
try:
    from .injury_adjustment import (
        TrainedInjuryModel,
        InjuryCalibration,
        build_trained_injury_model,
    )
except (ImportError, Exception):
    TrainedInjuryModel = None
    InjuryCalibration = None
    build_trained_injury_model = None

# New ensemble models
try:
    from .xgb_model import XGBSpreadModel, create_xgb_model
except (ImportError, Exception):
    XGBSpreadModel = None
    create_xgb_model = None

try:
    from .lgbm_model import LGBMSpreadModel, create_lgbm_model
except (ImportError, Exception):
    LGBMSpreadModel = None
    create_lgbm_model = None

try:
    from .catboost_model import CatBoostSpreadModel, create_catboost_model
except (ImportError, Exception):
    CatBoostSpreadModel = None
    create_catboost_model = None

try:
    from .neural_model import NeuralSpreadModel, create_neural_model
except (ImportError, Exception):
    NeuralSpreadModel = None
    create_neural_model = None

try:
    from .ensemble import EnsembleModel, create_default_ensemble
except (ImportError, Exception):
    EnsembleModel = None
    create_default_ensemble = None

# Archived models (moved to archive/unused_models/ on 2026-01-09)
# - stacking.py -> StackedEnsembleModel, StackingConfig, create_stacked_ensemble (not used in production)
# - bayesian.py -> BayesianLinearModel, BayesianPrediction, BayesianEnsembleUncertainty, MCDropoutWrapper (not used in production)
# - conformal.py -> ConformalPredictor, AdaptiveConformalPredictor, BayesianConformalPredictor (only used in tests)
# - online_learning.py -> OnlineUpdater, AdaptiveEnsembleUpdater, RetrainTrigger, UpdateResult (only used in tests)

__all__ = [
    # Base interface
    "BaseModel",
    # Production models (consolidated)
    "SpreadPredictionModel",
    "CalibratedModel",
    "InjuryAdjuster",
    # Trained injury model (Phase 5)
    "TrainedInjuryModel",
    "InjuryCalibration",
    "build_trained_injury_model",
    # Ensemble models (BaseModel compatible)
    "XGBSpreadModel",
    "LGBMSpreadModel",
    "CatBoostSpreadModel",
    "NeuralSpreadModel",
    "EnsembleModel",
    # Factory functions
    "create_xgb_model",
    "create_lgbm_model",
    "create_catboost_model",
    "create_neural_model",
    "create_default_ensemble",
]
