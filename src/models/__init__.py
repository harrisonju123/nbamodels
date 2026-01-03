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

try:
    from .point_spread import PointSpreadModel
except (ImportError, Exception):
    PointSpreadModel = None

try:
    from .totals import TotalsModel
except (ImportError, Exception):
    TotalsModel = None

try:
    from .dual_model import DualPredictionModel
except (ImportError, Exception):
    DualPredictionModel = None

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

try:
    from .stacking import (
        StackedEnsembleModel,
        StackingConfig,
        create_stacked_ensemble,
    )
except (ImportError, Exception):
    StackedEnsembleModel = None
    StackingConfig = None
    create_stacked_ensemble = None

try:
    from .bayesian import (
        BayesianLinearModel,
        BayesianPrediction,
        BayesianEnsembleUncertainty,
        MCDropoutWrapper,
    )
except (ImportError, Exception):
    BayesianLinearModel = None
    BayesianPrediction = None
    BayesianEnsembleUncertainty = None
    MCDropoutWrapper = None

try:
    from .conformal import (
        ConformalPredictor,
        AdaptiveConformalPredictor,
        BayesianConformalPredictor,
        PredictionInterval,
        create_conformal_wrapper,
        create_bayesian_conformal_wrapper,
    )
except (ImportError, Exception):
    ConformalPredictor = None
    AdaptiveConformalPredictor = None
    BayesianConformalPredictor = None
    PredictionInterval = None
    create_conformal_wrapper = None
    create_bayesian_conformal_wrapper = None

try:
    from .online_learning import (
        OnlineUpdater,
        AdaptiveEnsembleUpdater,
        RetrainTrigger,
        UpdateResult,
    )
except (ImportError, Exception):
    OnlineUpdater = None
    AdaptiveEnsembleUpdater = None
    RetrainTrigger = None
    UpdateResult = None

__all__ = [
    # Base interface
    "BaseModel",
    # Original models
    "SpreadPredictionModel",
    "CalibratedModel",
    "PointSpreadModel",
    "TotalsModel",
    "DualPredictionModel",
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
    "StackedEnsembleModel",
    "StackingConfig",
    # Factory functions
    "create_xgb_model",
    "create_lgbm_model",
    "create_catboost_model",
    "create_neural_model",
    "create_default_ensemble",
    "create_stacked_ensemble",
    # Bayesian uncertainty
    "BayesianLinearModel",
    "BayesianPrediction",
    "BayesianEnsembleUncertainty",
    "MCDropoutWrapper",
    # Conformal prediction
    "ConformalPredictor",
    "AdaptiveConformalPredictor",
    "BayesianConformalPredictor",
    "PredictionInterval",
    "create_conformal_wrapper",
    "create_bayesian_conformal_wrapper",
    # Online learning
    "OnlineUpdater",
    "AdaptiveEnsembleUpdater",
    "RetrainTrigger",
    "UpdateResult",
]
