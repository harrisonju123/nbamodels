#!/usr/bin/env python3
"""
Quant System Integration Test

Tests all components of the quant-style betting system:
- Phase 1: Alpha Monitoring
- Phase 2: Ensemble Framework
- Phase 3: Uncertainty Quantification
- Phase 4: Regime Detection
- Phase 5: Player-Level Modeling

Runs backtest comparisons and generates performance reports.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    name: str
    n_bets: int
    win_rate: float
    roi: float
    avg_clv: float
    sharpe_ratio: float
    max_drawdown: float
    profit: float


def test_phase1_monitoring():
    """Test Alpha Monitoring infrastructure."""
    print("\n" + "=" * 60)
    print("PHASE 1: Alpha Monitoring")
    print("=" * 60)

    results = {}

    # Test AlphaMonitor
    try:
        from src.monitoring.alpha_monitor import AlphaMonitor
        monitor = AlphaMonitor()

        # Test with sample data using correct column names
        # AlphaMonitor expects 'outcome' column with 'win'/'loss' values
        # and both 'logged_at' and 'settled_at' columns
        np.random.seed(42)
        outcomes = np.random.choice(['win', 'loss'], 100, p=[0.55, 0.45])
        dates = pd.date_range('2024-01-01', periods=100)
        sample_bets = pd.DataFrame({
            'bet_id': range(100),
            'logged_at': dates,
            'settled_at': dates + pd.Timedelta(hours=3),  # Settled 3 hours after logging
            'model_prob': np.random.uniform(0.5, 0.7, 100),
            'market_prob': np.random.uniform(0.45, 0.65, 100),
            'outcome': outcomes,
            'bet_amount': [100] * 100,
            'odds': np.random.uniform(1.8, 2.2, 100),
        })
        sample_bets['profit'] = np.where(
            sample_bets['outcome'] == 'win',
            sample_bets['bet_amount'] * (sample_bets['odds'] - 1),
            -sample_bets['bet_amount']
        )
        sample_bets['clv'] = sample_bets['model_prob'] - sample_bets['market_prob']

        # Test rolling metrics
        metrics = monitor.get_rolling_metrics(sample_bets, window=20)
        results['AlphaMonitor'] = 'OK'
        print(f"  AlphaMonitor: OK")
        print(f"    Rolling metrics columns: {list(metrics.columns) if not metrics.empty else 'empty'}")

    except Exception as e:
        results['AlphaMonitor'] = f'FAIL: {e}'
        print(f"  AlphaMonitor: FAIL - {e}")

    # Test FeatureDriftMonitor (correct class name)
    try:
        from src.monitoring.feature_drift import FeatureDriftMonitor
        monitor = FeatureDriftMonitor()
        results['FeatureDriftMonitor'] = 'OK'

        # Test PSI calculation
        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(0.2, 1.1, 1000)
        psi = monitor.calculate_psi(baseline, current)
        print(f"  FeatureDriftMonitor: OK")
        print(f"    PSI calculation: {psi:.4f}")

    except Exception as e:
        results['FeatureDriftMonitor'] = f'FAIL: {e}'
        print(f"  FeatureDriftMonitor: FAIL - {e}")

    return results


def test_phase2_ensemble():
    """Test Ensemble Framework."""
    print("\n" + "=" * 60)
    print("PHASE 2: Ensemble Framework")
    print("=" * 60)

    results = {}

    # Test BaseModel interface
    try:
        from src.models.base_model import BaseModel
        results['BaseModel'] = 'OK'
        print("  BaseModel interface: OK")
    except Exception as e:
        results['BaseModel'] = f'FAIL: {e}'
        print(f"  BaseModel: FAIL - {e}")

    # Generate sample data for model testing (as DataFrames)
    np.random.seed(42)
    n_samples = 500
    n_features = 20

    feature_names = [f'feature_{i}' for i in range(n_features)]
    X = pd.DataFrame(np.random.randn(n_samples, n_features), columns=feature_names)
    y = pd.Series((X['feature_0'] + X['feature_1'] * 0.5 + np.random.randn(n_samples) * 0.5 > 0).astype(int))

    X_train, X_val = X.iloc[:400], X.iloc[400:]
    y_train, y_val = y.iloc[:400], y.iloc[400:]

    # Test XGBoost model (uses params dict, not individual args)
    try:
        from src.models.xgb_model import XGBSpreadModel
        model = XGBSpreadModel(params={'n_estimators': 50, 'max_depth': 3})
        model.fit(X_train, y_train, X_val, y_val)

        probs = model.predict_proba(X_val)
        uncertainty = model.get_uncertainty(X_val)

        accuracy = ((probs > 0.5).astype(int) == y_val.values).mean()
        results['XGBSpreadModel'] = f'OK (acc: {accuracy:.2%})'
        print(f"  XGBSpreadModel: OK (val accuracy: {accuracy:.2%})")
    except Exception as e:
        results['XGBSpreadModel'] = f'FAIL: {e}'
        print(f"  XGBSpreadModel: FAIL - {e}")

    # Test LightGBM model
    try:
        from src.models.lgbm_model import LGBMSpreadModel
        model = LGBMSpreadModel(params={'n_estimators': 50, 'max_depth': 3})
        model.fit(X_train, y_train, X_val, y_val)

        probs = model.predict_proba(X_val)
        accuracy = ((probs > 0.5).astype(int) == y_val.values).mean()
        results['LGBMSpreadModel'] = f'OK (acc: {accuracy:.2%})'
        print(f"  LGBMSpreadModel: OK (val accuracy: {accuracy:.2%})")
    except Exception as e:
        results['LGBMSpreadModel'] = f'FAIL: {e}'
        print(f"  LGBMSpreadModel: FAIL - {e}")

    # Test CatBoost model
    try:
        from src.models.catboost_model import CatBoostSpreadModel
        if CatBoostSpreadModel is None:
            results['CatBoostSpreadModel'] = 'SKIP (not installed)'
            print("  CatBoostSpreadModel: SKIP (CatBoost not installed)")
        else:
            model = CatBoostSpreadModel(params={'iterations': 50, 'depth': 3})
            model.fit(X_train, y_train, X_val, y_val)

            probs = model.predict_proba(X_val)
            accuracy = ((probs > 0.5).astype(int) == y_val.values).mean()
            results['CatBoostSpreadModel'] = f'OK (acc: {accuracy:.2%})'
            print(f"  CatBoostSpreadModel: OK (val accuracy: {accuracy:.2%})")
    except Exception as e:
        error_msg = str(e)
        if 'CatBoost required' in error_msg or 'catboost' in error_msg.lower():
            results['CatBoostSpreadModel'] = 'SKIP (not installed)'
            print("  CatBoostSpreadModel: SKIP (CatBoost not installed)")
        else:
            results['CatBoostSpreadModel'] = f'FAIL: {e}'
            print(f"  CatBoostSpreadModel: FAIL - {e}")

    # Test Neural model
    try:
        from src.models.neural_model import NeuralSpreadModel
        if NeuralSpreadModel is None:
            results['NeuralSpreadModel'] = 'SKIP (PyTorch not installed)'
            print("  NeuralSpreadModel: SKIP (PyTorch not installed)")
        else:
            model = NeuralSpreadModel()
            model.fit(X_train, y_train, X_val, y_val)

            probs = model.predict_proba(X_val)
            accuracy = ((probs > 0.5).astype(int) == y_val.values).mean()
            results['NeuralSpreadModel'] = f'OK (acc: {accuracy:.2%})'
            print(f"  NeuralSpreadModel: OK (val accuracy: {accuracy:.2%})")
    except Exception as e:
        error_msg = str(e)
        if 'PyTorch required' in error_msg or 'torch' in error_msg.lower():
            results['NeuralSpreadModel'] = 'SKIP (not installed)'
            print("  NeuralSpreadModel: SKIP (PyTorch not installed)")
        else:
            results['NeuralSpreadModel'] = f'FAIL: {e}'
            print(f"  NeuralSpreadModel: FAIL - {e}")

    # Test Ensemble
    try:
        from src.models.ensemble import EnsembleModel
        from src.models.xgb_model import XGBSpreadModel
        from src.models.lgbm_model import LGBMSpreadModel

        models = [
            XGBSpreadModel(params={'n_estimators': 50, 'max_depth': 3}),
            LGBMSpreadModel(params={'n_estimators': 50, 'max_depth': 3}),
        ]

        ensemble = EnsembleModel(models, weighting='equal')
        ensemble.fit(X_train, y_train, X_val, y_val)

        probs = ensemble.predict_proba(X_val)
        uncertainty = ensemble.get_uncertainty(X_val)

        accuracy = ((probs > 0.5).astype(int) == y_val.values).mean()
        results['EnsembleModel'] = f'OK (acc: {accuracy:.2%})'
        print(f"  EnsembleModel: OK (val accuracy: {accuracy:.2%})")
        print(f"    Avg uncertainty: {uncertainty.mean():.4f}")

    except Exception as e:
        results['EnsembleModel'] = f'FAIL: {e}'
        print(f"  EnsembleModel: FAIL - {e}")

    return results


def test_phase3_uncertainty():
    """Test Uncertainty Quantification."""
    print("\n" + "=" * 60)
    print("PHASE 3: Uncertainty Quantification")
    print("=" * 60)

    results = {}

    # Generate sample data as DataFrames
    np.random.seed(42)
    n_samples = 500
    feature_names = [f'feature_{i}' for i in range(10)]
    X = pd.DataFrame(np.random.randn(n_samples, 10), columns=feature_names)
    y = pd.Series(X['feature_0'] + X['feature_1'] * 0.5 + np.random.randn(n_samples) * 2)

    X_train, X_cal, X_test = X.iloc[:300], X.iloc[300:400], X.iloc[400:]
    y_train, y_cal, y_test = y.iloc[:300], y.iloc[300:400], y.iloc[400:]

    # ARCHIVED: ConformalPredictor tests (see tests/archive/test_archived_models.py)
    # Model archived to archive/unused_models/conformal.py on 2026-01-09
    results['ConformalPredictor'] = 'SKIPPED (archived)'
    print(f"  ConformalPredictor: SKIPPED (archived)")

    # ARCHIVED: AdaptiveConformalPredictor tests (see tests/archive/test_archived_models.py)
    # Model archived to archive/unused_models/conformal.py on 2026-01-09
    results['AdaptiveConformalPredictor'] = 'SKIPPED (archived)'
    print(f"  AdaptiveConformalPredictor: SKIPPED (archived)")

    # Test Kelly bet sizer
    try:
        from src.betting.kelly import KellyBetSizer

        kelly = KellyBetSizer(fraction=0.2)

        # Standard Kelly (method is calculate_kelly, not calculate)
        std_size = kelly.calculate_kelly(win_prob=0.55, odds=2.0, odds_format='decimal')

        # Uncertainty-adjusted Kelly (method is calculate_uncertainty_adjusted_kelly, uses prob_uncertainty not prob_std)
        if hasattr(kelly, 'calculate_uncertainty_adjusted_kelly'):
            adj_result = kelly.calculate_uncertainty_adjusted_kelly(
                win_prob=0.55,
                prob_uncertainty=0.05,
                odds=2.0,
                odds_format='decimal'
            )
            results['KellyBetSizer'] = f'OK (std: {std_size:.2%}, adj: {adj_result.adjusted_fraction:.2%})'
            print(f"  KellyBetSizer: OK")
            print(f"    Standard size: {std_size:.2%}")
            print(f"    Adjusted size: {adj_result.adjusted_fraction:.2%}")
            print(f"    Confidence factor: {adj_result.confidence_factor:.2f}")
        else:
            results['KellyBetSizer'] = f'OK (std: {std_size:.2%})'
            print(f"  KellyBetSizer: OK")
            print(f"    Standard size: {std_size:.2%}")

    except Exception as e:
        results['KellyBetSizer'] = f'FAIL: {e}'
        print(f"  KellyBetSizer: FAIL - {e}")

    return results


def test_phase4_regime():
    """Test Regime Detection."""
    print("\n" + "=" * 60)
    print("PHASE 4: Regime Detection")
    print("=" * 60)

    results = {}

    # Test RegimeDetector
    try:
        from src.monitoring.regime_detection import RegimeDetector, Regime

        detector = RegimeDetector(
            clv_threshold=0.0,
            win_rate_threshold=0.524,
            min_samples_for_detection=30,
        )

        # Create sample metric series with change point
        np.random.seed(42)
        series1 = np.random.normal(0.02, 0.05, 50)  # Positive CLV
        series2 = np.random.normal(-0.01, 0.05, 50)  # Negative CLV
        metric_series = np.concatenate([series1, series2])

        # Detect change points
        changepoints = detector.detect_changepoints(metric_series)

        results['RegimeDetector'] = f'OK (found {len(changepoints)} changepoints)'
        print(f"  RegimeDetector: OK")
        print(f"    Changepoints found: {[cp.index for cp in changepoints]}")

        # Add performance samples and get current regime
        for i, clv in enumerate(metric_series[-40:]):
            win = np.random.random() > 0.45
            roi = 0.1 if win else -0.1
            detector.add_performance_sample(clv, win, roi)

        regime = detector.get_current_regime()  # No arguments
        print(f"    Current regime: {regime.value}")

    except Exception as e:
        results['RegimeDetector'] = f'FAIL: {e}'
        print(f"  RegimeDetector: FAIL - {e}")

    # ARCHIVED: OnlineUpdater tests (see tests/archive/test_archived_models.py)
    # Model archived to archive/unused_models/online_learning.py on 2026-01-09
    results['OnlineUpdater'] = 'SKIPPED (archived)'
    print(f"  OnlineUpdater: SKIPPED (archived)")

    return results


def test_phase5_player():
    """Test Player-Level Modeling."""
    print("\n" + "=" * 60)
    print("PHASE 5: Player-Level Modeling")
    print("=" * 60)

    results = {}

    # Test PlayerImpactModel
    try:
        from src.features.player_impact import PlayerImpactModel

        model = PlayerImpactModel()

        # Create synthetic stint data
        np.random.seed(42)
        n_stints = 200
        n_players = 50

        stint_data = []
        for i in range(n_stints):
            # Random 5 players on each team
            home_players = np.random.choice(range(1, n_players//2), 5, replace=False)
            away_players = np.random.choice(range(n_players//2, n_players), 5, replace=False)

            # Random point differential
            pt_diff = np.random.normal(0, 5)
            duration = np.random.uniform(1, 5)

            stint_data.append({
                'stint_id': i,
                'home_players': list(home_players),
                'away_players': list(away_players),
                'pt_diff': pt_diff,
                'duration': duration,
            })

        # For now, just test that the model can be created
        results['PlayerImpactModel'] = 'OK (model created)'
        print(f"  PlayerImpactModel: OK")
        print(f"    (Requires stint data to train)")

    except Exception as e:
        results['PlayerImpactModel'] = f'FAIL: {e}'
        print(f"  PlayerImpactModel: FAIL - {e}")

    # Test LineupFeatureBuilder
    try:
        from src.features.lineup_features import LineupFeatureBuilder, LineupImpact

        builder = LineupFeatureBuilder()

        # Test without model (should return zeros)
        features = builder.build_game_features(
            home_team_id=1610612747,  # Lakers
            away_team_id=1610612738,  # Celtics
        )

        results['LineupFeatureBuilder'] = f'OK ({len(features)} features)'
        print(f"  LineupFeatureBuilder: OK")
        print(f"    Generated {len(features)} features")

    except Exception as e:
        results['LineupFeatureBuilder'] = f'FAIL: {e}'
        print(f"  LineupFeatureBuilder: FAIL - {e}")

    # Test TrainedInjuryModel
    try:
        from src.models.injury_adjustment import TrainedInjuryModel, InjuryCalibration

        model = TrainedInjuryModel()

        # Create synthetic training data
        np.random.seed(42)
        training_data = []
        for i in range(200):
            home_impact = np.random.exponential(2) if np.random.random() > 0.6 else 0
            away_impact = np.random.exponential(2) if np.random.random() > 0.6 else 0

            injury_diff = away_impact - home_impact
            true_effect = injury_diff * 0.5
            point_diff = np.random.normal(3 + true_effect, 12)

            training_data.append({
                'game_id': f'game_{i}',
                'home_team_id': 1,
                'away_team_id': 2,
                'point_diff': point_diff,
                'home_won': 1 if point_diff > 0 else 0,
                'home_injury_impact': home_impact,
                'away_injury_impact': away_impact,
                'injury_diff': injury_diff,
                'home_star_out': 0,
                'away_star_out': 0,
                'star_diff': 0,
            })

        training_df = pd.DataFrame(training_data)
        calibration = model.fit(training_df)

        results['TrainedInjuryModel'] = f'OK (impact_to_spread: {calibration.impact_to_spread:.3f})'
        print(f"  TrainedInjuryModel: OK")
        print(f"    Learned impact_to_spread: {calibration.impact_to_spread:.3f}")
        print(f"    Validation MAE: {calibration.validation_mae:.2f}")

    except Exception as e:
        results['TrainedInjuryModel'] = f'FAIL: {e}'
        print(f"  TrainedInjuryModel: FAIL - {e}")

    return results


def run_backtest_comparison():
    """Run backtest comparison between old and new system."""
    print("\n" + "=" * 60)
    print("BACKTEST COMPARISON")
    print("=" * 60)

    # Check if we have the required data
    games_path = Path("data/raw/games.parquet")
    if not games_path.exists():
        print("  No games data available for backtesting")
        print("  Run: python -c 'from src.data import NBAStatsClient; ...' to fetch data")
        return None

    try:
        games_df = pd.read_parquet(games_path)
        print(f"  Loaded {len(games_df)} games")
        print(f"  Date range: {games_df['date'].min()} to {games_df['date'].max()}")

        # TODO: Implement full backtest comparison when we have:
        # 1. Trained models
        # 2. Historical odds data
        # 3. Injury data

        print("\n  Full backtest requires:")
        print("    - Trained ensemble models")
        print("    - Historical odds data")
        print("    - Injury data (optional)")
        print("\n  Run scripts/train_models.py first to train models")

        return None

    except Exception as e:
        print(f"  Error loading data: {e}")
        return None


def generate_report(results: Dict[str, Dict[str, str]]):
    """Generate summary report."""
    print("\n" + "=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)

    total_tests = 0
    passed = 0
    failed = 0
    skipped = 0

    for phase, phase_results in results.items():
        print(f"\n{phase}:")
        for component, status in phase_results.items():
            total_tests += 1
            if status.startswith('OK'):
                passed += 1
                symbol = ''
            elif status.startswith('SKIP'):
                skipped += 1
                symbol = ''
            else:
                failed += 1
                symbol = ''
            print(f"  {symbol} {component}: {status}")

    print("\n" + "-" * 60)
    print(f"Total: {total_tests} tests")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {skipped}")

    if failed == 0:
        print("\nAll core components working!")
    else:
        print(f"\n{failed} component(s) need attention")

    return {
        'total': total_tests,
        'passed': passed,
        'failed': failed,
        'skipped': skipped,
    }


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("QUANT SYSTEM INTEGRATION TEST")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    results = {}

    # Run all phase tests
    results['Phase 1: Alpha Monitoring'] = test_phase1_monitoring()
    results['Phase 2: Ensemble Framework'] = test_phase2_ensemble()
    results['Phase 3: Uncertainty'] = test_phase3_uncertainty()
    results['Phase 4: Regime Detection'] = test_phase4_regime()
    results['Phase 5: Player Modeling'] = test_phase5_player()

    # Run backtest comparison
    backtest_results = run_backtest_comparison()

    # Generate summary report
    summary = generate_report(results)

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    summary = main()

    # Exit with error code if any tests failed
    sys.exit(0 if summary['failed'] == 0 else 1)
