#!/usr/bin/env python3
"""
Monitor Model Drift

Daily script to monitor model calibration and detect performance drift.
Saves calibration snapshots and sends alerts if drift is detected.

Usage:
    python scripts/monitor_model_drift.py
    python scripts/monitor_model_drift.py --model-type spread --strategy-type primary
    python scripts/monitor_model_drift.py --alert-only  # Just check drift, don't save snapshot
"""

import argparse
import sys
from datetime import datetime
from loguru import logger

from src.market_analysis.model_drift import ModelDriftMonitor


def main():
    parser = argparse.ArgumentParser(description="Monitor model drift")
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        help="Model type to monitor (spread, totals, moneyline)",
    )
    parser.add_argument(
        "--strategy-type",
        type=str,
        default=None,
        help="Strategy type to monitor",
    )
    parser.add_argument(
        "--alert-only",
        action="store_true",
        help="Only check for drift, don't save snapshot",
    )
    parser.add_argument(
        "--baseline-window",
        type=int,
        default=200,
        help="Number of bets for baseline (default: 200)",
    )
    parser.add_argument(
        "--alert-window",
        type=int,
        default=50,
        help="Number of recent bets to check (default: 50)",
    )

    args = parser.parse_args()

    logger.info("Starting model drift monitoring")
    logger.info(f"Baseline window: {args.baseline_window} bets")
    logger.info(f"Alert window: {args.alert_window} bets")

    monitor = ModelDriftMonitor(
        baseline_window=args.baseline_window,
        alert_window=args.alert_window,
    )

    # Determine which models/strategies to monitor
    models_to_monitor = []

    if args.model_type and args.strategy_type:
        models_to_monitor.append((args.model_type, args.strategy_type))
    elif args.model_type:
        # Monitor all strategies for this model type
        models_to_monitor.append((args.model_type, None))
    else:
        # Monitor all common combinations
        models_to_monitor.extend([
            ('spread', 'primary'),
            ('totals', 'primary'),
        ])

    # Save snapshots first (if not alert-only)
    if not args.alert_only:
        logger.info("Saving calibration snapshots...")
        for model_type, strategy_type in models_to_monitor:
            if strategy_type:
                try:
                    snapshot = monitor.save_calibration_snapshot(
                        model_type=model_type,
                        strategy_type=strategy_type,
                    )
                    logger.success(
                        f"[{model_type}/{strategy_type}] "
                        f"Brier: {snapshot.brier_score:.4f}, "
                        f"ECE: {snapshot.ece:.4f}, "
                        f"Accuracy: {snapshot.accuracy:.3f}, "
                        f"N: {snapshot.n_samples}"
                    )
                except Exception as e:
                    logger.error(f"Failed to save snapshot for {model_type}/{strategy_type}: {e}")

    # Check for drift
    logger.info("Checking for model drift...")
    any_alerts = False

    for model_type, strategy_type in models_to_monitor:
        try:
            alerts = monitor.detect_calibration_drift(
                model_type=model_type if model_type else None,
                strategy_type=strategy_type if strategy_type else None,
            )

            if alerts:
                any_alerts = True
                label = f"{model_type or 'all'}/{strategy_type or 'all'}"
                logger.warning(f"\n{'='*60}")
                logger.warning(f"DRIFT DETECTED: {label}")
                logger.warning(f"{'='*60}")

                for alert in alerts:
                    severity_emoji = "🔴" if alert.severity == "critical" else "🟡"
                    logger.warning(
                        f"\n{severity_emoji} {alert.severity.upper()}: {alert.metric}"
                    )
                    logger.warning(f"  Current: {alert.current_value:.4f}")
                    logger.warning(f"  Baseline: {alert.baseline_value:.4f}")
                    logger.warning(f"  Change: {alert.change_pct:+.2f}%")
                    logger.warning(f"  Recommendation: {alert.recommendation}")

                logger.warning(f"{'='*60}\n")
            else:
                label = f"{model_type or 'all'}/{strategy_type or 'all'}"
                logger.success(f"✓ [{label}] No drift detected")

        except Exception as e:
            logger.error(f"Failed to check drift for {model_type}/{strategy_type}: {e}")

    if any_alerts:
        logger.warning("\n⚠️  DRIFT ALERTS DETECTED - Review model performance immediately")
        sys.exit(1)
    else:
        logger.success("\n✓ All models performing within expected parameters")
        sys.exit(0)


if __name__ == "__main__":
    main()
