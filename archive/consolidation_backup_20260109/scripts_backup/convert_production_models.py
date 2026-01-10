#!/usr/bin/env python3
"""
Convert production XGBoost models to BasePlayerPropModel format.

The production models were saved as raw XGBoost pickle files, but
PlayerPropsStrategy expects them wrapped in BasePlayerPropModel format
with metadata (feature_columns, calibrated_std, etc.).
"""

import os
import sys
import json
import pickle
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.player_props import (
    PointsPropModel,
    ReboundsPropModel,
    AssistsPropModel,
    ThreesPropModel,
)
from loguru import logger


def convert_model(prop_type: str, production_dir: str, output_dir: str):
    """
    Convert a raw XGBoost model to BasePlayerPropModel format.

    Args:
        prop_type: Prop type (pts, reb, ast, fg3m)
        production_dir: Directory with production models
        output_dir: Directory to save converted models
    """
    # Map prop types to model classes
    model_classes = {
        "pts": PointsPropModel,
        "reb": ReboundsPropModel,
        "ast": AssistsPropModel,
        "fg3m": ThreesPropModel,
    }

    if prop_type not in model_classes:
        logger.error(f"Unknown prop type: {prop_type}")
        return False

    # Load raw XGBoost model
    model_path = os.path.join(production_dir, f"{prop_type}_model.pkl")
    metadata_path = os.path.join(production_dir, f"{prop_type}_metadata.json")

    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return False

    if not os.path.exists(metadata_path):
        logger.error(f"Metadata not found: {metadata_path}")
        return False

    # Load raw model
    logger.info(f"Loading raw XGBoost model from {model_path}")
    with open(model_path, 'rb') as f:
        xgb_model = pickle.load(f)

    # Load metadata
    logger.info(f"Loading metadata from {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Create BasePlayerPropModel instance
    model_class = model_classes[prop_type]
    prop_model = model_class()

    # Set model attributes from production model
    prop_model.model = xgb_model
    prop_model.feature_columns = metadata['features']

    # Get feature importance from metadata
    if 'top_features' in metadata:
        import pandas as pd
        prop_model.feature_importance = pd.DataFrame([
            {"feature": f['feature'], "importance": f['importance']}
            for f in metadata['top_features']
        ])

    # Estimate calibrated_std from in-sample RMSE
    # This is an approximation - ideally we'd have validation set errors
    if 'performance' in metadata and 'in_sample_rmse' in metadata['performance']:
        prop_model.calibrated_std = metadata['performance']['in_sample_rmse']
        logger.info(f"Using calibrated_std = {prop_model.calibrated_std:.4f} from in-sample RMSE")
    else:
        # Default fallback
        prop_model.calibrated_std = 1.0
        logger.warning(f"No RMSE found, using default calibrated_std = 1.0")

    # Set params from metadata
    if 'hyperparameters' in metadata:
        # Filter out None values
        prop_model.params = {
            k: v for k, v in metadata['hyperparameters'].items()
            if v is not None
        }

    # Save in BasePlayerPropModel format
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{prop_type}_model.pkl")

    logger.info(f"Saving converted model to {output_path}")
    prop_model.save(output_path)

    # Verify it loads correctly
    logger.info("Verifying model loads correctly...")
    loaded_model = model_class.load(output_path)
    logger.info(f"✅ Successfully converted {prop_type.upper()} model")
    logger.info(f"   Features: {len(loaded_model.feature_columns)}")
    logger.info(f"   Calibrated std: {loaded_model.calibrated_std:.4f}")

    return True


def main():
    """Convert all production models."""
    production_dir = "models/player_props_production/latest"
    output_dir = "models/player_props_production/latest_converted"

    logger.info("=" * 60)
    logger.info("Converting Production Models to BasePlayerPropModel Format")
    logger.info("=" * 60)

    prop_types = ["pts", "reb", "ast", "fg3m"]
    success_count = 0

    for prop_type in prop_types:
        logger.info(f"\nConverting {prop_type.upper()}...")
        if convert_model(prop_type, production_dir, output_dir):
            success_count += 1
        else:
            logger.error(f"Failed to convert {prop_type.upper()}")

    logger.info("\n" + "=" * 60)
    logger.info(f"Conversion complete: {success_count}/{len(prop_types)} models converted")
    logger.info("=" * 60)

    if success_count == len(prop_types):
        logger.info("\n✅ All models converted successfully!")
        logger.info(f"   Converted models saved to: {output_dir}")
        logger.info(f"\n   Next step: Update config to use '{output_dir}'")
        return 0
    else:
        logger.error("\n❌ Some models failed to convert")
        return 1


if __name__ == "__main__":
    exit(main())
