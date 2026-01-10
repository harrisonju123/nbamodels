"""
Train Player Prop Models

Trains XGBoost models for predicting player statistics (PTS, REB, AST, 3PM).
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from loguru import logger
from src.models.player_props import (
    PointsPropModel,
    ReboundsPropModel,
    AssistsPropModel,
    ThreesPropModel,
)


def load_player_features(features_path: str = "data/features/player_game_features.parquet") -> pd.DataFrame:
    """
    Load player-level game features.

    Expected columns:
    - player_id, player_name, team
    - game_id, game_date
    - Target stats: pts, reb, ast, fg3m, stl, blk
    - Rolling features: pts_roll3, pts_roll5, min_roll5, etc.
    - Matchup features: opp_def_rating, opp_pace, etc.
    - Context: is_home, days_rest, etc.
    """
    if not os.path.exists(features_path):
        logger.error(f"Player features not found at {features_path}")
        logger.info("Please create player features first using player feature builder")
        return pd.DataFrame()

    df = pd.read_parquet(features_path)
    logger.info(f"Loaded {len(df)} player games from {features_path}")
    return df


def train_prop_model(
    df: pd.DataFrame,
    model_class,
    target_col: str,
    output_path: str,
    val_split: float = 0.2,
):
    """
    Train a single prop model.

    Args:
        df: Features DataFrame
        model_class: Model class (PointsPropModel, etc.)
        target_col: Target column name (pts, reb, ast, fg3m)
        output_path: Where to save trained model
        val_split: Validation set fraction
    """
    logger.info(f"Training {model_class.prop_type} model...")

    # Filter to players with sufficient data
    # Require at least 10 games and 10+ min/game average
    df_filtered = df[
        (df['min_roll5'] >= 10.0) &  # At least 10 min/game
        (df[target_col].notna())     # Target exists
    ].copy()

    logger.info(f"Filtered to {len(df_filtered)} samples ({len(df)} original)")

    if len(df_filtered) < 100:
        logger.error(f"Insufficient data for {model_class.prop_type} model")
        return None

    # Sort by date for temporal split
    df_filtered = df_filtered.sort_values('game_date').reset_index(drop=True)

    # Split train/val
    split_idx = int(len(df_filtered) * (1 - val_split))
    train_df = df_filtered[:split_idx]
    val_df = df_filtered[split_idx:]

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # Get features
    model = model_class()
    feature_cols = model.get_required_features()

    # Check feature availability
    missing_features = [f for f in feature_cols if f not in df_filtered.columns]
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")
        # Filter to available features
        feature_cols = [f for f in feature_cols if f in df_filtered.columns]

    # Train model
    model.fit(
        train_df[feature_cols],
        train_df[target_col],
        val_df[feature_cols],
        val_df[target_col],
    )

    # Evaluate
    train_metrics = model.evaluate(train_df[feature_cols], train_df[target_col])
    val_metrics = model.evaluate(val_df[feature_cols], val_df[target_col])

    logger.info(f"Train metrics: {train_metrics}")
    logger.info(f"Val metrics: {val_metrics}")

    # Show feature importance
    logger.info(f"Top 10 features:\n{model.feature_importance.head(10)}")

    # Save model
    model.save(output_path)
    logger.success(f"{model_class.prop_type} model saved to {output_path}")

    return model


def main():
    """Train all player prop models."""
    logger.info("=== Training Player Prop Models ===\n")

    # Load features
    features_path = "data/features/player_game_features.parquet"
    df = load_player_features(features_path)

    if df.empty:
        logger.error("No player features available. Cannot train models.")
        logger.info(
            "To create player features, you need to:\n"
            "1. Collect player box scores from NBA API\n"
            "2. Build rolling statistics (pts_roll3, pts_roll5, etc.)\n"
            "3. Add matchup features (opp_def_rating, opp_pace)\n"
            "4. Save to player_game_features.parquet"
        )
        return

    # Create output directory
    output_dir = "models/player_props"
    os.makedirs(output_dir, exist_ok=True)

    # Train each model
    models_config = [
        (PointsPropModel, "pts", "pts_model.pkl"),
        (ReboundsPropModel, "reb", "reb_model.pkl"),
        (AssistsPropModel, "ast", "ast_model.pkl"),
        (ThreesPropModel, "fg3m", "3pm_model.pkl"),
    ]

    trained_models = {}

    for model_class, target_col, filename in models_config:
        output_path = os.path.join(output_dir, filename)

        try:
            model = train_prop_model(
                df=df,
                model_class=model_class,
                target_col=target_col,
                output_path=output_path,
            )

            if model:
                trained_models[model_class.prop_type] = model

        except Exception as e:
            logger.error(f"Error training {model_class.prop_type}: {e}", exc_info=True)

    logger.info(f"\n=== Training Complete ===")
    logger.info(f"Successfully trained {len(trained_models)}/{len(models_config)} models")

    if trained_models:
        logger.success(f"Models saved to {output_dir}")
        logger.info("You can now use PlayerPropsStrategy with these models!")


if __name__ == "__main__":
    main()
