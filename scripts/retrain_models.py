"""
Model Retraining Pipeline

Automatically detects new game data and retrains models when needed.
Validates new models before deployment.

Usage:
    python scripts/retrain_models.py              # Check and retrain if needed
    python scripts/retrain_models.py --force      # Force retrain all models
    python scripts/retrain_models.py --dry-run    # Show what would happen
"""

import sys
import os
import argparse
import pickle
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from dotenv import load_dotenv
load_dotenv('.env')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/retrain.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


class RetrainingPipeline:
    """Manages model retraining with validation."""

    # Minimum new games required to trigger retraining
    MIN_NEW_GAMES = 50

    # Model paths
    MODEL_DIR = Path("models")
    DATA_DIR = Path("data")
    BACKUP_DIR = Path("models/backups")

    # Training metadata file
    METADATA_FILE = Path("models/training_metadata.json")

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.metadata = self._load_metadata()

        # Ensure directories exist
        self.BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(exist_ok=True)

    def _load_metadata(self) -> Dict[str, Any]:
        """Load training metadata."""
        if self.METADATA_FILE.exists():
            with open(self.METADATA_FILE) as f:
                return json.load(f)
        return {
            "last_retrain": None,
            "last_game_date": None,
            "games_count": 0,
            "models": {}
        }

    def _save_metadata(self):
        """Save training metadata."""
        with open(self.METADATA_FILE, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)

    def check_new_data(self) -> Tuple[bool, int, Optional[str]]:
        """
        Check if there's enough new data to warrant retraining.

        Returns:
            (should_retrain, new_games_count, latest_game_date)
        """
        games_path = self.DATA_DIR / "raw" / "games.parquet"

        if not games_path.exists():
            logger.warning("No games data found")
            return False, 0, None

        games = pd.read_parquet(games_path)
        current_game_count = len(games)
        latest_game_date = str(games['date'].max())

        previous_count = self.metadata.get("games_count", 0)
        new_games = current_game_count - previous_count

        logger.info(f"Games in dataset: {current_game_count}")
        logger.info(f"Previous count: {previous_count}")
        logger.info(f"New games: {new_games}")
        logger.info(f"Latest game date: {latest_game_date}")

        should_retrain = new_games >= self.MIN_NEW_GAMES

        if should_retrain:
            logger.info(f"Retraining triggered: {new_games} new games >= {self.MIN_NEW_GAMES} threshold")
        else:
            logger.info(f"No retraining needed: {new_games} new games < {self.MIN_NEW_GAMES} threshold")

        return should_retrain, new_games, latest_game_date

    def fetch_new_data(self) -> bool:
        """Fetch latest game data from NBA API."""
        logger.info("Fetching latest game data...")

        if self.dry_run:
            logger.info("[DRY RUN] Would fetch new data")
            return True

        try:
            from src.data import NBAStatsClient

            client = NBAStatsClient()
            current_season = 2024  # Update for current season

            # Fetch current season data
            logger.info(f"Fetching {current_season}-{current_season+1} season...")
            new_games = client.get_season_games(current_season)

            # Load existing data
            games_path = self.DATA_DIR / "raw" / "games.parquet"
            if games_path.exists():
                existing = pd.read_parquet(games_path)
                # Merge and deduplicate
                combined = pd.concat([existing, new_games], ignore_index=True)
                combined = combined.drop_duplicates(subset=['game_id'])
                combined = combined.sort_values('date').reset_index(drop=True)
            else:
                combined = new_games

            # Save updated data
            combined.to_parquet(games_path)
            logger.info(f"Updated games data: {len(combined)} total games")

            return True

        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            return False

    def rebuild_features(self) -> bool:
        """Rebuild feature dataset from game data."""
        logger.info("Rebuilding features...")

        if self.dry_run:
            logger.info("[DRY RUN] Would rebuild features")
            return True

        try:
            from src.features.game_features import GameFeatureBuilder

            builder = GameFeatureBuilder()
            games = pd.read_parquet(self.DATA_DIR / "raw" / "games.parquet")

            # Use build_game_features method
            features = builder.build_game_features(games)

            # Merge target columns from games (needed for point_spread and totals models)
            target_cols = ["game_id", "home_score", "away_score"]
            if all(col in games.columns for col in target_cols):
                targets = games[target_cols].copy()
                features = features.merge(targets, on="game_id", how="left")
                logger.info("Merged score targets from raw games")

            # Save features
            features_path = self.DATA_DIR / "features" / "game_features.parquet"
            features_path.parent.mkdir(parents=True, exist_ok=True)
            features.to_parquet(features_path)

            logger.info(f"Built features: {len(features)} games, {len(features.columns)} features")
            return True

        except Exception as e:
            logger.error(f"Failed to rebuild features: {e}")
            import traceback
            traceback.print_exc()
            return False

    def backup_models(self):
        """Backup current models before retraining."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_subdir = self.BACKUP_DIR / timestamp
        backup_subdir.mkdir(parents=True, exist_ok=True)

        models_to_backup = [
            "dual_model.pkl",
            "spread_model.pkl",
            "point_spread_model.pkl",
            "totals_model.pkl"
        ]

        for model_file in models_to_backup:
            src = self.MODEL_DIR / model_file
            if src.exists():
                dst = backup_subdir / model_file
                if not self.dry_run:
                    import shutil
                    shutil.copy2(src, dst)
                logger.info(f"Backed up: {model_file} -> {backup_subdir.name}/")

        return backup_subdir

    def train_model(self, model_name: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Train a specific model and return metrics.

        Returns:
            (success, metrics_dict)
        """
        logger.info(f"Training {model_name}...")

        if self.dry_run:
            logger.info(f"[DRY RUN] Would train {model_name}")
            return True, {"dry_run": True}

        try:
            if model_name == "dual_model":
                from src.models.dual_model import train_dual_model
                metrics = train_dual_model()

            elif model_name == "spread_model":
                from src.models.spread_model import train_spread_model
                metrics = train_spread_model()

            elif model_name == "point_spread_model":
                from src.models.point_spread import train_point_spread_model
                metrics = train_point_spread_model()

            elif model_name == "totals_model":
                from src.models.totals import train_totals_model
                metrics = train_totals_model()

            else:
                logger.error(f"Unknown model: {model_name}")
                return False, {}

            logger.info(f"Successfully trained {model_name}")
            return True, metrics if metrics else {}

        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return False, {"error": str(e)}

    def validate_model(self, model_name: str, old_model_path: Path) -> Tuple[bool, str]:
        """
        Validate new model against the old one.

        Returns:
            (is_valid, reason)
        """
        new_model_path = self.MODEL_DIR / f"{model_name}.pkl"

        if not new_model_path.exists():
            return False, "New model file not found"

        if not old_model_path.exists():
            logger.info(f"No old model to compare for {model_name}, accepting new model")
            return True, "No previous model"

        try:
            # Load models
            with open(old_model_path, 'rb') as f:
                old_model = pickle.load(f)
            with open(new_model_path, 'rb') as f:
                new_model = pickle.load(f)

            # Basic sanity checks
            # 1. New model file should exist and be loadable
            if new_model is None:
                return False, "New model is None"

            # 2. Model should have expected attributes
            if hasattr(new_model, 'feature_columns'):
                if len(new_model.feature_columns) < 10:
                    return False, f"Too few features: {len(new_model.feature_columns)}"

            # 3. For regression models, check calibrated_std is reasonable
            if hasattr(new_model, 'calibrated_std'):
                if new_model.calibrated_std < 5 or new_model.calibrated_std > 25:
                    return False, f"Unusual calibrated_std: {new_model.calibrated_std}"

            logger.info(f"Model {model_name} passed validation")
            return True, "Passed all checks"

        except Exception as e:
            return False, f"Validation error: {e}"

    def run(self, force: bool = False) -> bool:
        """
        Run the full retraining pipeline.

        Args:
            force: Force retraining regardless of new data

        Returns:
            True if successful
        """
        logger.info("=" * 60)
        logger.info("Starting Model Retraining Pipeline")
        logger.info(f"Timestamp: {datetime.now()}")
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")
        logger.info("=" * 60)

        # Step 1: Check for new data
        should_retrain, new_games, latest_date = self.check_new_data()

        if not force and not should_retrain:
            logger.info("No retraining needed. Exiting.")
            return True

        if force:
            logger.info("Force retraining requested")

        # Step 2: Fetch latest data
        if not self.fetch_new_data():
            logger.error("Data fetch failed")
            return False

        # Step 3: Rebuild features
        if not self.rebuild_features():
            logger.error("Feature rebuild failed")
            return False

        # Step 4: Backup existing models
        backup_dir = self.backup_models()

        # Step 5: Train all models
        models_to_train = [
            "dual_model",
            "spread_model",
            "point_spread_model",
            "totals_model"
        ]

        results = {}
        all_success = True

        for model_name in models_to_train:
            success, metrics = self.train_model(model_name)
            results[model_name] = {"success": success, "metrics": metrics}

            if success and not self.dry_run:
                # Validate the new model
                old_model_path = backup_dir / f"{model_name}.pkl"
                is_valid, reason = self.validate_model(model_name, old_model_path)
                results[model_name]["valid"] = is_valid
                results[model_name]["validation_reason"] = reason

                if not is_valid:
                    logger.warning(f"Model {model_name} failed validation: {reason}")
                    # Restore from backup
                    if old_model_path.exists():
                        import shutil
                        shutil.copy2(old_model_path, self.MODEL_DIR / f"{model_name}.pkl")
                        logger.info(f"Restored {model_name} from backup")
                    all_success = False
            elif not success:
                all_success = False

        # Step 6: Update metadata
        if all_success and not self.dry_run:
            games = pd.read_parquet(self.DATA_DIR / "raw" / "games.parquet")
            self.metadata["last_retrain"] = datetime.now().isoformat()
            self.metadata["last_game_date"] = str(games['date'].max())
            self.metadata["games_count"] = len(games)
            self.metadata["models"] = results
            self._save_metadata()

        # Summary
        logger.info("=" * 60)
        logger.info("Retraining Pipeline Complete")
        logger.info("=" * 60)
        for model_name, result in results.items():
            status = "OK" if result.get("success") and result.get("valid", True) else "FAILED"
            logger.info(f"  {model_name}: {status}")

        return all_success


def main():
    parser = argparse.ArgumentParser(description="Model Retraining Pipeline")
    parser.add_argument("--force", action="store_true", help="Force retrain all models")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without making changes")
    parser.add_argument("--fetch-only", action="store_true", help="Only fetch new data, don't retrain")
    args = parser.parse_args()

    pipeline = RetrainingPipeline(dry_run=args.dry_run)

    if args.fetch_only:
        pipeline.fetch_new_data()
    else:
        success = pipeline.run(force=args.force)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
