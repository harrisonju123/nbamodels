"""
Prediction Cache

Caches predictions to disk to avoid repeated API calls and model inference.
"""

import os
import json
from datetime import datetime
from typing import Dict, Optional
import pandas as pd
from loguru import logger


CACHE_DIR = "data/cache"
CACHE_FILE = os.path.join(CACHE_DIR, "predictions_cache.json")


def get_cache_info() -> Optional[Dict]:
    """Get information about the current cache."""
    if not os.path.exists(CACHE_FILE):
        return None

    try:
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
        return {
            "timestamp": cache.get("timestamp"),
            "markets": list(cache.get("predictions", {}).keys()),
            "num_games": cache.get("num_games", 0),
        }
    except Exception as e:
        logger.error(f"Error reading cache: {e}")
        return None


def load_cached_predictions() -> Optional[Dict[str, pd.DataFrame]]:
    """Load predictions from cache file."""
    if not os.path.exists(CACHE_FILE):
        logger.info("No cache file found")
        return None

    try:
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)

        timestamp = cache.get("timestamp")
        logger.info(f"Loading cached predictions from {timestamp}")

        predictions = {}
        for market, data in cache.get("predictions", {}).items():
            if data:  # data is a list/dict here, not a DataFrame, so this is safe
                df = pd.DataFrame(data)
                if not df.empty:
                    predictions[market] = df

        return predictions
    except Exception as e:
        logger.error(f"Error loading cache: {e}")
        return None


def save_predictions_to_cache(predictions: Dict[str, pd.DataFrame]) -> bool:
    """Save predictions to cache file."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)

        # Convert DataFrames to JSON-serializable format
        cache_data = {
            "timestamp": datetime.now().isoformat(),
            "num_games": 0,
            "predictions": {}
        }

        for market, df in predictions.items():
            if df is not None and not df.empty:
                # Convert datetime columns to strings
                df_copy = df.copy()
                for col in df_copy.columns:
                    if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                        df_copy[col] = df_copy[col].astype(str)

                cache_data["predictions"][market] = df_copy.to_dict(orient='records')
                cache_data["num_games"] = max(cache_data["num_games"], len(df))

        with open(CACHE_FILE, 'w') as f:
            json.dump(cache_data, f, indent=2, default=str)

        logger.info(f"Saved predictions to cache: {len(cache_data['predictions'])} markets")
        return True
    except Exception as e:
        logger.error(f"Error saving cache: {e}")
        return False


def refresh_predictions(min_edge: float = 0.02) -> Dict[str, pd.DataFrame]:
    """Refresh predictions by calling APIs and models, then cache results."""
    from src.multi_predictions import MultiPredictor

    logger.info("Refreshing predictions from APIs...")

    predictor = MultiPredictor(min_edge=min_edge)

    # Get recent games for features
    recent_games = predictor.get_recent_games(days=60)

    # Get current odds
    odds = predictor.get_current_odds()

    if odds.empty:
        logger.warning("No odds available")
        return {}

    # Build features
    features = predictor.build_features_for_upcoming(recent_games, odds)

    if features.empty:
        logger.warning("No features built")
        return {}

    # Generate predictions for all markets
    predictions = predictor.generate_all_predictions(features, odds)

    # Save to cache
    save_predictions_to_cache(predictions)

    return predictions


def clear_cache():
    """Clear the prediction cache."""
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
        logger.info("Cache cleared")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv('.env')

    # Test refresh
    print("Refreshing predictions...")
    predictions = refresh_predictions()

    print(f"\nCached {len(predictions)} markets")
    for market, df in predictions.items():
        print(f"  {market}: {len(df)} games")

    # Test load
    print("\nLoading from cache...")
    loaded = load_cached_predictions()
    print(f"Loaded {len(loaded)} markets")

    # Get cache info
    info = get_cache_info()
    print(f"\nCache info: {info}")
