"""
Prediction Cache

Caches predictions to disk to avoid repeated API calls and model inference.
"""

import os
import json
from datetime import datetime
from typing import Dict, Optional, List
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


def refresh_predictions(min_edge: float = 0.02, use_pipeline_logic: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Refresh predictions by calling APIs and models, then cache results.

    Args:
        min_edge: Minimum edge threshold (only used if use_pipeline_logic=False)
        use_pipeline_logic: If True, use daily betting pipeline logic with filters.
                           If False, use MultiPredictor for all markets.

    Returns:
        Dictionary of predictions by market type
    """
    if use_pipeline_logic:
        return _refresh_with_pipeline_logic()
    else:
        return _refresh_with_multipredictor(min_edge)


def _refresh_with_pipeline_logic() -> Dict[str, pd.DataFrame]:
    """Refresh predictions using the same logic as daily_betting_pipeline.py"""
    logger.info("Refreshing predictions using pipeline logic...")

    try:
        # Import pipeline functions
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))

        # Import the actual pipeline module to reuse its logic
        from scripts.daily_betting_pipeline import get_todays_games, evaluate_games, fetch_current_odds
        from src.betting.edge_strategy import EdgeStrategy

        # Get today's games with predictions
        games_df = get_todays_games()

        if games_df.empty:
            logger.warning("No games available from pipeline")
            return {}

        # Fetch current odds and merge with games
        current_odds = fetch_current_odds(games_df['game_id'].tolist())

        if current_odds.empty:
            logger.warning("No odds available")
            return {}

        # Merge odds with games (same as pipeline does)
        games_df = games_df.merge(
            current_odds[['game_id', 'market_spread', 'home_odds', 'away_odds', 'bookmaker']],
            on='game_id',
            how='left'
        )

        # Calculate model edge
        games_df['model_edge'] = games_df['pred_diff'] + games_df['market_spread']

        # For line shopping, get all odds
        from src.data.odds_api import OddsAPIClient
        odds_client = OddsAPIClient()
        all_odds = odds_client.get_current_odds(markets=["h2h", "spreads", "totals"])

        # Initialize strategy (same as pipeline - use the class method)
        strategy = EdgeStrategy.clv_filtered_strategy()

        # Evaluate games with strategy
        signals = evaluate_games(games_df, strategy)

        # Convert signals to DataFrame format
        predictions = _convert_signals_to_predictions(signals, games_df, all_odds)

        # Save to cache
        save_predictions_to_cache(predictions)

        logger.info(f"Cached {len(predictions)} market predictions from pipeline")
        return predictions

    except Exception as e:
        logger.error(f"Error refreshing with pipeline logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}


def _refresh_with_multipredictor(min_edge: float) -> Dict[str, pd.DataFrame]:
    """Refresh predictions using MultiPredictor (original logic)."""
    from src.multi_predictions import MultiPredictor

    logger.info("Refreshing predictions with MultiPredictor...")

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


def _convert_signals_to_predictions(signals: List, games_df: pd.DataFrame, all_odds: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Convert BetSignal objects to prediction DataFrames for caching."""
    if not signals:
        return {}

    # Build a predictions DataFrame from signals
    predictions_data = []

    for signal in signals:
        # Skip PASS signals (no bet)
        if signal.bet_side == 'PASS':
            continue

        # Find the corresponding game
        game = games_df[games_df['game_id'] == signal.game_id]
        if game.empty:
            continue

        game_row = game.iloc[0]

        # Get odds for this game
        game_odds = all_odds[all_odds['game_id'] == signal.game_id] if not all_odds.empty else pd.DataFrame()

        # Get odds for this bet
        odds_value = game_row.get('home_odds', -110) if signal.bet_side == 'HOME' else game_row.get('away_odds', -110)

        # Calculate Kelly criterion
        # Convert American odds to decimal probability
        if odds_value < 0:
            implied_prob = abs(odds_value) / (abs(odds_value) + 100)
        else:
            implied_prob = 100 / (odds_value + 100)

        # Our edge is the model edge vs market spread
        edge_points = abs(game_row.get('model_edge', 0))

        # Estimate win probability from edge (rough approximation: each point = ~2% edge)
        our_win_prob = 0.5 + (edge_points * 0.02)
        our_win_prob = max(0.0, min(1.0, our_win_prob))  # Clamp to [0, 1]

        # Kelly = (bp - q) / b where b = decimal odds - 1, p = our prob, q = 1 - p
        decimal_odds = (100 / abs(odds_value)) if odds_value < 0 else (odds_value / 100)
        kelly = max(0.0, (decimal_odds * our_win_prob - (1 - our_win_prob)) / decimal_odds)

        pred = {
            'game_id': signal.game_id,
            'commence_time': game_row.get('commence_time'),
            'home_team': signal.home_team,
            'away_team': signal.away_team,
            'bet_side': signal.bet_side,  # Already uppercase HOME or AWAY
            'bet_home': signal.bet_side == 'HOME',
            'bet_away': signal.bet_side == 'AWAY',
            'line': game_row.get('market_spread', 0),
            'odds': odds_value,
            'edge_vs_market': edge_points,
            'kelly': kelly,
            'confidence': signal.confidence,
            'model_edge': game_row.get('model_edge', 0),
            'pred_diff': game_row.get('pred_diff', 0),
        }

        predictions_data.append(pred)

    # Create DataFrame
    if predictions_data:
        df = pd.DataFrame(predictions_data)
        return {
            'pipeline': df,  # Main pipeline picks
            'spread': df,    # Also categorize as spread
        }

    return {}


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
