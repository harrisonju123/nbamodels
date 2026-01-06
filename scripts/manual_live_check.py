"""
Manual Live Game Check

Run this script manually to check live games and odds.
Only makes API calls when you run it - no automatic polling.

Usage:
    python -m scripts.manual_live_check
    python -m scripts.manual_live_check --save  # Save to database
"""

import argparse
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.data.live_game_client import LiveGameClient
from src.data.live_odds_tracker import LiveOddsTracker
from src.models.live_win_probability import LiveWinProbModel
from src.betting.live_edge_detector import LiveEdgeDetector

# Set up logging
logger.add(
    "logs/manual_check_{time}.log",
    rotation="1 day",
    retention="30 days",
    level="INFO"
)


def main():
    """Manual check of live games."""
    parser = argparse.ArgumentParser(description="Manually check live NBA games")

    parser.add_argument(
        '--save',
        action='store_true',
        help="Save data to database (default: just display)"
    )

    parser.add_argument(
        '--min-edge',
        type=float,
        default=0.05,
        help="Minimum edge threshold (default 5%%)"
    )

    args = parser.parse_args()

    # Initialize components
    game_client = LiveGameClient()
    odds_tracker = LiveOddsTracker()
    win_prob_model = LiveWinProbModel()
    edge_detector = LiveEdgeDetector(min_edge=args.min_edge)

    print("\n" + "=" * 60)
    print(f"📊 Live Game Check - {datetime.now().strftime('%I:%M:%S %p')}")
    print("=" * 60)

    # Get live games
    live_games = game_client.get_live_games()

    if live_games.empty:
        print("\n⚠️  No live games currently")
        return

    print(f"\n🏀 Found {len(live_games)} live game(s)")

    # Track API usage
    total_api_calls = 0

    # Process each game
    for idx, game in live_games.iterrows():
        game_dict = game.to_dict()
        game_id = game_dict['game_id']
        home_team = game_dict['home_team']
        away_team = game_dict['away_team']

        print(f"\n{'─' * 60}")
        print(f"📊 {away_team} @ {home_team}")
        print(f"   Score: {game_dict['away_score']}-{game_dict['home_score']}")
        print(f"   Q{game_dict['period']} {game_dict.get('game_clock', '')}")

        # Save game state if requested
        if args.save:
            game_client.save_game_state(game_dict)
            print(f"   ✓ Saved game state")

        # Get live odds
        print(f"   Fetching odds...")
        odds = odds_tracker.get_live_odds(teams=[(home_team, away_team)])
        total_api_calls += 1

        if odds.empty:
            print(f"   ⚠️  No odds available")
            continue

        # Save odds if requested
        if args.save:
            saved = odds_tracker.save_odds_snapshot(odds)
            print(f"   ✓ Saved {saved} odds records")

        # Get odds for display
        odds_game_id = odds['game_id'].iloc[0] if not odds.empty else None
        if not odds_game_id:
            print(f"   ⚠️  Could not determine odds game_id")
            continue

        latest_odds = odds_tracker.get_latest_odds(odds_game_id)

        if not latest_odds:
            print(f"   ⚠️  Could not parse odds")
            continue

        # Display odds
        if 'spread' in latest_odds:
            spread = latest_odds['spread']
            print(f"   📈 Spread: {spread['spread_value']:+.1f} ({spread['bookmaker']})")
            print(f"      Home: {spread['home_odds']:+d}, Away: {spread['away_odds']:+d}")

        if 'moneyline' in latest_odds:
            ml = latest_odds['moneyline']
            print(f"   💰 Moneyline: ({ml['bookmaker']})")
            print(f"      Home: {ml['home_odds']:+d}, Away: {ml['away_odds']:+d}")

        if 'total' in latest_odds:
            total = latest_odds['total']
            print(f"   🎯 Total: {total['total_value']:.1f} ({total['bookmaker']})")
            print(f"      Over: {total['over_odds']:+d}, Under: {total['under_odds']:+d}")

        # Calculate win probability
        win_prob = win_prob_model.predict(game_dict)
        print(f"\n   🧮 Win Probability:")
        print(f"      Home: {win_prob['home_win_prob']:.1%}")
        print(f"      Away: {win_prob['away_win_prob']:.1%}")
        print(f"      Confidence: {win_prob['confidence']:.1%}")

        # Detect edges
        edges = edge_detector.find_edges(game_dict, latest_odds, win_prob)

        if edges:
            print(f"\n   🎯 EDGE(S) DETECTED!")
            for edge in edges:
                print(f"\n      {edge['alert_type'].upper()} - {edge['bet_side'].upper()}")
                print(f"      Edge: {edge['edge']:.1%}")
                print(f"      Confidence: {edge['confidence']}")
                print(f"      Model: {edge['model_prob']:.1%}, Market: {edge['market_prob']:.1%}")
                if edge.get('odds'):
                    print(f"      Odds: {edge['odds']:+d}")
                if edge.get('line_value') is not None:
                    print(f"      Line: {edge['line_value']:+.1f}")

                # Save alert if requested
                if args.save:
                    alert_id = edge_detector.save_alert(edge)
                    print(f"      ✓ Saved as Alert #{alert_id}")
        else:
            print(f"   ℹ️  No edges detected")

    print(f"\n{'=' * 60}")
    print(f"📊 Summary:")
    print(f"   Games checked: {len(live_games)}")
    print(f"   API calls used: {total_api_calls}")
    print(f"   Data saved: {'Yes' if args.save else 'No (use --save to save)'}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
