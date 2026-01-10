"""
Line Shopping Report

Shows the value gained from shopping odds across multiple bookmakers.
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
from loguru import logger
from src.data.odds_api import OddsAPIClient
from src.betting.line_shopping import LineShoppingEngine, get_line_shopping_summary


def generate_line_shopping_report():
    """Generate comprehensive line shopping report for today's games."""

    logger.info("=" * 80)
    logger.info("🛒 LINE SHOPPING REPORT")
    logger.info("=" * 80)
    logger.info("")

    # Fetch all odds
    logger.info("Fetching odds from all bookmakers...")
    client = OddsAPIClient()
    odds_df = client.get_current_odds(markets=['spreads'])

    if odds_df.empty:
        logger.warning("No odds data available")
        return

    # Get number of bookmakers
    num_books = odds_df['bookmaker'].nunique()
    num_games = odds_df['game_id'].nunique()

    logger.info(f"✓ Found {num_games} games from {num_books} bookmakers")
    logger.info("")

    # Initialize shopping engine
    engine = LineShoppingEngine()

    # Summary table
    summary = get_line_shopping_summary(odds_df)

    if summary.empty:
        logger.warning("Not enough data for line shopping comparison")
        return

    logger.info("📊 LINE SHOPPING VALUE BY GAME")
    logger.info("-" * 80)

    for game_id in summary['game_id'].unique():
        game_rows = summary[summary['game_id'] == game_id]
        game_info = game_rows.iloc[0]

        logger.info(f"\n{game_info['away_team']} @ {game_info['home_team']}")

        for _, row in game_rows.iterrows():
            side_name = row['side'].upper()
            best_odds = row['best_odds']
            worst_odds = row['worst_odds']
            value = row['value_vs_worst']
            num_books_avail = row['num_books']

            # Calculate potential profit difference on $100 bet
            if best_odds > 0:
                best_profit = 100 * (best_odds / 100)
            else:
                best_profit = 100 * (100 / abs(best_odds))

            if worst_odds > 0:
                worst_profit = 100 * (worst_odds / 100)
            else:
                worst_profit = 100 * (100 / abs(worst_odds))

            profit_diff = best_profit - worst_profit

            logger.info(f"  {side_name:4} | Best: {best_odds:+4.0f} | Worst: {worst_odds:+4.0f} | "
                       f"Value: {value:+.2%} | ${profit_diff:+.2f} on $100 | {num_books_avail} books")

    # Overall statistics
    logger.info("")
    logger.info("=" * 80)
    logger.info("📈 OVERALL LINE SHOPPING STATISTICS")
    logger.info("=" * 80)

    avg_value = summary['value_vs_worst'].mean()
    max_value = summary['value_vs_worst'].max()
    median_value = summary['value_vs_worst'].median()

    logger.info(f"Average value vs worst line:  {avg_value:+.2%}")
    logger.info(f"Median value vs worst line:   {median_value:+.2%}")
    logger.info(f"Maximum value found:          {max_value:+.2%}")
    logger.info("")

    # Estimate ROI improvement
    # Typical bet wins ~52-56% of the time at -110
    # Getting -105 instead of -110 = ~0.5% better ROI
    estimated_roi_gain = avg_value * 100  # Rough estimate
    logger.info(f"💰 Estimated ROI improvement from line shopping: ~{estimated_roi_gain:.1f}%")
    logger.info("")

    # Best opportunities
    logger.info("🎯 TOP 5 LINE SHOPPING OPPORTUNITIES")
    logger.info("-" * 80)

    top_opportunities = summary.nlargest(5, 'value_vs_worst')

    for idx, row in top_opportunities.iterrows():
        game_str = f"{row['away_team']} @ {row['home_team']}"
        side = row['side'].upper()
        value = row['value_vs_worst']
        best = row['best_odds']
        worst = row['worst_odds']

        logger.info(f"{game_str:40} | {side:4} | {value:+.2%} value ({best:+4.0f} vs {worst:+4.0f})")

    logger.info("")
    logger.info("=" * 80)

    # Detailed comparison for first game
    if num_games > 0:
        first_game_id = summary['game_id'].iloc[0]
        first_game_info = summary[summary['game_id'] == first_game_id].iloc[0]

        logger.info(f"📋 DETAILED COMPARISON: {first_game_info['away_team']} @ {first_game_info['home_team']}")
        logger.info("=" * 80)

        for side in ['home', 'away']:
            logger.info(f"\n{side.upper()} Side:")
            logger.info("-" * 80)

            comparison = engine.compare_all_books(odds_df, first_game_id, 'spread', side)

            if not comparison.empty:
                # Format for display
                display_cols = ['bookmaker', 'line', 'odds', 'implied_prob', 'profit_on_100']
                display_df = comparison[display_cols].copy()
                display_df.columns = ['Bookmaker', 'Line', 'Odds', 'Implied %', 'Profit/$100']

                # Format percentages
                display_df['Implied %'] = display_df['Implied %'].apply(lambda x: f"{x:.1%}")
                display_df['Profit/$100'] = display_df['Profit/$100'].apply(lambda x: f"${x:.2f}")

                print(display_df.to_string(index=False))
            else:
                logger.info("No odds available")

    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ Line shopping report complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        generate_line_shopping_report()
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        import traceback
        traceback.print_exc()
