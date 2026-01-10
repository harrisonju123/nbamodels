"""
Dashboard Enhancement Components

Provides interactive widgets and visualizations for better decision-making:
- Kelly sizing calculator
- Alternative data status display
- Model confidence indicators
- Enhanced pick visualization
"""

import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from src.utils.constants import BETS_DB_PATH


def kelly_calculator_widget(default_bankroll=1000.0):
    """
    Interactive Kelly sizing calculator.

    Shows recommended bet sizes for different Kelly fractions.
    """
    st.markdown("### 💰 Kelly Sizing Calculator")

    col1, col2 = st.columns(2)

    with col1:
        bankroll = st.number_input(
            "Bankroll ($)",
            min_value=100.0,
            max_value=100000.0,
            value=default_bankroll,
            step=100.0,
            help="Your total betting bankroll"
        )

    with col2:
        kelly_fraction = st.select_slider(
            "Kelly Fraction",
            options=[0.10, 0.25, 0.50, 0.75, 1.00],
            value=0.25,
            help="Fraction of full Kelly to bet (0.25 = quarter Kelly, recommended)"
        )

    # Example calculations
    st.markdown("#### Example Bet Sizes")

    example_kellys = [0.05, 0.10, 0.15, 0.20, 0.30]

    examples = []
    for full_kelly in example_kellys:
        bet_size = bankroll * full_kelly * kelly_fraction
        bet_size = max(10.0, min(50.0, bet_size))  # Cap at $10-50

        examples.append({
            'Full Kelly %': f"{full_kelly*100:.0f}%",
            'Your Fraction': f"{kelly_fraction*100:.0f}%",
            'Bet Size': f"${bet_size:.0f}",
            'Risk %': f"{(bet_size/bankroll)*100:.1f}%"
        })

    st.dataframe(
        pd.DataFrame(examples),
        hide_index=True,
        use_container_width=True
    )

    # Kelly fraction explanation
    with st.expander("ℹ️ What is Kelly Fraction?"):
        st.markdown("""
        **Full Kelly (1.00)**: Maximum growth but high variance
        - Can lead to large swings
        - Aggressive strategy

        **Half Kelly (0.50)**: Balanced approach
        - 75% of full Kelly growth
        - Much lower variance

        **Quarter Kelly (0.25)**: Conservative (Recommended)
        - 50% of full Kelly growth
        - Very low variance
        - Sustainable long-term

        **Tenth Kelly (0.10)**: Ultra-conservative
        - Minimal variance
        - Slow but steady growth
        """)

    return bankroll, kelly_fraction


def get_alternative_data_status(game_id=None, home_team=None, away_team=None, game_date=None):
    """
    Check alternative data availability for a game.

    Returns dict with status of:
    - referee_assigned
    - lineups_confirmed
    - news_available
    - sentiment_available
    """
    status = {
        'referee_assigned': False,
        'lineup_confirmed_home': False,
        'lineup_confirmed_away': False,
        'news_available_home': False,
        'news_available_away': False,
        'sentiment_available': False,
        'ref_count': 0,
        'lineup_count_home': 0,
        'lineup_count_away': 0,
        'news_count_home': 0,
        'news_count_away': 0,
    }

    try:
        conn = sqlite3.connect(BETS_DB_PATH)
        conn.row_factory = sqlite3.Row

        # Check referee assignments
        if game_id:
            ref_query = "SELECT COUNT(*) as count FROM referee_assignments WHERE game_id = ?"
            ref_count = conn.execute(ref_query, (game_id,)).fetchone()['count']
            status['referee_assigned'] = ref_count > 0
            status['ref_count'] = ref_count

        # Check lineup confirmations
        if game_id and home_team:
            lineup_home_query = """
                SELECT COUNT(*) as count
                FROM confirmed_lineups
                WHERE game_id = ? AND team_abbrev = ? AND is_starter = 1
            """
            lineup_count_home = conn.execute(lineup_home_query, (game_id, home_team)).fetchone()['count']
            status['lineup_confirmed_home'] = lineup_count_home >= 5
            status['lineup_count_home'] = lineup_count_home

        if game_id and away_team:
            lineup_away_query = """
                SELECT COUNT(*) as count
                FROM confirmed_lineups
                WHERE game_id = ? AND team_abbrev = ? AND is_starter = 1
            """
            lineup_count_away = conn.execute(lineup_away_query, (game_id, away_team)).fetchone()['count']
            status['lineup_confirmed_away'] = lineup_count_away >= 5
            status['lineup_count_away'] = lineup_count_away

        # Check news (last 24h)
        if home_team or away_team:
            news_query = """
                SELECT team_abbrev, COUNT(*) as count
                FROM news_entities
                WHERE team_abbrev IN (?, ?)
                  AND article_id IN (
                      SELECT id FROM news_articles
                      WHERE datetime(published_at) > datetime('now', '-24 hours')
                  )
                GROUP BY team_abbrev
            """
            news_results = conn.execute(news_query, (home_team or '', away_team or '')).fetchall()

            for row in news_results:
                if row['team_abbrev'] == home_team:
                    status['news_available_home'] = row['count'] > 0
                    status['news_count_home'] = row['count']
                elif row['team_abbrev'] == away_team:
                    status['news_available_away'] = row['count'] > 0
                    status['news_count_away'] = row['count']

        # Sentiment is stubbed for now
        status['sentiment_available'] = False

        conn.close()

    except Exception as e:
        # If error, return default (all False)
        pass

    return status


def alternative_data_indicators(status):
    """
    Display alternative data status as colored indicators.

    Args:
        status: Dict from get_alternative_data_status()

    Returns:
        Formatted string with emoji indicators
    """
    indicators = []

    # Referee
    if status['referee_assigned']:
        indicators.append(f"👨‍⚖️ Refs ({status['ref_count']})")
    else:
        indicators.append("⚪ Refs")

    # Lineups
    lineup_home = status['lineup_confirmed_home']
    lineup_away = status['lineup_confirmed_away']

    if lineup_home and lineup_away:
        indicators.append(f"✅ Lineups ({status['lineup_count_home']}/{status['lineup_count_away']})")
    elif lineup_home or lineup_away:
        indicators.append(f"🟡 Lineups ({status['lineup_count_home']}/{status['lineup_count_away']})")
    else:
        indicators.append("⚪ Lineups")

    # News
    news_total = status['news_count_home'] + status['news_count_away']
    if news_total > 5:
        indicators.append(f"📰 News ({news_total})")
    elif news_total > 0:
        indicators.append(f"🟡 News ({news_total})")
    else:
        indicators.append("⚪ News")

    return " | ".join(indicators)


def confidence_indicator(edge, kelly, alternative_data_score=0):
    """
    Calculate and display model confidence level.

    Args:
        edge: Edge percentage (0-100)
        kelly: Kelly percentage (0-1)
        alternative_data_score: 0-1 score for alt data availability

    Returns:
        tuple: (confidence_level, confidence_pct, color, emoji)
    """
    # Base confidence from edge
    base_confidence = min(edge / 10.0, 1.0)  # 10% edge = 100% base confidence

    # Kelly multiplier (high Kelly = high confidence)
    kelly_multiplier = min(kelly / 0.30, 1.0)  # 30% Kelly = 100% multiplier

    # Alternative data bonus
    alt_data_bonus = alternative_data_score * 0.2  # Up to 20% bonus

    # Combined confidence
    confidence = (base_confidence * 0.6 + kelly_multiplier * 0.3 + alt_data_bonus * 0.1)
    confidence_pct = confidence * 100

    # Classify
    if confidence_pct >= 80:
        level = "VERY HIGH"
        color = "#10b981"  # Green
        emoji = "🔥"
    elif confidence_pct >= 60:
        level = "HIGH"
        color = "#22c55e"  # Light green
        emoji = "✅"
    elif confidence_pct >= 40:
        level = "MEDIUM"
        color = "#f59e0b"  # Yellow
        emoji = "🟡"
    else:
        level = "LOW"
        color = "#ef4444"  # Red
        emoji = "⚠️"

    return level, confidence_pct, color, emoji


def enhanced_pick_display(game_row, bankroll=1000.0, kelly_fraction=0.25):
    """
    Enhanced display for a single pick with all relevant information.

    Args:
        game_row: DataFrame row with game/bet information
        bankroll: User's bankroll
        kelly_fraction: Kelly fraction to use
    """
    # Extract game info
    away_team = game_row.get('away_team', 'AWAY')
    home_team = game_row.get('home_team', 'HOME')
    game_id = game_row.get('game_id', '')

    # Determine bet side
    if 'bet_side' in game_row and pd.notna(game_row['bet_side']) and game_row['bet_side'] != 'PASS':
        is_betting_home = game_row['bet_side'] == 'HOME'
    else:
        is_betting_home = game_row.get('bet_home', False)

    bet_team = home_team if is_betting_home else away_team
    bet_side = 'home' if is_betting_home else 'away'

    # Get betting metrics
    line = game_row.get('line', game_row.get('spread_home', 0))
    if not is_betting_home:
        line = -line

    edge = game_row.get('edge_vs_market', game_row.get('home_edge', 0) if is_betting_home else game_row.get('away_edge', 0))
    kelly = game_row.get('kelly', game_row.get('home_kelly', 0) if is_betting_home else game_row.get('away_kelly', 0))

    # Calculate bet size
    bet_amt = bankroll * kelly * kelly_fraction
    bet_amt = max(10.0, min(50.0, bet_amt))  # Cap at $10-50

    # Get alternative data status
    alt_status = get_alternative_data_status(
        game_id=game_id,
        home_team=home_team,
        away_team=away_team
    )

    # Calculate alt data score
    alt_score = (
        (0.3 if alt_status['referee_assigned'] else 0) +
        (0.3 if alt_status['lineup_confirmed_home'] and alt_status['lineup_confirmed_away'] else 0) +
        (0.2 if alt_status['news_available_home'] or alt_status['news_available_away'] else 0)
    )

    # Get confidence
    conf_level, conf_pct, conf_color, conf_emoji = confidence_indicator(edge, kelly, alt_score)

    # Display
    with st.expander(f"{conf_emoji} **{away_team} @ {home_team}** - {conf_level} Confidence", expanded=True):
        # Top metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Pick", f"{bet_team} {line:+.1f}")

        with col2:
            st.metric("Edge", f"{edge:.1f}%")

        with col3:
            st.metric("Kelly", f"{kelly*100:.1f}%")

        with col4:
            st.metric("Suggested Bet", f"${bet_amt:.0f}")

        with col5:
            st.markdown(
                f"<div style='text-align: center; padding: 10px; background-color: {conf_color}; "
                f"border-radius: 5px; color: white;'>"
                f"<b>{conf_pct:.0f}%</b><br><small>Confidence</small></div>",
                unsafe_allow_html=True
            )

        st.divider()

        # Alternative data status
        col1, col2 = st.columns([3, 1])

        with col1:
            st.caption("📊 **Data Availability:** " + alternative_data_indicators(alt_status))

        with col2:
            if st.button(f"📝 Log Bet", key=f"log_{game_id}", use_container_width=True):
                return {
                    'log_bet': True,
                    'game_id': game_id,
                    'home_team': home_team,
                    'away_team': away_team,
                    'bet_side': bet_side,
                    'bet_amount': bet_amt,
                    'line': line,
                }

    return None


def alternative_data_status_summary():
    """
    Display overall alternative data collection status.
    """
    st.markdown("### 📊 Alternative Data Status")

    try:
        conn = sqlite3.connect(BETS_DB_PATH)
        conn.row_factory = sqlite3.Row

        # Get counts
        ref_count = conn.execute("SELECT COUNT(*) as count FROM referee_assignments").fetchone()['count']
        lineup_count = conn.execute("SELECT COUNT(*) as count FROM confirmed_lineups").fetchone()['count']
        news_count = conn.execute("SELECT COUNT(*) as count FROM news_articles").fetchone()['count']

        # Recent data (last 24h)
        ref_recent = conn.execute(
            "SELECT COUNT(*) as count FROM referee_assignments WHERE collected_at > datetime('now', '-24 hours')"
        ).fetchone()['count']

        lineup_recent = conn.execute(
            "SELECT COUNT(*) as count FROM confirmed_lineups WHERE collected_at > datetime('now', '-24 hours')"
        ).fetchone()['count']

        news_recent = conn.execute(
            "SELECT COUNT(*) as count FROM news_articles WHERE collected_at > datetime('now', '-24 hours')"
        ).fetchone()['count']

        conn.close()

        # Display
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "👨‍⚖️ Referee Assignments",
                f"{ref_count} total",
                f"+{ref_recent} today",
                delta_color="normal"
            )

        with col2:
            st.metric(
                "✅ Lineup Confirmations",
                f"{lineup_count} total",
                f"+{lineup_recent} today",
                delta_color="normal"
            )

        with col3:
            st.metric(
                "📰 News Articles",
                f"{news_count} total",
                f"+{news_recent} today",
                delta_color="normal"
            )

        # Status indicators
        st.caption(f"✅ Collecting every 15 min (5-11 PM) | 📰 Collecting hourly | 👨‍⚖️ Collecting daily (10 AM)")

    except Exception as e:
        st.error(f"Error loading alternative data status: {e}")
