"""
NBA Betting Model Dashboard

Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv('.env')

# Time conversion helper
def format_time_est(iso_time: str) -> str:
    """Convert ISO timestamp to EST formatted string."""
    if not iso_time:
        return ""
    try:
        # Parse ISO format (usually UTC from odds API)
        if iso_time.endswith('Z'):
            dt = datetime.fromisoformat(iso_time.replace('Z', '+00:00'))
        elif '+' in iso_time or iso_time.count('-') > 2:
            dt = datetime.fromisoformat(iso_time)
        else:
            # Assume UTC if no timezone
            dt = datetime.fromisoformat(iso_time).replace(tzinfo=ZoneInfo('UTC'))

        # Convert to EST
        est = ZoneInfo('America/New_York')
        dt_est = dt.astimezone(est)
        return dt_est.strftime('%b %d, %I:%M %p EST')
    except Exception:
        return iso_time[:16] if len(iso_time) >= 16 else iso_time

# Page config
st.set_page_config(
    page_title="NBA Betting Model",
    page_icon="🏀",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    /* Spacing variables */
    :root {
        --spacing-xs: 0.25rem;
        --spacing-sm: 0.5rem;
        --spacing-md: 1rem;
        --spacing-lg: 1.5rem;
    }

    /* Reduce default padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    /* Card styling */
    .metric-card {
        background-color: #1E1E1E;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #2D2D2D;
    }

    /* Bet recommendation styling */
    .bet-card {
        background: linear-gradient(135deg, #1a472a 0%, #0d2818 100%);
        border-radius: 8px;
        padding: 0.75rem;
        border-left: 4px solid #4CAF50;
        margin-bottom: 0.5rem;
    }

    .no-bet-card {
        background-color: #1E1E1E;
        border-radius: 8px;
        padding: 0.75rem;
        opacity: 0.6;
    }

    /* Market headers */
    .market-header {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #9E9E9E;
        margin-bottom: 0.5rem;
    }

    /* Status colors */
    .positive { color: #4CAF50; }
    .negative { color: #EF5350; }
    .neutral { color: #9E9E9E; }

    /* Game with bets indicator */
    .has-bets {
        border-left: 3px solid #4CAF50;
    }

    /* Cleaner dividers */
    hr {
        margin: 1rem 0;
        border-color: #2D2D2D;
    }

    /* Summary metrics */
    div[data-testid="stMetric"] {
        background-color: #1E1E1E;
        padding: 0.75rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def load_games():
    """Load historical games data."""
    return pd.read_parquet('data/raw/games.parquet')


@st.cache_resource
def load_model():
    """Load trained model (tuned version)."""
    with open('models/spread_model_tuned.pkl', 'rb') as f:
        return pickle.load(f)


def get_predictions_cached():
    """Get moneyline predictions from cache."""
    try:
        from src.prediction_cache import load_cached_predictions, get_cache_info
        predictions = load_cached_predictions()
        cache_info = get_cache_info()
        if predictions and "moneyline" in predictions:
            return predictions["moneyline"], cache_info
        return None, cache_info
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        return None, None


def refresh_moneyline_predictions():
    """Refresh just moneyline predictions."""
    from src.prediction_cache import refresh_predictions
    predictions = refresh_predictions(min_edge=0.02)
    if predictions and "moneyline" in predictions:
        return predictions["moneyline"]
    return None


def run_backtest(games_df, model_data, test_season=2024):
    """Run backtest on historical data."""
    from src.features import GameFeatureBuilder
    from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

    builder = GameFeatureBuilder()
    features = builder.build_game_features(games_df)
    feature_cols = model_data['feature_cols']

    # Filter to test season
    test_df = features[features['season'] == test_season].copy()
    test_df = test_df.dropna(subset=['home_win'])

    if test_df.empty:
        return None

    X = test_df[feature_cols].fillna(0)
    y = test_df['home_win']

    model = model_data['model']
    probs = model.predict_proba(X)[:, 1]
    preds = (probs > 0.5).astype(int)

    # Simulate betting
    IMPLIED_PROB = 0.524
    MIN_EDGE = 0.03

    results = []
    bankroll = 10000
    bankroll_history = [bankroll]

    for i, (prob, actual, date) in enumerate(zip(probs, y.values, test_df['date'].values)):
        edge = prob - IMPLIED_PROB
        bet_on_home = edge > MIN_EDGE
        bet_on_away = (1 - prob) - IMPLIED_PROB > MIN_EDGE

        if bet_on_home:
            bet_size = min(bankroll * 0.02, bankroll * 0.05)
            won = actual == 1
            profit = bet_size * (100/110) if won else -bet_size
            bankroll += profit
            results.append({
                'date': date,
                'prob': prob,
                'edge': edge,
                'side': 'home',
                'won': won,
                'profit': profit,
                'bankroll': bankroll
            })
        elif bet_on_away:
            bet_size = min(bankroll * 0.02, bankroll * 0.05)
            won = actual == 0
            profit = bet_size * (100/110) if won else -bet_size
            bankroll += profit
            results.append({
                'date': date,
                'prob': 1 - prob,
                'edge': (1 - prob) - IMPLIED_PROB,
                'side': 'away',
                'won': won,
                'profit': profit,
                'bankroll': bankroll
            })

        bankroll_history.append(bankroll)

    return {
        'accuracy': accuracy_score(y, preds),
        'auc': roc_auc_score(y, probs),
        'brier': brier_score_loss(y, probs),
        'bets': pd.DataFrame(results),
        'bankroll_history': bankroll_history,
        'total_games': len(test_df),
        'probs': probs,
        'actuals': y.values
    }


def main():
    st.title("NBA Betting Model Dashboard")

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "Predictions",
        "Bet Tracker",
        "Strategy Performance",
        "Analytics"
    ])

    # Account settings in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Bet Sizing")
    account_value = st.sidebar.number_input(
        "Account Balance ($)",
        min_value=100,
        max_value=1000000,
        value=1000,
        step=100,
        help="Your total betting bankroll"
    )
    # Store in session state for use across pages
    st.session_state['account_value'] = account_value

    # Load data
    try:
        games = load_games()
        model_data = load_model()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    if page == "Predictions":
        show_predictions_page()
    elif page == "Bet Tracker":
        show_bet_tracker_page(games)
    elif page == "Strategy Performance":
        show_strategy_page()
    elif page == "Analytics":
        show_analytics_page(games, model_data)


def get_all_market_predictions_cached():
    """Get predictions from cache (fast, no API calls)."""
    try:
        from src.prediction_cache import load_cached_predictions, get_cache_info
        predictions = load_cached_predictions()
        cache_info = get_cache_info()
        return predictions, cache_info
    except Exception as e:
        st.error(f"Error loading cached predictions: {e}")
        return None, None


def refresh_all_predictions():
    """Refresh predictions (slow, calls APIs)."""
    try:
        from src.prediction_cache import refresh_predictions
        return refresh_predictions(min_edge=0.02)
    except Exception as e:
        st.error(f"Error refreshing predictions: {e}")
        return None


def show_bet_tracker_page(games):
    """Display bet tracking and performance."""
    st.header("Bet Tracker")

    from src.bet_tracker import (
        get_pending_bets, get_settled_bets,
        get_performance_summary, settle_all_pending, log_manual_bet
    )

    # Auto-settle completed games on page load
    settled_count = settle_all_pending(games)
    if settled_count > 0:
        st.success(f"Auto-settled {settled_count} completed bet(s)")

    # Manual Bet Entry Section
    st.subheader("Log a Bet")

    # Get today's games from predictions for easy selection
    predictions, _ = get_all_market_predictions_cached()

    with st.form("log_bet_form"):
        col1, col2 = st.columns(2)

        with col1:
            # Game selection
            game_options = {"-- Select Game --": None}
            if predictions:
                for market in ["moneyline", "spread", "totals"]:
                    if market in predictions and predictions[market] is not None:
                        for _, row in predictions[market].iterrows():
                            game_key = f"{row['away_team']} @ {row['home_team']}"
                            if game_key not in game_options:
                                game_options[game_key] = {
                                    "game_id": row["game_id"],
                                    "home_team": row["home_team"],
                                    "away_team": row["away_team"],
                                    "commence_time": row.get("commence_time", "")
                                }

            selected_game = st.selectbox("Game", list(game_options.keys()))

            bet_type = st.selectbox("Bet Type", ["moneyline", "spread", "totals"])

            if bet_type == "moneyline":
                bet_side = st.selectbox("Side", ["home", "away"])
                line = None
            elif bet_type == "spread":
                bet_side = st.selectbox("Side", ["home", "away"])
                line = st.number_input("Spread Line", value=-3.0, step=0.5,
                                       help="Home team spread (e.g., -3.5 means home -3.5)")
            else:  # totals
                bet_side = st.selectbox("Side", ["over", "under"])
                line = st.number_input("Total Line", value=220.0, step=0.5)

        with col2:
            odds = st.number_input(
                "Odds (American)",
                value=-110,
                step=5,
                help="American odds (e.g., -110, +150)"
            )

            bet_amount = st.number_input(
                "Bet Amount ($)",
                min_value=1.0,
                value=100.0,
                step=10.0
            )

            bookmaker = st.text_input("Sportsbook (optional)", placeholder="e.g., DraftKings")

        submitted = st.form_submit_button("Log Bet", type="primary")

        if submitted:
            if selected_game == "-- Select Game --" or game_options[selected_game] is None:
                st.error("Please select a game")
            else:
                game_info = game_options[selected_game]
                try:
                    log_manual_bet(
                        game_id=game_info["game_id"],
                        home_team=game_info["home_team"],
                        away_team=game_info["away_team"],
                        commence_time=game_info["commence_time"],
                        bet_type=bet_type,
                        bet_side=bet_side,
                        odds=odds,
                        bet_amount=bet_amount,
                        line=line,
                        bookmaker=bookmaker if bookmaker else None,
                    )
                    st.success(f"Logged: ${bet_amount:.0f} on {bet_type} {bet_side} at {odds:+.0f}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error logging bet: {e}")

    st.markdown("---")

    # Action buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Settle Completed Games", type="secondary"):
            count = settle_all_pending(games)
            if count > 0:
                st.success(f"Settled {count} bets")
                st.rerun()
            else:
                st.info("No bets to settle")

    with col2:
        if st.button("Refresh"):
            st.rerun()

    st.markdown("---")

    # Performance summary
    summary = get_performance_summary()

    st.subheader("Performance")
    cols = st.columns(4)
    cols[0].metric("Bets", summary["total_bets"])
    cols[1].metric("Win Rate", f"{summary['win_rate']:.1%}" if summary['total_bets'] > 0 else "N/A")
    cols[2].metric("Profit", f"${summary['total_profit']:+,.0f}")
    cols[3].metric("ROI", f"{summary['roi']:.1%}" if summary['total_bets'] > 0 else "N/A")

    st.markdown("---")

    # Tabs for pending vs settled
    tab1, tab2 = st.tabs(["Pending Bets", "Settled Bets"])

    with tab1:
        pending = get_pending_bets()
        if pending.empty:
            st.info("No pending bets")
        else:
            st.write(f"**{len(pending)} pending bets**")

            # Sort by commence time
            pending = pending.sort_values("commence_time")

            # Include bet_amount if available
            has_amount = "bet_amount" in pending.columns and pending["bet_amount"].notna().any()
            if has_amount:
                display_cols = ["commence_time", "away_team", "home_team", "bet_type", "bet_side", "bet_amount", "odds", "line"]
            else:
                display_cols = ["commence_time", "away_team", "home_team", "bet_type", "bet_side", "odds", "line"]

            display_df = pending[display_cols].copy()
            display_df["odds"] = display_df["odds"].apply(lambda x: f"{x:+.0f}" if pd.notna(x) else "")
            display_df["line"] = display_df["line"].apply(lambda x: f"{x:+.1f}" if pd.notna(x) else "")
            display_df["commence_time"] = display_df["commence_time"].apply(format_time_est)

            if has_amount:
                display_df["bet_amount"] = display_df["bet_amount"].apply(lambda x: f"${x:.0f}" if pd.notna(x) else "-")
                display_df = display_df.rename(columns={"bet_amount": "wager"})

            st.dataframe(display_df, use_container_width=True, hide_index=True)

    with tab2:
        settled = get_settled_bets()
        if settled.empty:
            st.info("No settled bets yet")
        else:
            st.write(f"**{len(settled)} settled bets**")

            # Sort by settle time descending
            settled = settled.sort_values("settled_at", ascending=False)

            # Build display columns based on available data
            has_amount = "bet_amount" in settled.columns and settled["bet_amount"].notna().any()
            has_clv = "clv" in settled.columns and settled["clv"].notna().any()

            display_cols = ["settled_at", "away_team", "home_team", "bet_type", "bet_side"]
            if has_amount:
                display_cols.append("bet_amount")
            display_cols.append("odds")
            if has_clv:
                display_cols.append("clv")
            display_cols.extend(["outcome", "profit"])

            display_df = settled[display_cols].copy()
            display_df["odds"] = display_df["odds"].apply(lambda x: f"{x:+.0f}" if pd.notna(x) else "")
            display_df["profit"] = display_df["profit"].apply(lambda x: f"${x:+.0f}" if pd.notna(x) else "")
            display_df["settled_at"] = display_df["settled_at"].apply(lambda x: str(x)[:10] if x else "")

            if has_amount:
                display_df["bet_amount"] = display_df["bet_amount"].apply(lambda x: f"${x:.0f}" if pd.notna(x) else "-")
                display_df = display_df.rename(columns={"bet_amount": "wager"})

            if has_clv:
                display_df["clv"] = display_df["clv"].apply(lambda x: f"{x:+.1%}" if pd.notna(x) else "-")

            st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Cumulative profit chart if we have settled bets
    if not settled.empty and len(settled) > 1:
        st.markdown("---")
        st.subheader("Cumulative Profit")

        settled_sorted = settled.sort_values("settled_at")
        settled_sorted["cumulative_profit"] = settled_sorted["profit"].cumsum()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(settled_sorted) + 1)),
            y=settled_sorted["cumulative_profit"],
            mode='lines+markers',
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.1)' if settled_sorted["cumulative_profit"].iloc[-1] > 0 else 'rgba(255, 0, 0, 0.1)',
            line=dict(color='green' if settled_sorted["cumulative_profit"].iloc[-1] > 0 else 'red')
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            xaxis_title="Bet Number",
            yaxis_title="Cumulative Profit ($)",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)


def show_predictions_page():
    st.header("Predictions")

    from src.bet_tracker import log_manual_bet

    # Load all predictions from cache
    all_predictions, cache_info = get_all_market_predictions_cached()

    # Header row: cache info, market filter, and refresh
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        if cache_info:
            st.caption(f"Updated: {cache_info['timestamp'][:16]} | {cache_info['num_games']} games")
        else:
            st.caption("No cached predictions")

    with col2:
        # Market filter
        available_markets = []
        if all_predictions:
            available_markets = [m for m in all_predictions.keys()
                               if all_predictions[m] is not None and not all_predictions[m].empty]
        selected_markets = st.multiselect(
            "Markets",
            options=available_markets if available_markets else ["moneyline", "spread", "totals"],
            default=available_markets if available_markets else [],
            label_visibility="collapsed"
        )

    with col3:
        if st.button("Refresh", type="primary", key="refresh_all", use_container_width=True):
            with st.spinner("Updating..."):
                all_predictions = refresh_all_predictions()
                if all_predictions:
                    st.rerun()

    if all_predictions is None or not selected_markets:
        st.info("No predictions available. Click Refresh to load.")
        return

    # Get account value
    account = st.session_state.get('account_value', 1000)
    kelly_multiplier = 0.2

    # Count bets across selected markets and identify games with bets
    total_bets = 0
    bet_counts = {}
    games_with_bets = set()

    for market in selected_markets:
        df = all_predictions.get(market)
        if df is None or df.empty:
            continue
        if market == "moneyline":
            bets = int(df["bet_home"].sum() + df["bet_away"].sum())
            for _, r in df.iterrows():
                if r["bet_home"] or r["bet_away"]:
                    games_with_bets.add(r["game_id"])
        elif market == "spread":
            bets = int(df["bet_cover"].sum())
            for _, r in df.iterrows():
                if r["bet_cover"]:
                    games_with_bets.add(r["game_id"])
        elif market == "totals":
            bets = int(df["bet_over"].sum() + df["bet_under"].sum())
            for _, r in df.iterrows():
                if r["bet_over"] or r["bet_under"]:
                    games_with_bets.add(r["game_id"])
        elif market == "ats":
            bets = int(df["bet_home"].sum() + df["bet_away"].sum())
            for _, r in df.iterrows():
                if r["bet_home"] or r["bet_away"]:
                    games_with_bets.add(r["game_id"])
        elif market == "consensus":
            bets = int(df["bet_home"].sum() + df["bet_away"].sum())
            for _, r in df.iterrows():
                if r["bet_home"] or r["bet_away"]:
                    games_with_bets.add(r["game_id"])
        elif market == "edge":
            bets = int(df["bet_home"].sum() + df["bet_away"].sum())
            for _, r in df.iterrows():
                if r["bet_home"] or r["bet_away"]:
                    games_with_bets.add(r["game_id"])
        else:
            bets = 0
        bet_counts[market] = bets
        total_bets += bets

    # Summary metrics - cleaner 3-column layout
    cols = st.columns(3)
    cols[0].metric("Recommended Bets", total_bets)
    cols[1].metric("Games with Edge", len(games_with_bets))
    cols[2].metric("Account", f"${account:,.0f}")

    st.divider()

    # Get unique games and sort by time
    game_times = {}
    game_info = {}
    for market in selected_markets:
        df = all_predictions.get(market)
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            gid = row.get("game_id")
            if gid and gid not in game_times:
                game_times[gid] = row.get("commence_time", "")
                game_info[gid] = {
                    "home_team": row.get("home_team"),
                    "away_team": row.get("away_team"),
                    "commence_time": row.get("commence_time")
                }

    sorted_games = sorted(game_times.keys(), key=lambda g: game_times.get(g, ""))

    # Display each game
    for game_id in sorted_games:
        info = game_info.get(game_id, {})
        has_bet = game_id in games_with_bets

        # Game header with [BET] indicator
        game_label = f"{info.get('away_team', '?')} @ {info.get('home_team', '?')}"
        if has_bet:
            game_label = f"[BET] {game_label}"

        with st.expander(game_label, expanded=has_bet):
            st.caption(format_time_est(info.get('commence_time', '')))

            # Dynamic columns based on selected markets
            cols = st.columns(len(selected_markets))

            # MONEYLINE
            if "moneyline" in selected_markets:
                col_idx = selected_markets.index("moneyline")
                with cols[col_idx]:
                    st.markdown('<div class="market-header">MONEYLINE</div>', unsafe_allow_html=True)
                    ml_df = all_predictions.get("moneyline")
                    if ml_df is not None and not ml_df.empty:
                        row = ml_df[ml_df["game_id"] == game_id]
                        if not row.empty:
                            row = row.iloc[0]
                            if row["bet_home"]:
                                kelly_pct = row['home_kelly'] * kelly_multiplier
                                suggested_amount = account * kelly_pct
                                home_team = info['home_team']
                                st.success(f"**{home_team}** ML")
                                st.caption(f"${suggested_amount:.0f} | {row['best_home_odds']:+.0f}")
                                with st.popover("Log Bet"):
                                    st.markdown(f"**{home_team}** to win")
                                    ml_odds = st.number_input("Your odds", value=int(row['best_home_odds']), step=5, key=f"ml_home_odds_{game_id}")
                                    ml_amt = st.number_input("Amount ($)", value=float(suggested_amount), step=10.0, key=f"ml_home_amt_{game_id}")
                                    if st.button(f"Log ${ml_amt:.0f} on {home_team}", key=f"ml_home_log_{game_id}", type="primary", use_container_width=True):
                                        log_manual_bet(game_id, info['home_team'], info['away_team'], info['commence_time'],
                                                       "moneyline", "home", ml_odds, ml_amt, model_prob=row['model_home_prob'])
                                        st.success("Bet logged!")
                                        st.rerun()
                            elif row["bet_away"]:
                                kelly_pct = row['away_kelly'] * kelly_multiplier
                                suggested_amount = account * kelly_pct
                                away_team = info['away_team']
                                st.success(f"**{away_team}** ML")
                                st.caption(f"${suggested_amount:.0f} | {row['best_away_odds']:+.0f}")
                                with st.popover("Log Bet"):
                                    st.markdown(f"**{away_team}** to win")
                                    ml_odds = st.number_input("Your odds", value=int(row['best_away_odds']), step=5, key=f"ml_away_odds_{game_id}")
                                    ml_amt = st.number_input("Amount ($)", value=float(suggested_amount), step=10.0, key=f"ml_away_amt_{game_id}")
                                    if st.button(f"Log ${ml_amt:.0f} on {away_team}", key=f"ml_away_log_{game_id}", type="primary", use_container_width=True):
                                        log_manual_bet(game_id, info['home_team'], info['away_team'], info['commence_time'],
                                                       "moneyline", "away", ml_odds, ml_amt, model_prob=row['model_away_prob'])
                                        st.success("Bet logged!")
                                        st.rerun()
                            else:
                                st.caption("No edge")
                        else:
                            st.caption("No data")
                    else:
                        st.caption("No data")

            # SPREAD
            if "spread" in selected_markets:
                col_idx = selected_markets.index("spread")
                with cols[col_idx]:
                    st.markdown('<div class="market-header">SPREAD</div>', unsafe_allow_html=True)
                    sp_df = all_predictions.get("spread")
                    if sp_df is not None and not sp_df.empty:
                        row = sp_df[sp_df["game_id"] == game_id]
                        if not row.empty:
                            row = row.iloc[0]
                            spread_line = row['spread_line']
                            point_edge = row.get('point_edge', 0)
                            bet_side = row.get('bet_side', 'HOME')
                            filters_passed = row.get('filters_passed', '')

                            if row["bet_cover"]:
                                kelly_pct = row['kelly'] * kelly_multiplier
                                suggested_amount = account * kelly_pct

                                # Show correct team based on bet_side
                                if bet_side == "HOME":
                                    bet_team = info['home_team']
                                    bet_line = spread_line
                                else:
                                    bet_team = info['away_team']
                                    bet_line = -spread_line  # Flip for away

                                st.success(f"**{bet_team} {bet_line:+.1f}**")
                                st.caption(f"${suggested_amount:.0f} | Edge: {abs(point_edge):.1f}")
                                st.caption(f"{filters_passed}")

                                with st.popover("Log Bet"):
                                    st.markdown(f"**{bet_team} {bet_line:+.1f}**")
                                    st.caption(f"Filters: {filters_passed}")
                                    sp_line = st.number_input("Your line", value=float(bet_line), step=0.5, key=f"sp_line_{game_id}")
                                    sp_odds = st.number_input("Your odds", value=int(row['best_spread_odds']), step=5, key=f"sp_odds_{game_id}")
                                    sp_amt = st.number_input("Amount ($)", value=float(suggested_amount), step=10.0, key=f"sp_amt_{game_id}")
                                    if st.button(f"Log ${sp_amt:.0f} on {bet_team} {sp_line:+.1f}", key=f"sp_log_{game_id}", type="primary", use_container_width=True):
                                        log_manual_bet(game_id, info['home_team'], info['away_team'], info['commence_time'],
                                                       "spread", bet_side.lower(), sp_odds, sp_amt, line=sp_line, model_prob=row['model_cover_prob'])
                                        st.success("Bet logged!")
                                        st.rerun()
                            else:
                                # Show why bet was skipped
                                reason = ""
                                if row.get('team_excluded', False):
                                    reason = "(excluded team)"
                                elif row.get('betting_on_b2b', False):
                                    reason = "(B2B)"
                                elif abs(point_edge) < 5:
                                    reason = f"(edge {abs(point_edge):.1f})"
                                st.caption(f"{info['home_team']} {spread_line:+.1f} {reason}")
                        else:
                            st.caption("No data")
                    else:
                        st.caption("No data")

            # TOTALS
            if "totals" in selected_markets:
                col_idx = selected_markets.index("totals")
                with cols[col_idx]:
                    st.markdown('<div class="market-header">TOTALS</div>', unsafe_allow_html=True)
                    tot_df = all_predictions.get("totals")
                    if tot_df is not None and not tot_df.empty:
                        row = tot_df[tot_df["game_id"] == game_id]
                        if not row.empty:
                            row = row.iloc[0]
                            if row["bet_over"]:
                                kelly_pct = row['over_kelly'] * kelly_multiplier
                                bet_amount = account * kelly_pct
                                total_line = row['total_line']
                                st.success(f"**Over {total_line:.1f}**")
                                st.caption(f"${bet_amount:.0f} | {row['best_over_odds']:+.0f}")
                                with st.popover("Log Bet"):
                                    st.markdown(f"**Over {total_line:.1f}** points")
                                    tot_line = st.number_input("Your line", value=float(total_line), step=0.5, key=f"tot_over_line_{game_id}")
                                    tot_odds = st.number_input("Your odds", value=int(row['best_over_odds']), step=5, key=f"tot_over_odds_{game_id}")
                                    tot_amt = st.number_input("Amount ($)", value=float(bet_amount), step=10.0, key=f"tot_over_amt_{game_id}")
                                    if st.button(f"Log ${tot_amt:.0f} on Over {tot_line:.1f}", key=f"tot_over_log_{game_id}", type="primary", use_container_width=True):
                                        log_manual_bet(game_id, info['home_team'], info['away_team'], info['commence_time'],
                                                       "totals", "over", tot_odds, tot_amt, line=tot_line, model_prob=row['model_over_prob'])
                                        st.success("Bet logged!")
                                        st.rerun()
                            elif row["bet_under"]:
                                kelly_pct = row['under_kelly'] * kelly_multiplier
                                bet_amount = account * kelly_pct
                                total_line = row['total_line']
                                st.success(f"**Under {total_line:.1f}**")
                                st.caption(f"${bet_amount:.0f} | {row['best_under_odds']:+.0f}")
                                with st.popover("Log Bet"):
                                    st.markdown(f"**Under {total_line:.1f}** points")
                                    tot_line = st.number_input("Your line", value=float(total_line), step=0.5, key=f"tot_under_line_{game_id}")
                                    tot_odds = st.number_input("Your odds", value=int(row['best_under_odds']), step=5, key=f"tot_under_odds_{game_id}")
                                    tot_amt = st.number_input("Amount ($)", value=float(bet_amount), step=10.0, key=f"tot_under_amt_{game_id}")
                                    if st.button(f"Log ${tot_amt:.0f} on Under {tot_line:.1f}", key=f"tot_under_log_{game_id}", type="primary", use_container_width=True):
                                        log_manual_bet(game_id, info['home_team'], info['away_team'], info['commence_time'],
                                                       "totals", "under", tot_odds, tot_amt, line=tot_line, model_prob=row['model_under_prob'])
                                        st.success("Bet logged!")
                                        st.rerun()
                            else:
                                st.caption(f"O/U {row['total_line']:.1f}")
                        else:
                            st.caption("No data")
                    else:
                        st.caption("No data")

            # ATS (Dual Model)
            if "ats" in selected_markets:
                col_idx = selected_markets.index("ats")
                with cols[col_idx]:
                    st.markdown('<div class="market-header">ATS (DUAL MODEL)</div>', unsafe_allow_html=True)
                    ats_df = all_predictions.get("ats")
                    if ats_df is not None and not ats_df.empty:
                        row = ats_df[ats_df["game_id"] == game_id]
                        if not row.empty:
                            row = row.iloc[0]
                            # Safe NaN handling for display values
                            market_spread = row.get('market_spread')
                            market_spread = float(market_spread) if pd.notna(market_spread) else 0.0
                            disagreement = row.get('disagreement')
                            disagreement = float(disagreement) if pd.notna(disagreement) else 0.0
                            edge = row.get('edge_vs_market')
                            edge = float(edge) if pd.notna(edge) else 0.0
                            kelly = row.get('kelly')
                            kelly = float(kelly) if pd.notna(kelly) else 0.0

                            if row["bet_home"]:
                                home_team = info['home_team']
                                kelly_pct = kelly * kelly_multiplier
                                suggested_amount = account * kelly_pct
                                st.success(f"**{home_team} ATS**")
                                st.caption(f"${suggested_amount:.0f} | {market_spread:+.1f}")
                                st.caption(f"Disagree: {disagreement:+.1f} | Edge: {edge:+.1f}")
                                with st.popover("Log Bet"):
                                    st.markdown(f"**{home_team}** covers {market_spread:+.1f}")
                                    mlp_spread = row.get('mlp_spread', 0)
                                    xgb_spread = row.get('xgb_spread', 0)
                                    confidence = row.get('confidence', 0)
                                    st.markdown(f"MLP: {mlp_spread:+.1f} | XGB: {xgb_spread:+.1f}")
                                    st.markdown(f"Confidence: {confidence:.0%}")
                                    ats_line = st.number_input("Your line", value=market_spread, step=0.5, key=f"ats_home_line_{game_id}")
                                    ats_odds = st.number_input("Your odds", value=-110, step=5, key=f"ats_home_odds_{game_id}")
                                    ats_amt = st.number_input("Amount ($)", value=float(suggested_amount) if suggested_amount > 0 else 100.0, step=10.0, key=f"ats_home_amt_{game_id}")
                                    if st.button(f"Log ${ats_amt:.0f} on {home_team} {ats_line:+.1f}", key=f"ats_home_log_{game_id}", type="primary", use_container_width=True):
                                        log_manual_bet(game_id, info['home_team'], info['away_team'], info['commence_time'],
                                                       "spread", "home", ats_odds, ats_amt, line=ats_line)
                                        st.success("Bet logged!")
                                        st.rerun()
                            elif row["bet_away"]:
                                away_team = info['away_team']
                                away_spread = -market_spread
                                kelly_pct = kelly * kelly_multiplier
                                suggested_amount = account * kelly_pct
                                st.success(f"**{away_team} ATS**")
                                st.caption(f"${suggested_amount:.0f} | {away_spread:+.1f}")
                                st.caption(f"Disagree: {disagreement:+.1f} | Edge: {edge:+.1f}")
                                with st.popover("Log Bet"):
                                    st.markdown(f"**{away_team}** covers {away_spread:+.1f}")
                                    mlp_spread = row.get('mlp_spread', 0)
                                    xgb_spread = row.get('xgb_spread', 0)
                                    confidence = row.get('confidence', 0)
                                    st.markdown(f"MLP: {mlp_spread:+.1f} | XGB: {xgb_spread:+.1f}")
                                    st.markdown(f"Confidence: {confidence:.0%}")
                                    ats_line = st.number_input("Your line", value=away_spread, step=0.5, key=f"ats_away_line_{game_id}")
                                    ats_odds = st.number_input("Your odds", value=-110, step=5, key=f"ats_away_odds_{game_id}")
                                    ats_amt = st.number_input("Amount ($)", value=float(suggested_amount) if suggested_amount > 0 else 100.0, step=10.0, key=f"ats_away_amt_{game_id}")
                                    if st.button(f"Log ${ats_amt:.0f} on {away_team} {ats_line:+.1f}", key=f"ats_away_log_{game_id}", type="primary", use_container_width=True):
                                        log_manual_bet(game_id, info['home_team'], info['away_team'], info['commence_time'],
                                                       "spread", "away", ats_odds, ats_amt, line=ats_line)
                                        st.success("Bet logged!")
                                        st.rerun()
                            else:
                                # Show disagreement info even when no bet
                                st.caption(f"Disagree: {disagreement:+.1f}")
                                st.caption(f"Edge: {edge:+.1f}")
                        else:
                            st.caption("No data")
                    else:
                        st.caption("No data")

            # CONSENSUS (Both EDGE + ATS agree)
            if "consensus" in selected_markets:
                col_idx = selected_markets.index("consensus")
                with cols[col_idx]:
                    st.markdown('<div class="market-header">CONSENSUS</div>', unsafe_allow_html=True)
                    cons_df = all_predictions.get("consensus")
                    if cons_df is not None and not cons_df.empty:
                        row = cons_df[cons_df["game_id"] == game_id]
                        if not row.empty:
                            row = row.iloc[0]
                            # Safe NaN handling for display values
                            market_spread = row.get('market_spread')
                            market_spread = float(market_spread) if pd.notna(market_spread) else 0.0
                            model_edge = row.get('model_edge')
                            model_edge = float(model_edge) if pd.notna(model_edge) else 0.0
                            kelly = row.get('kelly')
                            kelly = float(kelly) if pd.notna(kelly) else 0.0
                            confidence = row.get('confidence', 'NONE')
                            consensus_reason = row.get('consensus_reason', '')

                            if row["bet_home"]:
                                home_team = info['home_team']
                                kelly_pct = kelly * kelly_multiplier
                                suggested_amount = account * kelly_pct
                                st.success(f"**{home_team}** ({confidence})")
                                st.caption(f"${suggested_amount:.0f} | {market_spread:+.1f}")
                                st.caption(f"EDGE + ATS agree")
                                with st.popover("Log Bet"):
                                    st.markdown(f"**{home_team}** covers {market_spread:+.1f}")
                                    st.markdown(f"**Consensus Reason:**")
                                    st.caption(consensus_reason)
                                    st.markdown(f"Confidence: {confidence}")
                                    cons_line = st.number_input("Your line", value=market_spread, step=0.5, key=f"cons_home_line_{game_id}")
                                    cons_odds = st.number_input("Your odds", value=-110, step=5, key=f"cons_home_odds_{game_id}")
                                    cons_amt = st.number_input("Amount ($)", value=float(suggested_amount) if suggested_amount > 0 else 100.0, step=10.0, key=f"cons_home_amt_{game_id}")
                                    if st.button(f"Log ${cons_amt:.0f} on {home_team} {cons_line:+.1f}", key=f"cons_home_log_{game_id}", type="primary", use_container_width=True):
                                        log_manual_bet(game_id, info['home_team'], info['away_team'], info['commence_time'],
                                                       "spread", "home", cons_odds, cons_amt, line=cons_line)
                                        st.success("Bet logged!")
                                        st.rerun()
                            elif row["bet_away"]:
                                away_team = info['away_team']
                                away_spread = -market_spread
                                kelly_pct = kelly * kelly_multiplier
                                suggested_amount = account * kelly_pct
                                st.success(f"**{away_team}** ({confidence})")
                                st.caption(f"${suggested_amount:.0f} | {away_spread:+.1f}")
                                st.caption(f"EDGE + ATS agree")
                                with st.popover("Log Bet"):
                                    st.markdown(f"**{away_team}** covers {away_spread:+.1f}")
                                    st.markdown(f"**Consensus Reason:**")
                                    st.caption(consensus_reason)
                                    st.markdown(f"Confidence: {confidence}")
                                    cons_line = st.number_input("Your line", value=away_spread, step=0.5, key=f"cons_away_line_{game_id}")
                                    cons_odds = st.number_input("Your odds", value=-110, step=5, key=f"cons_away_odds_{game_id}")
                                    cons_amt = st.number_input("Amount ($)", value=float(suggested_amount) if suggested_amount > 0 else 100.0, step=10.0, key=f"cons_away_amt_{game_id}")
                                    if st.button(f"Log ${cons_amt:.0f} on {away_team} {cons_line:+.1f}", key=f"cons_away_log_{game_id}", type="primary", use_container_width=True):
                                        log_manual_bet(game_id, info['home_team'], info['away_team'], info['commence_time'],
                                                       "spread", "away", cons_odds, cons_amt, line=cons_line)
                                        st.success("Bet logged!")
                                        st.rerun()
                            else:
                                # Show why no consensus
                                edge_home = row.get('edge_bet_home', False)
                                ats_home = row.get('ats_bet_home', False)
                                if edge_home and not ats_home:
                                    st.caption("EDGE only (no ATS)")
                                elif ats_home and not edge_home:
                                    st.caption("ATS only (no EDGE)")
                                else:
                                    st.caption("No signal")
                        else:
                            st.caption("No data")
                    else:
                        st.caption("No data")

            # EDGE (Point Spread Strategy)
            if "edge" in selected_markets:
                col_idx = selected_markets.index("edge")
                with cols[col_idx]:
                    st.markdown('<div class="market-header">EDGE (5+ PT)</div>', unsafe_allow_html=True)
                    edge_df = all_predictions.get("edge")
                    if edge_df is not None and not edge_df.empty:
                        row = edge_df[edge_df["game_id"] == game_id]
                        if not row.empty:
                            row = row.iloc[0]
                            # Safe NaN handling
                            market_spread = row.get('market_spread')
                            market_spread = float(market_spread) if pd.notna(market_spread) else 0.0
                            model_edge = row.get('model_edge')
                            model_edge = float(model_edge) if pd.notna(model_edge) else 0.0
                            kelly = row.get('kelly')
                            kelly = float(kelly) if pd.notna(kelly) else 0.0
                            filters_passed = row.get('filters_passed', '')
                            confidence = row.get('confidence', 'LOW')

                            if row["bet_home"]:
                                home_team = info['home_team']
                                kelly_pct = kelly * kelly_multiplier
                                suggested_amount = account * kelly_pct
                                st.success(f"**{home_team} {market_spread:+.1f}**")
                                st.caption(f"${suggested_amount:.0f} | Edge: {model_edge:+.1f}")
                                st.caption(f"{filters_passed}")
                                with st.popover("Log Bet"):
                                    st.markdown(f"**{home_team}** covers {market_spread:+.1f}")
                                    st.caption(f"Filters: {filters_passed}")
                                    edge_line = st.number_input("Your line", value=market_spread, step=0.5, key=f"edge_home_line_{game_id}")
                                    edge_odds = st.number_input("Your odds", value=-110, step=5, key=f"edge_home_odds_{game_id}")
                                    edge_amt = st.number_input("Amount ($)", value=float(suggested_amount) if suggested_amount > 0 else 100.0, step=10.0, key=f"edge_home_amt_{game_id}")
                                    if st.button(f"Log ${edge_amt:.0f} on {home_team} {edge_line:+.1f}", key=f"edge_home_log_{game_id}", type="primary", use_container_width=True):
                                        log_manual_bet(game_id, info['home_team'], info['away_team'], info['commence_time'],
                                                       "spread", "home", edge_odds, edge_amt, line=edge_line)
                                        st.success("Bet logged!")
                                        st.rerun()
                            elif row["bet_away"]:
                                away_team = info['away_team']
                                away_spread = -market_spread
                                kelly_pct = kelly * kelly_multiplier
                                suggested_amount = account * kelly_pct
                                st.success(f"**{away_team} {away_spread:+.1f}**")
                                st.caption(f"${suggested_amount:.0f} | Edge: {model_edge:+.1f}")
                                st.caption(f"{filters_passed}")
                                with st.popover("Log Bet"):
                                    st.markdown(f"**{away_team}** covers {away_spread:+.1f}")
                                    st.caption(f"Filters: {filters_passed}")
                                    edge_line = st.number_input("Your line", value=away_spread, step=0.5, key=f"edge_away_line_{game_id}")
                                    edge_odds = st.number_input("Your odds", value=-110, step=5, key=f"edge_away_odds_{game_id}")
                                    edge_amt = st.number_input("Amount ($)", value=float(suggested_amount) if suggested_amount > 0 else 100.0, step=10.0, key=f"edge_away_amt_{game_id}")
                                    if st.button(f"Log ${edge_amt:.0f} on {away_team} {edge_line:+.1f}", key=f"edge_away_log_{game_id}", type="primary", use_container_width=True):
                                        log_manual_bet(game_id, info['home_team'], info['away_team'], info['commence_time'],
                                                       "spread", "away", edge_odds, edge_amt, line=edge_line)
                                        st.success("Bet logged!")
                                        st.rerun()
                            else:
                                # Show edge info
                                st.caption(f"Edge: {model_edge:+.1f}")
                                st.caption(f"{confidence}")
                        else:
                            st.caption("No data")
                    else:
                        st.caption("No data")


def show_strategy_page():
    """Display validated betting strategy performance and backtest results."""
    st.header("Strategy Performance")

    # Strategy description
    st.markdown("""
    ### Validated Dual Model ATS Strategy

    This strategy uses the disagreement between MLP and XGBoost spread predictions
    to identify betting opportunities. When MLP predicts a significantly different spread
    than XGBoost, it signals potential mispricing.
    """)

    # Strategy parameters
    st.subheader("Strategy Parameters")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Min Disagreement", "4.0 pts", help="MLP must disagree with XGBoost by at least 4 points")
    with col2:
        st.metric("Min Edge vs Market", "2.0 pts", help="MLP prediction must differ from Elo market by at least 2 points")
    with col3:
        st.metric("Position Filter", "HOME Only", help="Only bet on home teams (historically more reliable)")

    st.markdown("---")

    # Multi-season validation results
    st.subheader("Multi-Season Validation (2020-2024)")

    # Hardcoded validation results from the backtest
    validation_data = [
        {"Season": 2020, "Bets": 64, "ATS Rate": "67.2%", "ROI": "+28.9%", "Status": "✓"},
        {"Season": 2021, "Bets": 61, "ATS Rate": "62.3%", "ROI": "+18.3%", "Status": "✓"},
        {"Season": 2022, "Bets": 55, "ATS Rate": "63.6%", "ROI": "+21.1%", "Status": "✓"},
        {"Season": 2023, "Bets": 52, "ATS Rate": "48.1%", "ROI": "-8.9%", "Status": "✗"},
        {"Season": 2024, "Bets": 48, "ATS Rate": "75.0%", "ROI": "+42.7%", "Status": "✓"},
    ]

    df_validation = pd.DataFrame(validation_data)

    # Summary metrics
    total_bets = sum(d["Bets"] for d in validation_data)
    total_wins = sum(d["Bets"] * float(d["ATS Rate"].replace("%", "")) / 100 for d in validation_data)
    overall_ats = total_wins / total_bets if total_bets > 0 else 0
    profitable_seasons = sum(1 for d in validation_data if d["Status"] == "✓")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Bets", f"{total_bets}")
    with col2:
        st.metric("Overall ATS Rate", f"{overall_ats:.1%}")
    with col3:
        st.metric("Avg ROI", "+19.6%")
    with col4:
        st.metric("Profitable Seasons", f"{profitable_seasons}/5")

    st.markdown("---")

    # Season breakdown table
    st.subheader("Season-by-Season Results")
    st.dataframe(
        df_validation,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Season": st.column_config.NumberColumn("Season", format="%d"),
            "Bets": st.column_config.NumberColumn("Bets"),
            "ATS Rate": st.column_config.TextColumn("ATS Win Rate"),
            "ROI": st.column_config.TextColumn("ROI"),
            "Status": st.column_config.TextColumn("Profitable"),
        }
    )

    # Visualization
    st.subheader("Performance Visualization")

    col1, col2 = st.columns(2)

    with col1:
        # ROI by season bar chart
        roi_values = [28.9, 18.3, 21.1, -8.9, 42.7]
        seasons = [2020, 2021, 2022, 2023, 2024]
        colors = ['green' if r > 0 else 'red' for r in roi_values]

        fig_roi = go.Figure(data=[
            go.Bar(
                x=seasons,
                y=roi_values,
                marker_color=colors,
                text=[f"{r:+.1f}%" for r in roi_values],
                textposition='outside'
            )
        ])
        fig_roi.update_layout(
            title="ROI by Season",
            xaxis_title="Season",
            yaxis_title="ROI (%)",
            yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='white'),
            showlegend=False,
            height=350
        )
        st.plotly_chart(fig_roi, use_container_width=True)

    with col2:
        # ATS rate by season
        ats_values = [67.2, 62.3, 63.6, 48.1, 75.0]

        fig_ats = go.Figure(data=[
            go.Bar(
                x=seasons,
                y=ats_values,
                marker_color=['green' if a > 52.4 else 'red' for a in ats_values],
                text=[f"{a:.1f}%" for a in ats_values],
                textposition='outside'
            )
        ])
        fig_ats.add_hline(y=52.4, line_dash="dash", line_color="yellow",
                         annotation_text="Breakeven (52.4%)")
        fig_ats.update_layout(
            title="ATS Win Rate by Season",
            xaxis_title="Season",
            yaxis_title="ATS Win Rate (%)",
            yaxis=dict(range=[0, 100]),
            showlegend=False,
            height=350
        )
        st.plotly_chart(fig_ats, use_container_width=True)

    # Strategy explanation
    st.markdown("---")
    st.subheader("How the Strategy Works")

    st.markdown("""
    1. **Dual Model Disagreement**: The MLP (neural network) and XGBoost models are trained
       on the same data but learn different patterns. When they disagree significantly,
       one model may have captured information the market is missing.

    2. **Edge vs Market**: We compare the MLP prediction to the Elo-based market expectation.
       A larger edge suggests higher potential value.

    3. **Home Team Filter**: Historical analysis shows the strategy performs significantly
       better on home teams (59-75% ATS) compared to away teams (43-50% ATS).

    4. **Kelly Criterion Sizing**: Bets are sized using fractional Kelly (15%) with a
       5% max bet cap to manage risk.
    """)

    # Key insight callout
    st.info("""
    **Key Insight**: The home-only filter dramatically improved results.
    HOME bets historically hit 59-75% ATS while AWAY bets only hit 43-50% ATS.
    By focusing only on home teams, we capture the profitable signal while avoiding losses.
    """)


def show_analytics_page(games, model_data):
    st.header("Analytics")

    # Run evaluation
    with st.spinner("Evaluating model..."):
        results = run_backtest(games, model_data, test_season=2024)

    if results is None:
        st.error("Could not evaluate model")
        return

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{results['accuracy']:.1%}")
    col2.metric("AUC-ROC", f"{results['auc']:.3f}")
    col3.metric("Brier Score", f"{results['brier']:.4f}")
    col4.metric("Test Games", results['total_games'])

    st.markdown("---")

    # Calibration plot
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Calibration Plot")

        probs = results['probs']
        actuals = results['actuals']

        # Bin predictions
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_indices = np.digitize(probs, bins) - 1
        bin_indices = np.clip(bin_indices, 0, 9)

        actual_rates = []
        pred_rates = []
        counts = []

        for i in range(10):
            mask = bin_indices == i
            if mask.sum() > 0:
                actual_rates.append(actuals[mask].mean())
                pred_rates.append(probs[mask].mean())
                counts.append(mask.sum())
            else:
                actual_rates.append(np.nan)
                pred_rates.append(np.nan)
                counts.append(0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(dash='dash', color='gray')
        ))
        fig.add_trace(go.Scatter(
            x=pred_rates,
            y=actual_rates,
            mode='markers+lines',
            name='Model',
            marker=dict(size=10),
            text=[f'n={c}' for c in counts],
            hovertemplate='Predicted: %{x:.1%}<br>Actual: %{y:.1%}<br>%{text}'
        ))
        fig.update_layout(
            xaxis_title='Predicted Probability',
            yaxis_title='Actual Win Rate',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Prediction Distribution")

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=probs,
            nbinsx=30,
            name='All Predictions'
        ))
        fig.update_layout(
            xaxis_title='Predicted Home Win Probability',
            yaxis_title='Count',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # ROC Curve
    st.subheader("ROC Curve")
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(actuals, probs)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'Model (AUC={results["auc"]:.3f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                             line=dict(dash='dash'), name='Random'))
    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
