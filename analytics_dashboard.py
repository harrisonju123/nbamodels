"""
NBA Betting Dashboard - Simplified

One dashboard, three tabs:
1. Today - Today's picks
2. Performance - Simple P&L tracking
3. History - Past bets

Run with: streamlit run analytics_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.bet_tracker import get_bet_history, get_performance_summary
from src.utils.constants import BETS_DB_PATH

# Page configuration
st.set_page_config(
    page_title="NBA Betting Dashboard",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 1rem 2rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 20px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 10px 0;
    }

    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
    }

    .positive {
        color: #10b981;
    }

    .negative {
        color: #ef4444;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🏀 NBA Betting Dashboard")
st.markdown("---")

# Load data
@st.cache_data(ttl=60)
def load_bets():
    """Load bets from database."""
    df = get_bet_history()
    if df.empty:
        return df

    # Convert timestamps
    df['logged_at'] = pd.to_datetime(df['logged_at'], format='ISO8601', utc=True)
    df['commence_time'] = pd.to_datetime(df['commence_time'], format='ISO8601', utc=True)

    return df

df_all = load_bets()

# Quick stats at top
col1, col2, col3, col4 = st.columns(4)

if not df_all.empty:
    settled = df_all[df_all['outcome'].notna()]

    if len(settled) > 0:
        total_profit = settled['profit'].sum()
        total_wagered = settled['bet_amount'].sum()
        roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0
        win_rate = (settled['outcome'] == 'win').sum() / len(settled) * 100

        with col1:
            st.metric("Total Profit", f"${total_profit:,.2f}",
                     delta_color="normal" if total_profit >= 0 else "inverse")

        with col2:
            st.metric("ROI", f"{roi:.1f}%",
                     delta_color="normal" if roi >= 0 else "inverse")

        with col3:
            st.metric("Win Rate", f"{win_rate:.1f}%")

        with col4:
            st.metric("Total Bets", f"{len(settled)}")
else:
    st.info("No bets found. Start betting to see stats!")

st.markdown("---")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🎲 Today", "📈 Performance", "📊 History"])

# ============================================================================
# TAB 1: TODAY'S PICKS
# ============================================================================
with tab1:
    st.markdown("### Today's Betting Picks")

    try:
        from src.prediction_cache import load_cached_predictions, get_cache_info, refresh_predictions
        from src.bet_tracker import get_regime_status

        # Load predictions
        predictions = load_cached_predictions()
        cache_info = get_cache_info()

        # Header with refresh
        col1, col2 = st.columns([4, 1])
        with col1:
            if cache_info:
                st.caption(f"📅 Updated: {cache_info.get('timestamp', 'Unknown')[:16]} | {cache_info.get('num_games', 0)} games")
            else:
                st.caption("No cached predictions available")

        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                with st.spinner("Refreshing..."):
                    predictions = refresh_predictions(min_edge=0.02)
                    st.success("Refreshed!")
                    st.rerun()

        # Check for predictions
        has_predictions = bool(predictions and any(
            df is not None and not df.empty
            for df in predictions.values()
        ))

        if not has_predictions:
            st.info("No predictions available. Click Refresh to load today's picks.")
        else:
            # Display picks
            from src.dashboard_enhancements import enhanced_pick_display

            for game_id, pred_df in predictions.items():
                if pred_df is not None and not pred_df.empty:
                    enhanced_pick_display(pred_df, game_id)

    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        st.info("To see today's picks, run the betting pipeline first.")

# ============================================================================
# TAB 2: PERFORMANCE
# ============================================================================
with tab2:
    st.markdown("### Performance Overview")

    if df_all.empty:
        st.info("No bets to analyze yet.")
    else:
        settled = df_all[df_all['outcome'].notna()].copy()

        if len(settled) == 0:
            st.info("No settled bets yet. Check back after games complete.")
        else:
            # Performance metrics
            st.markdown("#### Key Metrics")

            col1, col2, col3 = st.columns(3)

            wins = (settled['outcome'] == 'win').sum()
            losses = (settled['outcome'] == 'loss').sum()
            pushes = (settled['outcome'] == 'push').sum()

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Record</div>
                    <div class="metric-value">{wins}-{losses}-{pushes}</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                total_wagered = settled['bet_amount'].sum()
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Wagered</div>
                    <div class="metric-value">${total_wagered:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                avg_bet = settled['bet_amount'].mean()
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Avg Bet Size</div>
                    <div class="metric-value">${avg_bet:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### Cumulative Profit")

            # Profit chart
            settled_sorted = settled.sort_values('logged_at').copy()
            settled_sorted['cumulative_profit'] = settled_sorted['profit'].fillna(0).cumsum()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=settled_sorted['logged_at'],
                y=settled_sorted['cumulative_profit'],
                mode='lines',
                name='Cumulative Profit',
                line=dict(color='#667eea', width=3),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.1)'
            ))

            fig.update_layout(
                template='plotly_dark',
                height=400,
                margin=dict(l=0, r=0, t=30, b=0),
                hovermode='x unified',
                yaxis_title="Profit ($)",
                xaxis_title="Date"
            )

            st.plotly_chart(fig, use_container_width=True)

            # Recent performance
            st.markdown("#### Recent Bets (Last 10)")

            recent = settled.nlargest(10, 'logged_at')[
                ['commence_time', 'away_team', 'home_team', 'bet_side', 'line',
                 'bet_amount', 'outcome', 'profit']
            ].copy()

            # Format for display
            recent['commence_time'] = recent['commence_time'].dt.strftime('%m/%d')
            recent['bet_amount'] = recent['bet_amount'].apply(lambda x: f"${x:.0f}")
            recent['profit'] = recent['profit'].apply(
                lambda x: f"${x:+.2f}" if pd.notna(x) else "-"
            )

            st.dataframe(
                recent,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "commence_time": "Date",
                    "away_team": "Away",
                    "home_team": "Home",
                    "bet_side": "Side",
                    "line": "Line",
                    "bet_amount": "Amount",
                    "outcome": "Result",
                    "profit": "Profit"
                }
            )

# ============================================================================
# TAB 3: HISTORY
# ============================================================================
with tab3:
    st.markdown("### Bet History")

    if df_all.empty:
        st.info("No bets found.")
    else:
        # Filters
        col1, col2, col3 = st.columns([2, 2, 2])

        with col1:
            date_filter = st.selectbox(
                "Time Period",
                ["All Time", "Last 7 Days", "Last 30 Days", "Last 90 Days"]
            )

        with col2:
            outcome_filter = st.multiselect(
                "Outcome",
                ["win", "loss", "push", "pending"],
                default=[]
            )

        with col3:
            team_filter = st.text_input("Team Name", "")

        # Apply filters
        df_filtered = df_all.copy()

        if date_filter != "All Time":
            days_map = {"Last 7 Days": 7, "Last 30 Days": 30, "Last 90 Days": 90}
            cutoff = datetime.now(df_filtered['logged_at'].dt.tz) - timedelta(days=days_map[date_filter])
            df_filtered = df_filtered[df_filtered['logged_at'] >= cutoff]

        if outcome_filter:
            df_filtered = df_filtered[df_filtered['outcome'].isin(outcome_filter)]

        if team_filter:
            mask = (
                df_filtered['home_team'].str.contains(team_filter, case=False, na=False) |
                df_filtered['away_team'].str.contains(team_filter, case=False, na=False)
            )
            df_filtered = df_filtered[mask]

        # Display table
        st.caption(f"Showing {len(df_filtered)} bets")

        display_cols = ['logged_at', 'commence_time', 'away_team', 'home_team',
                       'bet_side', 'line', 'bet_amount', 'odds', 'outcome', 'profit']

        df_display = df_filtered[display_cols].copy()
        df_display['logged_at'] = df_display['logged_at'].dt.strftime('%m/%d %H:%M')
        df_display['commence_time'] = df_display['commence_time'].dt.strftime('%m/%d %H:%M')
        df_display['bet_amount'] = df_display['bet_amount'].apply(lambda x: f"${x:.0f}")
        df_display['profit'] = df_display['profit'].apply(
            lambda x: f"${x:+.2f}" if pd.notna(x) else "-"
        )

        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "logged_at": "Logged",
                "commence_time": "Game",
                "away_team": "Away",
                "home_team": "Home",
                "bet_side": "Side",
                "line": "Line",
                "bet_amount": "Amount",
                "odds": "Odds",
                "outcome": "Result",
                "profit": "Profit"
            }
        )

        # Export button
        if st.button("📥 Export to CSV"):
            csv = df_filtered.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"bets_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
