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

    # Ensure strategy_type column exists (for backwards compatibility)
    if 'strategy_type' not in df.columns:
        df['strategy_type'] = 'spread'  # Default for old bets

    return df

df_all = load_bets()

# Strategy filter (sidebar)
with st.sidebar:
    st.markdown("### 🎯 Strategy Filter")

    if not df_all.empty and 'strategy_type' in df_all.columns:
        available_strategies = ['All'] + sorted(df_all['strategy_type'].dropna().unique().tolist())
        strategy_filter = st.selectbox(
            "Select Strategy",
            available_strategies,
            help="Filter bets by strategy type"
        )

        # Apply filter
        if strategy_filter != 'All':
            df_filtered = df_all[df_all['strategy_type'] == strategy_filter].copy()
            st.caption(f"Showing {len(df_filtered)} {strategy_filter} bets")
        else:
            df_filtered = df_all.copy()
            st.caption(f"Showing all {len(df_filtered)} bets")
    else:
        strategy_filter = 'All'
        df_filtered = df_all.copy()

    st.markdown("---")
    st.caption("💡 **Strategies Available:**")
    st.caption("• Spread - Point spread betting")
    st.caption("• Totals - Over/Under betting")
    st.caption("• Arbitrage - Cross-book arbs")
    st.caption("• Player Props - Player stats")
else:
    df_filtered = df_all.copy()
    strategy_filter = 'All'

# Quick stats at top
col1, col2, col3, col4 = st.columns(4)

if not df_filtered.empty:
    settled = df_filtered[df_filtered['outcome'].notna()]

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

    # Show strategy filter status
    if strategy_filter != 'All':
        st.info(f"📊 Viewing **{strategy_filter}** strategy only. Change filter in sidebar to see all bets.")
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
        # Strategy Breakdown (always show all strategies here)
        if 'strategy_type' in df_all.columns:
            st.markdown("#### 🎯 Performance by Strategy")

            settled_all = df_all[df_all['outcome'].notna()].copy()

            if len(settled_all) > 0:
                # Group by strategy
                strategy_stats = settled_all.groupby('strategy_type').agg({
                    'profit': 'sum',
                    'bet_amount': ['sum', 'count'],
                    'outcome': lambda x: (x == 'win').sum() / len(x) * 100 if len(x) > 0 else 0
                }).round(2)

                strategy_stats.columns = ['Profit', 'Wagered', 'Bets', 'Win Rate']
                strategy_stats['ROI %'] = (strategy_stats['Profit'] / strategy_stats['Wagered'] * 100).round(1)

                # Reorder columns
                strategy_stats = strategy_stats[['Bets', 'Wagered', 'Profit', 'ROI %', 'Win Rate']]

                # Format for display
                display_stats = strategy_stats.copy()
                display_stats['Wagered'] = display_stats['Wagered'].apply(lambda x: f"${x:,.0f}")
                display_stats['Profit'] = display_stats['Profit'].apply(
                    lambda x: f"${x:,.2f}" if x >= 0 else f"-${abs(x):,.2f}"
                )
                display_stats['Win Rate'] = display_stats['Win Rate'].apply(lambda x: f"{x:.1f}%")
                display_stats['ROI %'] = display_stats['ROI %'].apply(
                    lambda x: f"{x:.1f}%" if x >= 0 else f"{x:.1f}%"
                )

                st.dataframe(display_stats, use_container_width=True)

                # Best performing strategy
                best_strategy = strategy_stats['ROI %'].idxmax()
                best_roi = strategy_stats.loc[best_strategy, 'ROI %']
                st.success(f"🏆 Best performing: **{best_strategy}** ({best_roi:.1f}% ROI)")

            st.markdown("---")

        # Use filtered data for detailed metrics
        settled = df_filtered[df_filtered['outcome'].notna()].copy()

        if len(settled) == 0:
            st.info("No settled bets yet. Check back after games complete.")
        else:
            # Calculate metrics
            wins = (settled['outcome'] == 'win').sum()
            losses = (settled['outcome'] == 'loss').sum()
            pushes = (settled['outcome'] == 'push').sum()
            total_profit = settled['profit'].sum()
            total_wagered = settled['bet_amount'].sum()
            roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0
            win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0

            # Calculate advanced metrics
            settled_sorted = settled.sort_values('logged_at').copy()
            settled_sorted['cumulative_profit'] = settled_sorted['profit'].fillna(0).cumsum()

            # Sharpe Ratio (annualized)
            returns = settled_sorted['profit'] / settled_sorted['bet_amount']
            sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

            # Max Drawdown
            cumulative = settled_sorted['cumulative_profit']
            running_max = cumulative.cummax()
            drawdown = cumulative - running_max
            max_drawdown = drawdown.min()
            max_drawdown_pct = (max_drawdown / running_max.max() * 100) if running_max.max() > 0 else 0

            # Current drawdown
            current_drawdown = cumulative.iloc[-1] - running_max.iloc[-1]
            current_drawdown_pct = (current_drawdown / running_max.iloc[-1] * 100) if running_max.iloc[-1] > 0 else 0

            # CLV metrics (if available)
            has_clv = 'clv' in settled.columns and settled['clv'].notna().any()
            if has_clv:
                avg_clv = settled['clv'].mean()
                clv_positive_pct = (settled['clv'] > 0).sum() / len(settled) * 100

            # ALERTS
            alerts = []
            if current_drawdown_pct < -10:
                alerts.append(("⚠️ Large Drawdown", f"Currently down {abs(current_drawdown_pct):.1f}% from peak"))
            if len(settled) >= 30 and roi < 5:
                alerts.append(("⚠️ Low ROI", f"ROI of {roi:.1f}% below backtest expectations"))
            if len(settled) >= 30 and win_rate < 52:
                alerts.append(("⚠️ Low Win Rate", f"Win rate of {win_rate:.1f}% below 52% break-even"))
            if has_clv and avg_clv < -0.01:
                alerts.append(("⚠️ Negative CLV", f"Average CLV of {avg_clv:.2f} indicates poor bet timing"))

            # Display alerts
            if alerts:
                for title, msg in alerts:
                    st.warning(f"**{title}**: {msg}")
            else:
                st.success("✅ All metrics healthy")

            st.markdown("---")

            # Performance vs Backtest
            st.markdown("#### 📊 Live vs Backtest")

            backtest_roi = 21.0  # From proper backtest
            backtest_win_rate = 63.4

            col1, col2 = st.columns(2)

            with col1:
                roi_diff = roi - backtest_roi
                roi_color = "green" if roi_diff >= 0 else "red"
                st.markdown(f"""
                **ROI Comparison**
                - Live: **{roi:.1f}%**
                - Backtest: {backtest_roi:.1f}%
                - Difference: <span style='color:{roi_color}'>**{roi_diff:+.1f}%**</span>
                """, unsafe_allow_html=True)

                if len(settled) < 100:
                    st.caption(f"⚠️ Small sample ({len(settled)} bets). Need 200+ for statistical significance.")

            with col2:
                wr_diff = win_rate - backtest_win_rate
                wr_color = "green" if wr_diff >= 0 else "red"
                st.markdown(f"""
                **Win Rate Comparison**
                - Live: **{win_rate:.1f}%**
                - Backtest: {backtest_win_rate:.1f}%
                - Difference: <span style='color:{wr_color}'>**{wr_diff:+.1f}%**</span>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # Advanced Risk Metrics
            st.markdown("#### 📈 Risk Metrics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                sharpe_color = "green" if sharpe > 1 else ("orange" if sharpe > 0.5 else "red")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Sharpe Ratio</div>
                    <div class="metric-value" style="color:{sharpe_color}">{sharpe:.2f}</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                dd_color = "green" if max_drawdown_pct > -15 else ("orange" if max_drawdown_pct > -25 else "red")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value" style="color:{dd_color}">{max_drawdown_pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                profit_factor = (settled[settled['profit'] > 0]['profit'].sum() /
                                abs(settled[settled['profit'] < 0]['profit'].sum())) if (settled['profit'] < 0).any() else float('inf')
                pf_display = f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞"
                pf_color = "green" if profit_factor > 1.5 else ("orange" if profit_factor > 1 else "red")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Profit Factor</div>
                    <div class="metric-value" style="color:{pf_color}">{pf_display}</div>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Record</div>
                    <div class="metric-value">{wins}-{losses}-{pushes}</div>
                </div>
                """, unsafe_allow_html=True)

            # CLV Analysis
            if has_clv:
                st.markdown("---")
                st.markdown("#### 💎 Bet Quality (CLV)")

                col1, col2, col3 = st.columns(3)

                with col1:
                    clv_color = "green" if avg_clv > 0.01 else ("orange" if avg_clv > -0.01 else "red")
                    st.metric("Average CLV", f"{avg_clv:.2%}",
                             help="Closing Line Value - measures bet timing quality")

                with col2:
                    st.metric("Positive CLV %", f"{clv_positive_pct:.1f}%",
                             help="% of bets that beat the closing line")

                with col3:
                    max_clv = settled['clv'].max()
                    st.metric("Best CLV", f"{max_clv:.2%}",
                             help="Best single bet timing")

            st.markdown("---")
            st.markdown("#### 💰 Cumulative Profit")

            # Profit chart with drawdown shading
            fig = go.Figure()

            # Add cumulative profit line
            fig.add_trace(go.Scatter(
                x=settled_sorted['logged_at'],
                y=settled_sorted['cumulative_profit'],
                mode='lines',
                name='Cumulative Profit',
                line=dict(color='#667eea', width=3),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.1)'
            ))

            # Add peak line
            fig.add_trace(go.Scatter(
                x=settled_sorted['logged_at'],
                y=running_max,
                mode='lines',
                name='Peak',
                line=dict(color='gray', width=1, dash='dash'),
                opacity=0.5
            ))

            fig.update_layout(
                template='plotly_dark',
                height=400,
                margin=dict(l=0, r=0, t=30, b=0),
                hovermode='x unified',
                yaxis_title="Profit ($)",
                xaxis_title="Date",
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

            # Recent performance
            st.markdown("#### 🎯 Recent Bets (Last 10)")

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
