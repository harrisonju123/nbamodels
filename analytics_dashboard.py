"""
Advanced Analytics Dashboard for NBA Betting Model

Features:
- Real-time performance tracking
- CLV (Closing Line Value) analysis
- Model calibration and accuracy metrics
- Interactive visualizations
- Bankroll management analytics

Run with: streamlit run analytics_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.bet_tracker import (
    get_bet_history,
    get_performance_summary,
    get_enhanced_clv_summary,
    get_performance_decay_metrics,
)
from src.utils.constants import BETS_DB_PATH
from src.dashboard_enhancements import (
    kelly_calculator_widget,
    enhanced_pick_display,
    alternative_data_status_summary,
)

# Page configuration
st.set_page_config(
    page_title="NBA Betting Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 20px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
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
        letter-spacing: 1px;
    }

    .positive {
        color: #10b981;
    }

    .negative {
        color: #ef4444;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        background-color: #1f2937;
        border-radius: 8px 8px 0 0;
    }

    .stTabs [aria-selected="true"] {
        background-color: #374151;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("📊 Analytics Dashboard")
    st.markdown("---")

    # Date range filter
    st.subheader("Filters")

    date_range = st.selectbox(
        "Time Period",
        ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time", "Custom"],
        index=2  # Default to "Last 90 Days" to show synthetic historical bets
    )

    if date_range == "Custom":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
        with col2:
            end_date = st.date_input("End Date", datetime.now())
    else:
        if date_range == "Last 7 Days":
            days = 7
        elif date_range == "Last 30 Days":
            days = 30
        elif date_range == "Last 90 Days":
            days = 90
        else:
            days = None

        if days:
            start_date = datetime.now() - timedelta(days=days)
            end_date = datetime.now()
        else:
            start_date = None
            end_date = None

    # Bet type filter
    bet_types = st.multiselect(
        "Bet Types",
        ["spread", "totals", "moneyline"],
        default=["spread"]
    )

    # Kelly Calculator
    st.markdown("---")
    bankroll, kelly_fraction = kelly_calculator_widget(default_bankroll=1000.0)

    # Refresh button
    st.markdown("---")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()

# Helper functions
@st.cache_data(ttl=60)
def load_bets_data(start_date=None, end_date=None):
    """Load bets from database with caching."""
    # Limit to prevent memory issues with large datasets
    # Dashboard shows recent data - no need to load entire history
    df_all = get_bet_history()

    # Filter out test bets (IDs starting with 'test_')
    if not df_all.empty and 'id' in df_all.columns:
        df = df_all[~df_all['id'].str.startswith('test_', na=False)].copy()
        test_bet_count = len(df_all) - len(df)
        if test_bet_count > 0:
            st.sidebar.caption(f"📊 Filtered {test_bet_count} test bets from display")
    else:
        df = df_all

    # Hard limit to last 10,000 bets for performance
    if len(df) > 10000:
        df = df.nlargest(10000, 'logged_at')

    if df.empty:
        return df

    # Convert date strings to datetime (handle ISO8601 format with timezone)
    df['logged_at'] = pd.to_datetime(df['logged_at'], format='ISO8601', utc=True)

    # Filter by date range (ensure timezone-aware for comparison)
    if start_date:
        start_dt = pd.to_datetime(start_date, utc=True)
        df = df[df['logged_at'] >= start_dt]
    if end_date:
        end_dt = pd.to_datetime(end_date, utc=True)
        df = df[df['logged_at'] <= end_dt]

    return df

def calculate_cumulative_profit(df):
    """Calculate cumulative profit over time."""
    if df.empty:
        return pd.DataFrame()

    df_sorted = df.sort_values('logged_at').copy()
    df_sorted['cumulative_profit'] = df_sorted['profit'].fillna(0).cumsum()
    df_sorted['cumulative_roi'] = (df_sorted['cumulative_profit'] / df_sorted['bet_amount'].cumsum() * 100)

    return df_sorted

def create_performance_summary_cards(df):
    """Create metric cards for key performance indicators."""
    if df.empty:
        st.warning("No bets found for the selected period.")
        return

    # Calculate metrics
    settled_bets = df[df['outcome'].notna()]
    total_bets = len(settled_bets)

    if total_bets == 0:
        st.info("No settled bets in this period.")
        return

    wins = len(settled_bets[settled_bets['outcome'] == 'win'])
    losses = len(settled_bets[settled_bets['outcome'] == 'loss'])
    pushes = len(settled_bets[settled_bets['outcome'] == 'push'])

    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

    total_wagered = settled_bets['bet_amount'].sum()
    total_profit = settled_bets['profit'].sum()
    roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0

    avg_clv = settled_bets['clv'].mean() if 'clv' in settled_bets.columns else 0

    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Win Rate",
            value=f"{win_rate:.1f}%",
            delta=f"{wins}W-{losses}L-{pushes}P"
        )

    with col2:
        profit_color = "normal" if total_profit >= 0 else "inverse"
        st.metric(
            label="Total Profit",
            value=f"${total_profit:,.2f}",
            delta=f"{total_bets} bets",
            delta_color=profit_color
        )

    with col3:
        roi_color = "normal" if roi >= 0 else "inverse"
        st.metric(
            label="ROI",
            value=f"{roi:.2f}%",
            delta=f"${total_wagered:,.0f} wagered",
            delta_color=roi_color
        )

    with col4:
        clv_color = "normal" if avg_clv >= 0 else "inverse"
        st.metric(
            label="Avg CLV",
            value=f"{avg_clv:.2f}%",
            delta="Closing Line Value",
            delta_color=clv_color
        )

def create_profit_curve_chart(df):
    """Create cumulative profit curve."""
    df_cum = calculate_cumulative_profit(df)

    if df_cum.empty:
        return None

    fig = go.Figure()

    # Cumulative profit line
    fig.add_trace(go.Scatter(
        x=df_cum['logged_at'],
        y=df_cum['cumulative_profit'],
        mode='lines',
        name='Cumulative Profit',
        line=dict(color='#10b981', width=3),
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.1)'
    ))

    # Break-even line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title="Cumulative Profit Over Time",
        xaxis_title="Date",
        yaxis_title="Profit ($)",
        template="plotly_dark",
        hovermode='x unified',
        height=400
    )

    return fig

def create_win_rate_trend_chart(df):
    """Create rolling win rate trend chart."""
    if df.empty or len(df) < 10:
        return None

    df_sorted = df[df['outcome'].notna()].sort_values('logged_at').copy()

    # Calculate rolling win rate (20-bet window)
    window = min(20, len(df_sorted) // 5)
    df_sorted['is_win'] = (df_sorted['outcome'] == 'win').astype(int)
    df_sorted['rolling_win_rate'] = df_sorted['is_win'].rolling(window=window, min_periods=1).mean() * 100

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_sorted['logged_at'],
        y=df_sorted['rolling_win_rate'],
        mode='lines',
        name=f'{window}-Bet Rolling Win Rate',
        line=dict(color='#3b82f6', width=2)
    ))

    # Break-even line (52.38% for -110 odds)
    fig.add_hline(y=52.38, line_dash="dash", line_color="yellow",
                  annotation_text="Break-even (52.38%)", opacity=0.7)

    fig.update_layout(
        title=f"Win Rate Trend ({window}-Bet Rolling Average)",
        xaxis_title="Date",
        yaxis_title="Win Rate (%)",
        template="plotly_dark",
        hovermode='x unified',
        height=400
    )

    return fig

def create_clv_distribution_chart(df):
    """Create CLV distribution histogram."""
    if df.empty or 'clv' not in df.columns:
        return None

    df_clv = df[df['clv'].notna()].copy()

    if len(df_clv) < 5:
        return None

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=df_clv['clv'],
        nbinsx=30,
        name='CLV Distribution',
        marker_color='#8b5cf6',
        opacity=0.7
    ))

    # Add mean line
    mean_clv = df_clv['clv'].mean()
    fig.add_vline(x=mean_clv, line_dash="dash", line_color="green",
                  annotation_text=f"Mean: {mean_clv:.2f}%", opacity=0.8)

    fig.update_layout(
        title="Closing Line Value (CLV) Distribution",
        xaxis_title="CLV (%)",
        yaxis_title="Count",
        template="plotly_dark",
        showlegend=False,
        height=400
    )

    return fig

def create_clv_vs_profit_scatter(df):
    """Create scatter plot of CLV vs actual profit."""
    if df.empty or 'clv' not in df.columns:
        return None

    df_settled = df[(df['outcome'].notna()) & (df['clv'].notna())].copy()

    if len(df_settled) < 10:
        return None

    # Color by outcome
    df_settled['color'] = df_settled['outcome'].map({
        'win': 'green',
        'loss': 'red',
        'push': 'gray'
    })

    fig = go.Figure()

    for outcome in ['win', 'loss', 'push']:
        df_outcome = df_settled[df_settled['outcome'] == outcome]
        if len(df_outcome) == 0:
            continue

        fig.add_trace(go.Scatter(
            x=df_outcome['clv'],
            y=df_outcome['profit'],
            mode='markers',
            name=outcome.capitalize(),
            marker=dict(
                size=8,
                color=df_outcome['color'],
                opacity=0.6
            )
        ))

    fig.update_layout(
        title="CLV vs Profit Relationship",
        xaxis_title="CLV (%)",
        yaxis_title="Profit ($)",
        template="plotly_dark",
        hovermode='closest',
        height=400
    )

    return fig

def create_performance_by_edge_chart(df):
    """Create performance breakdown by model edge."""
    if df.empty or 'edge' not in df.columns:
        return None

    df_settled = df[df['outcome'].notna()].copy()

    if len(df_settled) < 10:
        return None

    # Create edge bins (6 edges = 5 bins, so need exactly 5 labels)
    df_settled['edge_bin'] = pd.cut(
        df_settled['edge'] * 100,
        bins=[0, 5, 7, 10, 15, 100],
        labels=['0-5%', '5-7%', '7-10%', '10-15%', '15%+']
    )

    # Calculate win rate by edge bin
    edge_performance = df_settled.groupby('edge_bin', observed=True).agg({
        'outcome': lambda x: (x == 'win').sum() / len(x) * 100,
        'profit': 'sum',
        'bet_amount': 'count'
    }).reset_index()

    edge_performance.columns = ['Edge Bin', 'Win Rate (%)', 'Profit', 'Bet Count']

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=edge_performance['Edge Bin'],
            y=edge_performance['Win Rate (%)'],
            name='Win Rate',
            marker_color='#3b82f6'
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=edge_performance['Edge Bin'],
            y=edge_performance['Bet Count'],
            name='Bet Count',
            mode='lines+markers',
            marker_color='#f59e0b',
            line=dict(width=3)
        ),
        secondary_y=True
    )

    fig.update_layout(
        title="Performance by Model Edge",
        template="plotly_dark",
        hovermode='x unified',
        height=400
    )

    fig.update_yaxes(title_text="Win Rate (%)", secondary_y=False)
    fig.update_yaxes(title_text="Bet Count", secondary_y=True)

    return fig

def create_daily_performance_heatmap(df):
    """Create heatmap of performance by day of week and hour."""
    if df.empty:
        return None

    df_settled = df[df['outcome'].notna()].copy()

    if len(df_settled) < 20:
        return None

    # logged_at is already datetime from load_bets_data()
    df_settled['day_of_week'] = df_settled['logged_at'].dt.day_name()
    df_settled['hour'] = df_settled['logged_at'].dt.hour

    # Calculate win rate by day and hour
    pivot = df_settled.groupby(['day_of_week', 'hour']).agg({
        'outcome': lambda x: (x == 'win').sum() / len(x) * 100 if len(x) > 0 else 0
    }).reset_index()

    pivot_table = pivot.pivot(index='day_of_week', columns='hour', values='outcome')

    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_table = pivot_table.reindex([d for d in day_order if d in pivot_table.index])

    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        colorscale='RdYlGn',
        zmid=52.38,  # Break-even point
        text=pivot_table.values.round(1),
        texttemplate='%{text}%',
        textfont={"size": 10},
        colorbar=dict(title="Win Rate (%)")
    ))

    fig.update_layout(
        title="Win Rate Heatmap by Day & Hour",
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        template="plotly_dark",
        height=400
    )

    return fig

# Main dashboard
st.title("📊 Advanced NBA Betting Analytics")
st.markdown("Real-time performance tracking and comprehensive analytics")

# Historical Backtest Results Banner
with st.expander("🎯 **Model Validation Results** - Optimized Edge Threshold", expanded=True):
    st.markdown("""
    ### 📊 Validated Performance (64 Synthetic Bets, Nov 2025 - Jan 2026)

    Using **real historical odds** and simulated outcomes to validate optimal edge threshold:
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **6%+ Edge Strategy** ⭐ ACTIVE
        - **Win Rate:** 64% (Expected)
        - **ROI:** +23%
        - **Current Threshold:** 6% minimum
        - **Status:** ✅ Optimized
        """)

    with col2:
        st.markdown("""
        **Sweet Spot: 8-10% Edge**
        - **Win Rate:** 75% 🔥
        - **ROI:** +47.5%
        - **Bets:** 20/64 in sweet spot
        - Highest performer
        """)

    with col3:
        st.markdown("""
        **⚠️ WARNING: 5-6% Edge**
        - **Win Rate:** 45% (LOSING!)
        - **ROI:** -11.7%
        - **DO NOT BET** below 6% edge
        - These bets lose money
        """)

    st.markdown("---")
    st.markdown("""
    **Key Insight:** Threshold raised from 5% to **6% based on synthetic validation**.
    5-6% edge bets lose money (-11.7% ROI). Current pipeline uses **6% minimum edge**.

    📄 Full details: `EDGE_THRESHOLD_UPDATE.md` | 📊 Synthetic bets: `SYNTHETIC_BETS_GUIDE.md`
    """)

# Load data
df_bets = load_bets_data(start_date, end_date)

# Filter by bet type
if bet_types and not df_bets.empty:
    df_bets = df_bets[df_bets['bet_type'].isin(bet_types)]

# Performance summary cards
st.markdown("### 📈 Performance Overview")
create_performance_summary_cards(df_bets)

# Create tabs for different analytics sections
tab1, tab2, tab3, tab4, tab5, tab_live = st.tabs([
    "📊 Performance Analytics",
    "💰 CLV Analysis",
    "🎯 Model Performance",
    "📅 Time-based Analytics",
    "🎲 Today's Picks",
    "🔴 Live Betting"
])

with tab1:
    # Check for settled vs unsettled bets
    if not df_bets.empty:
        settled_bets = df_bets[df_bets['outcome'].notna()]
        unsettled_bets = df_bets[df_bets['outcome'].isna()]

        if len(settled_bets) == 0 and len(unsettled_bets) > 0:
            st.info(f"""
            📊 **{len(unsettled_bets)} unsettled bets** found. Performance charts will appear once games complete.

            **To settle bets manually:**
            ```bash
            python scripts/settle_bets.py
            ```

            **Current unsettled bets:** {len(unsettled_bets)} waiting for game results
            """)

    # Data Quality Banner
    if not df_bets.empty:
        corrected_bets = len(df_bets[(df_bets['market_prob'] < 0.49) | (df_bets['market_prob'] > 0.51)])
        total_bets = len(df_bets)
        accuracy_pct = (corrected_bets / total_bets * 100) if total_bets > 0 else 0

        if accuracy_pct < 100:
            with st.expander("📊 **Data Quality Status** - Click to improve accuracy", expanded=False):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"""
                    **Current Status:** {corrected_bets}/{total_bets} bets ({accuracy_pct:.1f}%) have historically accurate odds.

                    **Why this matters:**
                    - Market % at 50%* is estimated, not actual odds
                    - Edge % calculations are approximate
                    - Can't evaluate true model performance

                    **How to fix:**
                    Run the backfill script to fetch historical odds from The Odds API:
                    ```bash
                    python scripts/backfill_historical_odds.py
                    ```

                    This will update your bets with actual market odds from the games.
                    """)

                with col2:
                    st.metric(
                        "Data Accuracy",
                        f"{accuracy_pct:.0f}%",
                        delta=f"{100 - accuracy_pct:.0f}% to go",
                        delta_color="inverse"
                    )

    st.markdown("### Performance Trends")

    col1, col2 = st.columns(2)

    with col1:
        profit_chart = create_profit_curve_chart(df_bets)
        if profit_chart:
            st.plotly_chart(profit_chart, use_container_width=True)
        else:
            st.info("Not enough data for profit curve")

    with col2:
        winrate_chart = create_win_rate_trend_chart(df_bets)
        if winrate_chart:
            st.plotly_chart(winrate_chart, use_container_width=True)
        else:
            st.info("Not enough data for win rate trend")

    # Performance by edge
    st.markdown("### Performance by Model Edge")
    edge_chart = create_performance_by_edge_chart(df_bets)
    if edge_chart:
        st.plotly_chart(edge_chart, use_container_width=True)
    else:
        st.info("Not enough data for edge analysis")

    # Bet History Table
    st.markdown("---")
    st.markdown("### 📋 Bet History")

    # Add info about data quality
    if not df_bets.empty:
        # Count bets with corrected market_prob
        corrected_count = len(df_bets[(df_bets['market_prob'] < 0.49) | (df_bets['market_prob'] > 0.51)])
        total_count = len(df_bets)

        if corrected_count > 0:
            st.info(f"ℹ️ **{corrected_count} of {total_count} bets** have historically accurate odds data. "
                   f"The remaining {total_count - corrected_count} bets have estimated market probabilities (50%).")
        else:
            st.warning("⚠️ All bets currently show estimated market probabilities (50%). "
                      "Run `python scripts/backfill_historical_odds.py` to fetch accurate historical odds.")

    if not df_bets.empty:
        # Prepare display dataframe
        display_df = df_bets.copy()

        # Add outcome emoji
        def outcome_emoji(result):
            if result == 'win':
                return "✅"
            elif result == 'loss':
                return "❌"
            elif result == 'push':
                return "↔️"
            else:
                return "⏳"

        display_df['status'] = display_df['outcome'].apply(outcome_emoji)

        # Format date
        display_df['date_formatted'] = pd.to_datetime(display_df['commence_time'], format='ISO8601').dt.strftime('%m/%d/%y')

        # Create matchup string
        display_df['matchup'] = display_df['away_team'] + ' @ ' + display_df['home_team']

        # Format bet details
        def format_bet(row):
            bet_type = row['bet_type']
            bet_side = row['bet_side']
            line = row['line']

            if bet_type == 'spread':
                # Round to nearest 0.5 to clean up corrupted data
                if pd.notna(line):
                    line_rounded = round(line * 2) / 2  # Round to nearest 0.5
                    return f"{bet_side.upper()} {line_rounded:+.1f}"
                else:
                    return f"{bet_side.upper()}"
            elif bet_type == 'total':
                if pd.notna(line):
                    line_rounded = round(line * 2) / 2  # Round to nearest 0.5
                    return f"{bet_side.upper()} {line_rounded:.1f}"
                else:
                    return f"{bet_side.upper()}"
            else:  # moneyline
                return f"{bet_side.upper()} ML"

        display_df['bet'] = display_df.apply(format_bet, axis=1)

        # Fix odds column - convert to integer, handling corrupted data
        def safe_int_odds(x):
            if pd.isna(x):
                return None
            try:
                # Handle bytes/binary data
                if isinstance(x, bytes):
                    return None
                # Convert to float first, then int
                return int(float(x))
            except (ValueError, TypeError):
                return None

        display_df['odds_int'] = display_df['odds'].apply(safe_int_odds)

        # Use profit column (already in database)
        profit_col = 'profit'

        # Format model and market probabilities
        if 'model_prob' in display_df.columns:
            display_df['model_prob_pct'] = (display_df['model_prob'] * 100).round(1)

        if 'market_prob' in display_df.columns:
            # Mark estimated vs actual market probabilities
            display_df['market_prob_pct'] = (display_df['market_prob'] * 100).round(1)
            display_df['market_prob_estimated'] = (display_df['market_prob'] >= 0.49) & (display_df['market_prob'] <= 0.51)

            # Format with indicator for estimated values
            def format_market_prob(row):
                prob = row['market_prob_pct']
                if pd.isna(prob):
                    return ''
                if row['market_prob_estimated']:
                    return f"{prob:.1f}*"  # Asterisk for estimated
                return f"{prob:.1f}"

            display_df['market_prob_formatted'] = display_df.apply(format_market_prob, axis=1)

        if 'edge' in display_df.columns:
            # Only show edge for non-estimated market probs
            def format_edge(row):
                if pd.isna(row.get('edge')):
                    return ''
                if row.get('market_prob_estimated', False):
                    return '~'  # Tilde for estimated edge
                edge = row['edge'] * 100
                return f"{edge:+.1f}"

            display_df['edge_formatted'] = display_df.apply(format_edge, axis=1)

        # Select and format columns
        display_columns = {
            'status': '',
            'date_formatted': 'Date',
            'matchup': 'Game',
            'bet': 'Bet',
            'odds_int': 'Odds',
            'model_prob_pct': 'Model %',
            'market_prob_formatted': 'Market %',
            'edge_formatted': 'Edge %',
            'outcome': 'Result',
            profit_col: 'Profit'
        }

        # Add CLV if available
        if 'clv' in display_df.columns:
            display_columns['clv'] = 'CLV'

        # Filter selector
        col1, col2, col3 = st.columns(3)

        with col1:
            result_filter = st.multiselect(
                "Filter by Result",
                options=['win', 'loss', 'push', 'unsettled'],
                default=['win', 'loss', 'push', 'unsettled'],
                format_func=lambda x: {'win': 'Wins', 'loss': 'Losses', 'push': 'Pushes', 'unsettled': 'Unsettled'}.get(x, x)
            )

        with col2:
            if 'bet_type' in display_df.columns:
                type_filter = st.multiselect(
                    "Filter by Type",
                    options=display_df['bet_type'].unique().tolist(),
                    default=display_df['bet_type'].unique().tolist()
                )
            else:
                type_filter = None

        with col3:
            date_range = st.selectbox(
                "Date Range",
                options=['All Time', 'Last 7 Days', 'Last 30 Days', 'This Season'],
                index=0
            )

        # Apply filters
        # Handle unsettled bets separately
        if 'unsettled' in result_filter:
            settled_filter = [x for x in result_filter if x != 'unsettled']
            if settled_filter:
                filtered_df = display_df[
                    (display_df['outcome'].isin(settled_filter)) |
                    (display_df['outcome'].isna())
                ]
            else:
                filtered_df = display_df[display_df['outcome'].isna()]
        else:
            filtered_df = display_df[display_df['outcome'].isin(result_filter)]

        if type_filter:
            filtered_df = filtered_df[filtered_df['bet_type'].isin(type_filter)]

        if date_range == 'Last 7 Days':
            cutoff = datetime.now() - timedelta(days=7)
            filtered_df = filtered_df[pd.to_datetime(filtered_df['commence_time'], format='ISO8601') >= cutoff]
        elif date_range == 'Last 30 Days':
            cutoff = datetime.now() - timedelta(days=30)
            filtered_df = filtered_df[pd.to_datetime(filtered_df['commence_time'], format='ISO8601') >= cutoff]

        # Display summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Showing", len(filtered_df))
        with col2:
            wins = len(filtered_df[filtered_df['outcome'] == 'win'])
            losses = len(filtered_df[filtered_df['outcome'] == 'loss'])
            if wins + losses > 0:
                st.metric("Win Rate", f"{wins / (wins + losses) * 100:.1f}%")
        with col3:
            total_profit = filtered_df[profit_col].sum() if not filtered_df.empty else 0
            st.metric("Total Profit", f"${total_profit:.2f}")
        with col4:
            if wins + losses > 0:
                roi = (total_profit / ((wins + losses) * 100)) * 100
                st.metric("ROI", f"{roi:.1f}%")

        # Legend
        st.caption("**Legend:** Market % with * = estimated (50%), Edge % with ~ = based on estimated market prob")

        # Display table
        st.dataframe(
            filtered_df[[col for col in display_columns.keys() if col in filtered_df.columns]].rename(
                columns=display_columns
            ),
            column_config={
                "": st.column_config.TextColumn(width="small"),
                "Odds": st.column_config.NumberColumn(format="%+d"),
                "Profit": st.column_config.NumberColumn(format="$%.2f"),
                "Model %": st.column_config.NumberColumn(format="%.1f"),
                "Market %": st.column_config.TextColumn(),
                "Edge %": st.column_config.TextColumn(),
                "CLV": st.column_config.NumberColumn(format="%.2f")
            },
            hide_index=True,
            use_container_width=True,
            height=400
        )

        # Download button
        csv = filtered_df[[col for col in display_columns.keys() if col in filtered_df.columns]].to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"bets_{date_range.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No bets found")

    # Live Betting Performance
    st.markdown("---")
    st.markdown("### 🔴 Live Betting Performance")

    try:
        from src.betting.live_bet_settlement import LiveBetSettlement
        settlement = LiveBetSettlement()
        live_stats = settlement.get_settlement_stats()

        if live_stats['total_bets'] > 0:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Live Bets", live_stats['total_bets'])
            with col2:
                win_rate_pct = live_stats['win_rate'] * 100
                st.metric("Win Rate", f"{win_rate_pct:.1f}%")
            with col3:
                st.metric("Total Profit", f"${live_stats['total_profit']:.2f}")
            with col4:
                roi = (live_stats['total_profit'] / (live_stats['total_bets'] * 100)) * 100 if live_stats['total_bets'] > 0 else 0
                st.metric("ROI", f"{roi:.1f}%")

            # Win/Loss breakdown
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Outcome Breakdown**")
                outcome_data = {
                    'Wins': live_stats['wins'],
                    'Losses': live_stats['losses'],
                    'Pushes': live_stats['pushes'],
                    'Pending': live_stats['unsettled']
                }
                st.bar_chart(outcome_data)

            with col2:
                if live_stats['confidence_breakdown']:
                    st.markdown("**Performance by Confidence**")
                    conf_df = pd.DataFrame(live_stats['confidence_breakdown'])
                    conf_df['win_rate_pct'] = conf_df['win_rate'] * 100
                    st.dataframe(
                        conf_df[['confidence', 'bets', 'wins', 'win_rate_pct', 'profit']].rename(columns={
                            'confidence': 'Confidence',
                            'bets': 'Bets',
                            'wins': 'Wins',
                            'win_rate_pct': 'Win Rate %',
                            'profit': 'Profit'
                        }),
                        hide_index=True
                    )
        else:
            st.info("No live betting data yet. Run the manual live check or monitor to start collecting data.")
    except Exception as e:
        st.warning(f"Could not load live betting stats: {e}")

with tab2:
    st.markdown("### CLV (Closing Line Value) Analysis")

    col1, col2 = st.columns(2)

    with col1:
        clv_dist_chart = create_clv_distribution_chart(df_bets)
        if clv_dist_chart:
            st.plotly_chart(clv_dist_chart, use_container_width=True)
        else:
            st.info("No CLV data available")

    with col2:
        clv_scatter = create_clv_vs_profit_scatter(df_bets)
        if clv_scatter:
            st.plotly_chart(clv_scatter, use_container_width=True)
        else:
            st.info("Not enough CLV data for scatter plot")

    # CLV statistics
    if not df_bets.empty and 'clv' in df_bets.columns:
        df_clv = df_bets[df_bets['clv'].notna()]

        if len(df_clv) > 0:
            st.markdown("### CLV Statistics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                positive_clv_pct = (df_clv['clv'] > 0).sum() / len(df_clv) * 100
                st.metric("Positive CLV Rate", f"{positive_clv_pct:.1f}%")

            with col2:
                st.metric("Median CLV", f"{df_clv['clv'].median():.2f}%")

            with col3:
                st.metric("Best CLV", f"{df_clv['clv'].max():.2f}%")

            with col4:
                st.metric("Worst CLV", f"{df_clv['clv'].min():.2f}%")

with tab3:
    st.markdown("### Model Calibration & Accuracy")

    if not df_bets.empty and 'model_prob' in df_bets.columns:
        df_settled = df_bets[df_bets['outcome'].notna()].copy()

        if len(df_settled) >= 20:
            # Calibration plot
            df_settled['prob_bin'] = pd.cut(df_settled['model_prob'], bins=10)
            calibration = df_settled.groupby('prob_bin', observed=True).agg({
                'outcome': lambda x: (x == 'win').sum() / len(x),
                'model_prob': 'mean',
                'bet_amount': 'count'
            }).reset_index()

            fig = go.Figure()

            # Perfect calibration line
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Perfect Calibration',
                line=dict(dash='dash', color='gray')
            ))

            # Actual calibration
            fig.add_trace(go.Scatter(
                x=calibration['model_prob'],
                y=calibration['outcome'],
                mode='markers+lines',
                name='Model Calibration',
                marker=dict(size=10, color='#10b981'),
                line=dict(width=2, color='#10b981')
            ))

            fig.update_layout(
                title="Model Calibration Plot",
                xaxis_title="Predicted Probability",
                yaxis_title="Actual Win Rate",
                template="plotly_dark",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough settled bets for calibration analysis")
    else:
        st.info("Model probability data not available")

with tab4:
    st.markdown("### Time-based Performance Analysis")

    heatmap = create_daily_performance_heatmap(df_bets)
    if heatmap:
        st.plotly_chart(heatmap, use_container_width=True)
    else:
        st.info("Not enough data for time-based analysis")

    # Recent bets table
    if not df_bets.empty:
        st.markdown("### Recent Bets")

        recent_bets = df_bets.sort_values('logged_at', ascending=False).head(20)

        # Format for display
        display_cols = ['logged_at', 'bet_type', 'bet_side', 'line', 'odds',
                       'edge', 'kelly', 'bet_amount', 'outcome', 'profit', 'clv']

        available_cols = [col for col in display_cols if col in recent_bets.columns]

        # Clean any encoding issues in string columns
        display_df = recent_bets[available_cols].copy()
        for col in display_df.select_dtypes(include=['object']).columns:
            try:
                # Convert to string and handle any encoding issues
                display_df[col] = display_df[col].apply(
                    lambda x: str(x).encode('utf-8', errors='replace').decode('utf-8') if pd.notna(x) else x
                )
            except:
                pass  # Skip columns that can't be converted

        # Format percentage columns
        if 'edge' in display_df.columns:
            display_df['edge'] = display_df['edge'].apply(
                lambda x: f"{x*100:.1f}%" if pd.notna(x) else None
            )
        if 'kelly' in display_df.columns:
            display_df['kelly'] = display_df['kelly'].apply(
                lambda x: f"{x*100:.1f}%" if pd.notna(x) else None
            )
        if 'clv' in display_df.columns:
            display_df['clv'] = display_df['clv'].apply(
                lambda x: f"{x*100:.1f}%" if pd.notna(x) else None
            )

        # Rename columns for better display
        display_df = display_df.rename(columns={
            'logged_at': 'Date',
            'bet_type': 'Type',
            'bet_side': 'Side',
            'line': 'Line',
            'odds': 'Odds',
            'edge': 'Edge %',
            'kelly': 'Kelly %',
            'bet_amount': 'Bet $',
            'outcome': 'Result',
            'profit': 'Profit $',
            'clv': 'CLV %'
        })

        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )

with tab5:
    st.markdown("### Today's Betting Picks")

    try:
        # Import prediction functions
        from src.prediction_cache import load_cached_predictions, get_cache_info, refresh_predictions
        from src.bet_tracker import log_manual_bet, get_regime_status

        # Load predictions from cache
        predictions = load_cached_predictions()
        cache_info = get_cache_info()

        # Header with cache info and refresh
        col1, col2 = st.columns([4, 1])
        with col1:
            if cache_info:
                st.caption(f"📅 Updated: {cache_info.get('timestamp', 'Unknown')[:16]} | {cache_info.get('num_games', 0)} games")
            else:
                st.caption("No cached predictions available")

        with col2:
            if st.button("🔄 Refresh", key="refresh_predictions", use_container_width=True):
                with st.spinner("Refreshing predictions..."):
                    predictions = refresh_predictions(min_edge=0.02)
                    st.success("Predictions refreshed!")
                    st.rerun()

        # Check if predictions dict is empty or has no valid DataFrames
        has_predictions = bool(predictions and any(
            df is not None and not df.empty
            for df in predictions.values()
        ))

        if not has_predictions:
            st.info("No predictions available. Click Refresh to load today's picks.")
        else:
            # Get regime status
            regime_status = get_regime_status()
            regime = regime_status.get('regime', 'normal') if regime_status else 'normal'

            # Display regime status
            if regime == 'edge_decay':
                st.warning(f"⚠️ EDGE DECAY: {regime_status.get('pause_reason', 'Performance below threshold')} - Bet sizes reduced 50%")
            elif regime == 'volatile':
                st.warning("⚠️ Volatile market conditions - Bet sizes reduced 50%")
            elif regime == 'hot_streak':
                st.info("ℹ️ Hot streak detected - Proceed with caution")
            else:
                st.success(f"✅ Regime: {regime.upper().replace('_', ' ')}")

            st.divider()

            # Alternative data status
            alternative_data_status_summary()

            st.divider()

            # Display predictions by market
            # Spread predictions from Pipeline (with CLV filter + edge strategy)
            # Check for 'pipeline' first (new), fall back to 'ats' (old cache) for compatibility
            picks_key = 'pipeline' if 'pipeline' in predictions else 'ats'

            if picks_key in predictions and predictions[picks_key] is not None and not predictions[picks_key].empty:
                st.markdown("#### 📈 Spread Picks (Daily Pipeline - CLV Filtered)")
                ats_df = predictions[picks_key]

                # Show only games with actual bets (not PASS)
                # Use bet_side column if available, otherwise fall back to bet_home/bet_away flags
                if 'bet_side' in ats_df.columns:
                    ats_bets = ats_df[ats_df['bet_side'].isin(['HOME', 'AWAY'])]
                else:
                    ats_bets = ats_df[
                        (ats_df.get('bet_home', False) == True) |
                        (ats_df.get('bet_away', False) == True)
                    ]

                if not ats_bets.empty:
                    for _, row in ats_bets.iterrows():
                        # Use enhanced pick display
                        bet_result = enhanced_pick_display(row, bankroll=bankroll, kelly_fraction=kelly_fraction)

                        # If user clicked Log Bet button, log it
                        if bet_result:
                            try:
                                log_manual_bet(
                                    game_id=bet_result['game_id'],
                                    home_team=bet_result['home_team'],
                                    away_team=bet_result['away_team'],
                                    commence_time=row['commence_time'],
                                    bet_type='spread',
                                    bet_side=bet_result['bet_side'],
                                    odds=-110,
                                    bet_amount=bet_result['bet_amount'],
                                    line=bet_result['line']
                                )
                                st.success("✅ Bet logged!")
                            except Exception as e:
                                st.error(f"Error logging bet: {e}")
                else:
                    st.info("No spread picks with edge today")

            # Moneyline predictions (Stacking model)
            if 'stacking' in predictions and predictions['stacking'] is not None and not predictions['stacking'].empty:
                st.markdown("#### 💰 Moneyline Picks (Stacking Ensemble)")
                stack_df = predictions['stacking']

                # Show only games with actual bets (not PASS)
                if 'bet_side' in stack_df.columns:
                    ml_bets = stack_df[stack_df['bet_side'].isin(['HOME', 'AWAY'])]
                else:
                    ml_bets = stack_df[
                        (stack_df.get('bet_home', False) == True) |
                        (stack_df.get('bet_away', False) == True)
                    ]

                if not ml_bets.empty:
                    for _, row in ml_bets.iterrows():
                        # Determine bet side from bet_side column or bet_home/bet_away flags
                        if 'bet_side' in row and pd.notna(row['bet_side']) and row['bet_side'] != 'PASS':
                            is_betting_home = row['bet_side'] == 'HOME'
                        else:
                            is_betting_home = row.get('bet_home', False)

                        bet_team = row['home_team'] if is_betting_home else row['away_team']
                        bet_side = 'home' if is_betting_home else 'away'
                        odds = row.get('best_home_odds', -110) if is_betting_home else row.get('best_away_odds', -110)
                        prob = row.get('stack_home_prob_adj', 0.5) if is_betting_home else row.get('stack_away_prob_adj', 0.5)
                        kelly = row.get('kelly', row.get('home_kelly', 0)) if is_betting_home else row.get('away_kelly', 0)
                        confidence = row.get('confidence', 'Medium')

                        with st.expander(f"**{row['away_team']} @ {row['home_team']}**", expanded=True):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Pick", f"{bet_team} ({odds:+d})")
                            with col2:
                                st.metric("Win Probability", f"{prob:.1%}")
                            with col3:
                                bet_amt = 1000 * kelly * 0.25
                                st.metric("Suggested Bet", f"${bet_amt:.0f}")

                            st.caption(f"Confidence: {confidence}")

                            # Quick log button
                            if st.button(f"📝 Log This Bet", key=f"log_ml_{row['game_id']}", use_container_width=True):
                                try:
                                    log_manual_bet(
                                        game_id=row['game_id'],
                                        home_team=row['home_team'],
                                        away_team=row['away_team'],
                                        commence_time=row['commence_time'],
                                        bet_type='moneyline',
                                        bet_side=bet_side,
                                        odds=odds,
                                        bet_amount=bet_amt,
                                        model_prob=prob
                                    )
                                    st.success("✅ Bet logged!")
                                except Exception as e:
                                    st.error(f"Error logging bet: {e}")
                else:
                    st.info("No moneyline picks with edge today")

            # Show message if no predictions with bets
            has_ats_bets = False
            # Check pipeline first, fall back to ats for old cache
            picks_check_key = 'pipeline' if 'pipeline' in predictions else 'ats'
            if picks_check_key in predictions and predictions[picks_check_key] is not None and not predictions[picks_check_key].empty:
                ats_df = predictions[picks_check_key]
                if 'bet_side' in ats_df.columns:
                    has_ats_bets = any(ats_df['bet_side'].isin(['HOME', 'AWAY']))
                else:
                    has_ats_bets = any(ats_df.get('bet_home', False) | ats_df.get('bet_away', False))

            has_ml_bets = False
            if 'stacking' in predictions and predictions['stacking'] is not None and not predictions['stacking'].empty:
                stack_df = predictions['stacking']
                if 'bet_side' in stack_df.columns:
                    has_ml_bets = any(stack_df['bet_side'].isin(['HOME', 'AWAY']))
                else:
                    has_ml_bets = any(stack_df.get('bet_home', False) | stack_df.get('bet_away', False))

            if not (has_ats_bets or has_ml_bets):
                st.info("No predictions with positive edge today. Click Refresh to reload.")

    except ImportError as e:
        st.error(f"Prediction modules not found: {e}")
        st.info("Make sure src/prediction_cache.py exists and is properly configured.")
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        import traceback
        with st.expander("Show error details"):
            st.code(traceback.format_exc())

with tab_live:
    st.header("🔴 Live Betting Opportunities")

    # Import live betting modules
    from src.data.live_betting_db import DB_PATH as LIVE_DB_PATH, get_stats as get_live_stats
    import sqlite3

    # Check if monitor is running
    st.info("💡 Start the live monitor with: `python -m scripts.live_game_monitor`")

    # Get live games
    try:
        conn = sqlite3.connect(LIVE_DB_PATH)

        # Get most recent game states (last 5 minutes)
        cutoff = (datetime.now() - timedelta(minutes=5)).isoformat()
        live_games_query = """
            SELECT
                game_id,
                home_team,
                away_team,
                home_score,
                away_score,
                quarter,
                time_remaining,
                game_status,
                MAX(timestamp) as last_update
            FROM live_game_state
            WHERE timestamp >= ?
            GROUP BY game_id
            ORDER BY last_update DESC
        """

        live_games_df = pd.read_sql_query(live_games_query, conn, params=[cutoff])

        if live_games_df.empty:
            st.warning("⏰ No live games currently. Games haven't started or monitor isn't running.")
        else:
            st.success(f"📡 Monitoring {len(live_games_df)} live games")

            # Display each live game
            for _, game in live_games_df.iterrows():
                with st.expander(f"**{game['away_team']} @ {game['home_team']}** - Q{game['quarter']} {game['time_remaining']}", expanded=True):
                    col1, col2, col3 = st.columns([2, 2, 3])

                    with col1:
                        st.metric("Score", f"{game['away_score']} - {game['home_score']}")
                        st.caption(f"Status: {game['game_status']}")

                    with col2:
                        st.metric("Quarter", f"Q{game['quarter']}")
                        st.caption(f"Time: {game['time_remaining']}")

                    with col3:
                        # Get latest alerts for this game
                        alerts_query = """
                            SELECT *
                            FROM live_edge_alerts
                            WHERE game_id = ?
                            AND dismissed = 0
                            ORDER BY timestamp DESC
                            LIMIT 3
                        """
                        alerts = pd.read_sql_query(alerts_query, conn, params=[game['game_id']])

                        if not alerts.empty:
                            st.markdown("**🎯 Active Alerts:**")
                            for _, alert in alerts.iterrows():
                                confidence_color = {
                                    'HIGH': '🟢',
                                    'MEDIUM': '🟡',
                                    'LOW': '🟠'
                                }.get(alert['confidence'], '⚪')

                                st.markdown(
                                    f"{confidence_color} **{alert['alert_type'].upper()} {alert['bet_side'].upper()}** "
                                    f"(Edge: {alert['edge']*100:.1f}%)"
                                )
                                st.caption(
                                    f"Model: {alert['model_prob']*100:.1f}% | "
                                    f"Market: {alert['market_prob']*100:.1f}% | "
                                    f"Odds: {alert['odds']:+d}"
                                )
                        else:
                            st.caption("No active alerts")

        # Recent alerts section
        st.markdown("---")
        st.subheader("📋 Recent Alerts (Last 24 Hours)")

        recent_cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
        recent_alerts_query = """
            SELECT
                timestamp,
                home_team || ' vs ' || away_team as game,
                alert_type,
                bet_side,
                ROUND(edge * 100, 1) as edge_pct,
                confidence,
                quarter,
                home_score || '-' || away_score as score,
                acted_on
            FROM live_edge_alerts
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
            LIMIT 50
        """

        recent_alerts = pd.read_sql_query(recent_alerts_query, conn, params=[recent_cutoff])

        if not recent_alerts.empty:
            # Format timestamp
            recent_alerts['timestamp'] = pd.to_datetime(recent_alerts['timestamp']).dt.strftime('%I:%M %p')

            # Display as table
            st.dataframe(
                recent_alerts,
                column_config={
                    "timestamp": "Time",
                    "game": "Game",
                    "alert_type": "Type",
                    "bet_side": "Side",
                    "edge_pct": st.column_config.NumberColumn("Edge %", format="%.1f%%"),
                    "confidence": "Confidence",
                    "quarter": "Q",
                    "score": "Score",
                    "acted_on": st.column_config.CheckboxColumn("Acted")
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No alerts in the last 24 hours")

        # Paper Bet Performance section
        st.markdown("---")
        st.subheader("💰 Paper Bet Performance")

        paper_bets_query = """
            SELECT
                pb.id,
                pb.timestamp,
                pb.home_team || ' vs ' || pb.away_team as game,
                pb.bet_type,
                pb.bet_side,
                pb.line_value,
                pb.odds,
                pb.stake,
                pb.confidence,
                pb.outcome,
                pb.profit,
                pb.quarter || ' ' || pb.time_remaining as game_state
            FROM live_paper_bets pb
            ORDER BY pb.timestamp DESC
            LIMIT 100
        """

        paper_bets = pd.read_sql_query(paper_bets_query, conn)

        if not paper_bets.empty:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)

            settled = paper_bets[paper_bets['outcome'].notna()]
            wins = len(settled[settled['outcome'] == 'win'])
            losses = len(settled[settled['outcome'] == 'loss'])
            total_profit = settled['profit'].sum() if not settled.empty else 0

            with col1:
                st.metric("Total Bets", len(paper_bets))
            with col2:
                win_rate = (wins / len(settled) * 100) if len(settled) > 0 else 0
                st.metric("Win Rate", f"{win_rate:.1f}%")
            with col3:
                st.metric("Profit/Loss", f"${total_profit:.2f}", delta=f"${total_profit:.2f}")
            with col4:
                pending = len(paper_bets[paper_bets['outcome'].isna()])
                st.metric("Pending", pending)

            # Format for display
            display_bets = paper_bets.copy()
            display_bets['timestamp'] = pd.to_datetime(display_bets['timestamp']).dt.strftime('%m/%d %I:%M %p')

            # Add outcome emoji
            def outcome_emoji(outcome):
                if pd.isna(outcome):
                    return "⏳"
                elif outcome == 'win':
                    return "✅"
                elif outcome == 'loss':
                    return "❌"
                else:  # push
                    return "↔️"

            display_bets['status'] = display_bets['outcome'].apply(outcome_emoji)

            # Format line value
            display_bets['line'] = display_bets.apply(
                lambda row: f"{row['line_value']:+.1f}" if pd.notna(row['line_value']) else "ML",
                axis=1
            )

            st.dataframe(
                display_bets[[
                    'status', 'timestamp', 'game', 'bet_type', 'bet_side',
                    'line', 'odds', 'confidence', 'game_state', 'profit'
                ]],
                column_config={
                    "status": "",
                    "timestamp": "Time",
                    "game": "Game",
                    "bet_type": "Type",
                    "bet_side": "Side",
                    "line": "Line",
                    "odds": st.column_config.NumberColumn("Odds", format="%+d"),
                    "confidence": "Conf",
                    "game_state": "When",
                    "profit": st.column_config.NumberColumn("Profit", format="$%.2f")
                },
                hide_index=True,
                use_container_width=True
            )

            # Settlement button
            if pending > 0:
                if st.button("🔄 Auto-Settle Completed Games"):
                    from src.betting.live_bet_settlement import LiveBetSettlement
                    settler = LiveBetSettlement()
                    settled_count = settler.auto_settle_completed_games()
                    if settled_count > 0:
                        st.success(f"Settled {settled_count} completed game(s)! Refresh to see results.")
                        st.rerun()
                    else:
                        st.info("No completed games found to settle")
        else:
            st.info("No paper bets yet. Start monitoring live games to generate paper bets from edge alerts!")

        # Database statistics
        st.markdown("---")
        st.subheader("📊 Collection Statistics")

        stats = get_live_stats()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Game States", f"{stats.get('live_game_state', 0):,}")

        with col2:
            st.metric("Odds Snapshots", f"{stats.get('live_odds_snapshot', 0):,}")

        with col3:
            st.metric("Total Alerts", f"{stats.get('live_edge_alerts', 0):,}")

        with col4:
            st.metric("Paper Bets", f"{stats.get('live_paper_bets', 0):,}")

        if stats.get('date_range'):
            st.caption(f"Data range: {stats['date_range']['start']} to {stats['date_range']['end']}")

        # Performance metrics (if we have paper bets)
        if stats.get('settled_bets', 0) > 0:
            st.markdown("---")
            st.subheader("📈 Paper Trading Performance")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Win Rate", f"{stats.get('win_rate', 0):.1%}")

            with col2:
                st.metric("Total Profit", f"${stats.get('total_profit', 0):.2f}")

            with col3:
                st.metric("ROI", f"{stats.get('roi', 0):.1%}")

        conn.close()

    except Exception as e:
        st.error(f"Error loading live betting data: {e}")
        st.info("Make sure the live betting database exists. Run the monitor first.")

# Footer
st.markdown("---")
# Safe: Use separate st.markdown calls - no dynamic content with unsafe_allow_html
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>NBA Betting Analytics Dashboard | Data updates every 60 seconds</p>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    f"🔄 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    unsafe_allow_html=False  # Explicit: no HTML in dynamic content
)
