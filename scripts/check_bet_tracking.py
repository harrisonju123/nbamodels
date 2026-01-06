import sqlite3
import pandas as pd
from datetime import datetime

# Check live_betting.db
db_path = 'data/bets/live_betting.db'
conn = sqlite3.connect(db_path)

# Get table names
tables = pd.read_sql('SELECT name FROM sqlite_master WHERE type="table"', conn)
print(f'Tables in {db_path}:')
print(tables)
print()

# Query bets table if it exists
try:
    bets = pd.read_sql('SELECT * FROM bets ORDER BY game_date DESC LIMIT 10', conn)
    print(f'Recent bets (showing 10 most recent):')
    print(bets[['game_date', 'home_team', 'away_team', 'bet_side', 'result', 'profit']].to_string())
    print()

    # Check all bets
    all_bets = pd.read_sql('SELECT * FROM bets', conn)
    all_bets['game_date'] = pd.to_datetime(all_bets['game_date'])

    print(f'Total bets: {len(all_bets)}')
    print(f'Date range: {all_bets["game_date"].min()} to {all_bets["game_date"].max()}')
    print()

    # Check if game dates are in the future (prospective) or past (retrospective)
    today = datetime.now()
    future_bets = all_bets[all_bets['game_date'] > today]
    past_bets = all_bets[all_bets['game_date'] <= today]

    print(f'Future games: {len(future_bets)}')
    print(f'Past games: {len(past_bets)}')

    if len(past_bets) > 0:
        settled = past_bets[past_bets['result'].notna()]
        print(f'Settled bets: {len(settled)}')

        if len(settled) > 0:
            total_profit = settled['profit'].sum()
            total_wagered = settled['amount'].sum()
            roi = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0
            wins = (settled['result'] == 'win').sum()
            win_rate = (wins / len(settled)) * 100

            print(f'\nPerformance on settled bets:')
            print(f'  ROI: {roi:.2f}%')
            print(f'  Win Rate: {win_rate:.1f}% ({wins}/{len(settled)})')
            print(f'  Total Profit: ${total_profit:.2f}')
            print(f'  Total Wagered: ${total_wagered:.2f}')

except Exception as e:
    print(f'Error querying bets: {e}')

conn.close()
