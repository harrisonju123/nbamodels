"""Validate betting strategy against multiple historical seasons."""

import sys
sys.path.insert(0, '.')

from src.betting.dual_model_backtest import run_dual_model_ats_backtest

# Test the best strategies against multiple seasons
seasons_to_test = [
    [2020],
    [2021],
    [2022],
    [2023],
    [2024],
]

strategies = [
    {"name": "Original (edge filter)", "min_disagreement": 3.0, "min_edge_vs_market": 0.0, "home_only": False},
    {"name": "Stricter both sides", "min_disagreement": 5.0, "min_edge_vs_market": 3.0, "home_only": False},
    {"name": "Moderate HOME", "min_disagreement": 4.0, "min_edge_vs_market": 2.0, "home_only": True},
]

print("=" * 100)
print("MULTI-SEASON VALIDATION - Dual Model ATS Strategy")
print("=" * 100)

# Store results by strategy
all_results = {s["name"]: [] for s in strategies}

for season in seasons_to_test:
    print(f"\n{'='*50}")
    print(f"SEASON {season[0]}")
    print(f"{'='*50}")

    for s in strategies:
        result, _ = run_dual_model_ats_backtest(
            test_seasons=season,
            kelly_fraction=0.15,
            min_disagreement=s["min_disagreement"],
            min_edge_vs_market=s["min_edge_vs_market"],
            home_only=s["home_only"],
            output_path=None,
        )

        all_results[s["name"]].append({
            "season": season[0],
            "bets": result.num_bets,
            "ats_rate": result.ats_win_rate,
            "roi": result.roi,
            "pnl": result.total_pnl,
            "home_pct": result.home_ats_pct,
            "away_pct": result.away_ats_pct,
        })

        print(f"  {s['name']:<25}: {result.num_bets:>3} bets, {result.ats_win_rate:>5.1%} ATS, {result.roi:>+6.1%} ROI")

# Summary by strategy
print("\n" + "=" * 100)
print("AGGREGATE RESULTS BY STRATEGY")
print("=" * 100)

for name, results in all_results.items():
    total_bets = sum(r["bets"] for r in results)
    total_wins = sum(r["bets"] * r["ats_rate"] for r in results)
    total_pnl = sum(r["pnl"] for r in results)

    if total_bets > 0:
        overall_ats = total_wins / total_bets
        # Approximate ROI (assuming similar wagered amounts)
        avg_roi = sum(r["roi"] * r["bets"] for r in results) / total_bets if total_bets > 0 else 0
    else:
        overall_ats = 0
        avg_roi = 0

    print(f"\n{name}")
    print("-" * 50)
    print(f"  Total Bets: {total_bets}")
    print(f"  Overall ATS: {overall_ats:.1%}")
    print(f"  Weighted Avg ROI: {avg_roi:+.1%}")
    print(f"  Total P&L: ${total_pnl:+,.0f}")
    print(f"  Seasons profitable: {sum(1 for r in results if r['roi'] > 0)}/{len(results)}")

    # Per-season breakdown
    print(f"  By Season:")
    for r in results:
        status = "✓" if r["roi"] > 0 else "✗"
        print(f"    {r['season']}: {r['bets']:>3} bets, {r['ats_rate']:>5.1%} ATS, {r['roi']:>+6.1%} ROI {status}")

# Final recommendation
print("\n" + "=" * 100)
print("RECOMMENDATION")
print("=" * 100)

# Find most consistent strategy (profitable in most seasons)
best_consistency = 0
best_strategy = None
for name, results in all_results.items():
    profitable_seasons = sum(1 for r in results if r['roi'] > 0)
    total_bets = sum(r["bets"] for r in results)
    if profitable_seasons > best_consistency and total_bets >= 50:
        best_consistency = profitable_seasons
        best_strategy = name

if best_strategy:
    results = all_results[best_strategy]
    total_bets = sum(r["bets"] for r in results)
    total_wins = sum(r["bets"] * r["ats_rate"] for r in results)
    overall_ats = total_wins / total_bets if total_bets > 0 else 0

    print(f"Best Strategy: {best_strategy}")
    print(f"  Profitable in {best_consistency}/{len(seasons_to_test)} seasons")
    print(f"  Total bets across all seasons: {total_bets}")
    print(f"  Overall ATS rate: {overall_ats:.1%}")
else:
    print("No strategy had enough bets for reliable validation.")
