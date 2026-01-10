"""Compare different betting strategies for dual model ATS."""

import sys
sys.path.insert(0, '.')

from src.betting.dual_model_backtest import run_dual_model_ats_backtest

strategies = [
    {"name": "Original (loose)", "min_disagreement": 3.0, "min_edge_vs_market": 0.0, "home_only": False},
    {"name": "Stricter thresholds", "min_disagreement": 5.0, "min_edge_vs_market": 3.0, "home_only": False},
    {"name": "HOME only + stricter", "min_disagreement": 5.0, "min_edge_vs_market": 3.0, "home_only": True},
    {"name": "Very strict HOME", "min_disagreement": 5.0, "min_edge_vs_market": 5.0, "home_only": True},
    {"name": "Moderate HOME", "min_disagreement": 4.0, "min_edge_vs_market": 2.0, "home_only": True},
]

print("=" * 80)
print("STRATEGY COMPARISON - Dual Model ATS with Kelly Criterion")
print("=" * 80)

results = []
for s in strategies:
    print(f"\nTesting: {s['name']}...")
    result, _ = run_dual_model_ats_backtest(
        test_seasons=[2024],
        kelly_fraction=0.15,
        min_disagreement=s["min_disagreement"],
        min_edge_vs_market=s["min_edge_vs_market"],
        home_only=s["home_only"],
        output_path=None,  # Don't save individual results
    )
    results.append({
        "name": s["name"],
        "bets": result.num_bets,
        "ats_rate": result.ats_win_rate,
        "roi": result.roi,
        "final": result.final_bankroll,
        "max_dd": result.max_drawdown,
        "sharpe": result.sharpe_ratio,
        "home_pct": result.home_ats_pct,
        "away_pct": result.away_ats_pct,
    })

print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
header = f"{'Strategy':<25} {'Bets':>6} {'ATS%':>7} {'ROI':>8} {'Final$':>10} {'MaxDD':>7} {'Sharpe':>7}"
print(header)
print("-" * 80)

for r in results:
    line = f"{r['name']:<25} {r['bets']:>6} {r['ats_rate']:>6.1%} {r['roi']:>+7.1%} {r['final']:>10,.0f} {r['max_dd']:>6.1%} {r['sharpe']:>7.2f}"
    print(line)

print("\n" + "-" * 80)
print("HOME vs AWAY breakdown:")
print("-" * 80)
for r in results:
    print(f"{r['name']:<25} Home: {r['home_pct']:>5.1%}   Away: {r['away_pct']:>5.1%}")

# Find best strategy
best = max(results, key=lambda x: x['roi'])
print("\n" + "=" * 80)
print(f"BEST STRATEGY: {best['name']}")
print(f"  ROI: {best['roi']:+.1%}")
print(f"  ATS Win Rate: {best['ats_rate']:.1%}")
print(f"  Bets: {best['bets']}")
print("=" * 80)
