import re
import sys
from collections import defaultdict, Counter

def parse_log(file_path):
    data = {
        "steps": [],
        "pairs": [],
        "quarantines": [],
        "overall_metrics": {
            "total_pnl": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "gross_profit": 0.0,
            "gross_loss": 0.0
        },
        "exit_reasons": Counter()
    }

    # Regex patterns
    # PAIR_SUMMARY step=1 pair=ATOMUSDT-MASKUSDT trades=1 ... gross=-0.14 costs=0.90 net=-1.04 ... Exits=[ZStop=1]
    pair_pattern = re.compile(r"PAIR_SUMMARY step=(\d+) pair=([\w-]+) trades=(\d+) .*? gross=([-\d.]+) costs=([-\d.]+) net=([-\d.]+) .*? Exits=\[(.*?)\]")
    
    # WF_STEP step=1 ... net=-9.18 sharpe=-9.3955 ...
    step_pattern = re.compile(r"WF_STEP step=(\d+) .*? net=([-\d.]+) sharpe=([-\d.]+)")
    
    # QUARANTINE_ENTER Pair: ATOMUSDT-MASKUSDT, Reason: StepRiskBreach ...
    quarantine_pattern = re.compile(r"QUARANTINE_ENTER Pair: ([\w-]+), Reason: (.*)")

    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Parse Pair Summary
                pm = pair_pattern.search(line)
                if pm:
                    step, pair, trades, gross, costs, net, exits = pm.groups()
                    trades = int(trades)
                    gross = float(gross)
                    costs = float(costs)
                    net = float(net)
                    
                    data["pairs"].append({
                        "step": int(step),
                        "pair": pair,
                        "trades": trades,
                        "gross": gross,
                        "costs": costs,
                        "net": net,
                        "exits": exits
                    })
                    
                    # Aggregate metrics
                    data["overall_metrics"]["total_pnl"] += net
                    data["overall_metrics"]["total_trades"] += trades
                    if net > 0:
                        data["overall_metrics"]["winning_trades"] += trades # Approximation if 1 trade/summary
                        data["overall_metrics"]["gross_profit"] += net # Using net for profit factor calc simplified
                    else:
                        data["overall_metrics"]["gross_loss"] += abs(net)

                    # Parse exits
                    # Exits=[ZStop=1, Signal=2]
                    exit_parts = exits.split(', ')
                    for part in exit_parts:
                        if '=' in part:
                            reason, count = part.split('=')
                            data["exit_reasons"][reason] += int(count)
                        elif part:
                            data["exit_reasons"][part] += 1

                # Parse Step Summary
                sm = step_pattern.search(line)
                if sm:
                    step, net, sharpe = sm.groups()
                    data["steps"].append({
                        "step": int(step),
                        "net": float(net),
                        "sharpe": float(sharpe)
                    })

                # Parse Quarantine
                qm = quarantine_pattern.search(line)
                if qm:
                    pair, reason = qm.groups()
                    data["quarantines"].append({
                        "pair": pair,
                        "reason": reason
                    })

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

    return data

def analyze_data(data):
    if not data:
        return

    # 1. Profitability by Pair
    pair_pnl = defaultdict(float)
    pair_trades = defaultdict(int)
    for p in data["pairs"]:
        pair_pnl[p["pair"]] += p["net"]
        pair_trades[p["pair"]] += p["trades"]

    sorted_pairs = sorted(pair_pnl.items(), key=lambda x: x[1], reverse=True)
    top_5 = sorted_pairs[:5]
    bottom_5 = sorted_pairs[-5:]

    # 2. Exit Analysis
    total_exits = sum(data["exit_reasons"].values())
    
    # 3. Cost Impact
    total_gross = sum(p["gross"] for p in data["pairs"])
    total_costs = sum(p["costs"] for p in data["pairs"])
    total_net = sum(p["net"] for p in data["pairs"])

    # Output Report
    print("=== ANALYSIS REPORT ===")
    print(f"Total PnL: {total_net:.2f}")
    print(f"Total Trades: {data['overall_metrics']['total_trades']}")
    print(f"Gross PnL: {total_gross:.2f}")
    print(f"Total Costs: {total_costs:.2f} ({(total_costs/abs(total_gross) if total_gross != 0 else 0)*100:.1f}% of Gross Impact)")
    
    print("\n--- TOP 5 PAIRS ---")
    for pair, pnl in top_5:
        print(f"{pair}: {pnl:.2f} ({pair_trades[pair]} trades)")

    print("\n--- BOTTOM 5 PAIRS ---")
    for pair, pnl in bottom_5:
        print(f"{pair}: {pnl:.2f} ({pair_trades[pair]} trades)")

    print("\n--- EXIT REASONS ---")
    for reason, count in data["exit_reasons"].most_common():
        pct = (count / total_exits * 100) if total_exits > 0 else 0
        print(f"{reason}: {count} ({pct:.1f}%)")

    print("\n--- QUARANTINE ANALYSIS ---")
    print(f"Total Quarantines: {len(data['quarantines'])}")
    reasons = Counter([q['reason'].split(' ')[0] for q in data['quarantines']]) # Group by first word
    for r, c in reasons.items():
        print(f"{r}: {c}")

    print("\n--- STEP STABILITY ---")
    steps = sorted(data["steps"], key=lambda x: x["step"])
    for s in steps:
        print(f"Step {s['step']}: Net={s['net']:.2f}, Sharpe={s['sharpe']:.2f}")

if __name__ == "__main__":
    log_path = "/Users/admin/Downloads/task_4dc447235df84cabb2ec725071fe2f5d_backtest.log"
    data = parse_log(log_path)
    analyze_data(data)
