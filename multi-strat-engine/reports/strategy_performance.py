
import csv
from collections import defaultdict
from pathlib import Path

EVENTS = Path('/opt/multi-strat-engine/trade_events.csv')
OUT = Path('/opt/multi-strat-engine/reports/strategy_performance.csv')

# This uses only events logged by the bot (entries + max-age closes).
# TP/SL closures may not be captured if Binance doesn't expose conditionals.

def main():
    if not EVENTS.exists():
        print('trade_events.csv not found')
        return
    stats = defaultdict(lambda: {"entries":0, "wins":0, "losses":0})
    with EVENTS.open() as f:
        r=csv.DictReader(f)
        for row in r:
            sid = row.get('strategy_id') or 'unknown'
            if row.get('event') == 'ENTRY':
                stats[sid]["entries"] += 1
            if row.get('event') == 'CLOSE_MAX_AGE':
                pnl = float(row.get('pnl',0))
                if pnl > 0: stats[sid]["wins"] += 1
                elif pnl < 0: stats[sid]["losses"] += 1
    with OUT.open('w', newline='') as f:
        w=csv.writer(f)
        w.writerow(["strategy_id","entries","wins","losses","win_rate"])
        for sid, v in stats.items():
            total = v['wins'] + v['losses']
            win_rate = (v['wins']/total) if total else 0
            w.writerow([sid, v['entries'], v['wins'], v['losses'], round(win_rate,4)])
    print(OUT)

if __name__ == '__main__':
    main()
