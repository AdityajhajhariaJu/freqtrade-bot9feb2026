
import json, time, csv
import ccxt
from pathlib import Path
from datetime import datetime, timezone

CONFIG_PATH = "/opt/freqtrade/user_data/config.binance_futures_live.json"
OUT_CSV = "/opt/multi-strat-engine/reports/heatmap_hourly.csv"

def main(days=7):
    cfg = json.loads(Path(CONFIG_PATH).read_text())
    ex = ccxt.binance({
        "apiKey": cfg["exchange"]["key"],
        "secret": cfg["exchange"]["secret"],
        "enableRateLimit": True,
        "options": {"defaultType":"future"},
    })
    end_ms = int(time.time()*1000)
    start_ms = end_ms - days*24*3600*1000
    data = ex.fapiPrivateGetIncome({"incomeType": "REALIZED_PNL", "startTime": start_ms, "endTime": end_ms, "limit": 1000})
    buckets = {h:0.0 for h in range(24)}
    for row in data:
        ts = int(row.get('time',0))
        dt = datetime.fromtimestamp(ts/1000, tz=timezone.utc)
        buckets[dt.hour] += float(row.get('income',0))
    with open(OUT_CSV, 'w', newline='') as f:
        w=csv.writer(f)
        w.writerow(["hour_utc","realized_pnl"])
        for h in range(24):
            w.writerow([h, buckets[h]])
    print(OUT_CSV)

if __name__ == '__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=7)
    args=ap.parse_args()
    main(args.days)
