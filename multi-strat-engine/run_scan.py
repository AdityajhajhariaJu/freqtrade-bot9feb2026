#!/usr/bin/env python3
"""
Signal-only runner for the multi-strategy engine.
- Fetches 1m futures candles for configured pairs (Binance USDT-M)
- Runs all strategies and correlation filters
- Prints surviving signals and filtered reasons

Requirements: ccxt installed in the freqtrade venv. No trading/execution here.
To run:
  cd /opt/multi-strat-engine
  /opt/freqtrade/.venv/bin/python run_scan.py
"""
import asyncio
import ccxt.async_support as ccxt
import time
import traceback
from strategies import Candle, ActiveTrade, run_signal_scan, CONFIG

PAIRS = CONFIG["pairs"]
TF = "1m"
LIMIT = 200
BALANCE = 100.0  # placeholder for sizing; adjust to your available USDT

async def fetch_pair(exchange, pair):
    try:
        ohlcv = await exchange.fetch_ohlcv(pair, timeframe=TF, limit=LIMIT)
        candles = [Candle(open=o[1], high=o[2], low=o[3], close=o[4], volume=o[5], timestamp=o[0]) for o in ohlcv]
        return pair, candles
    except Exception as e:
        print(f"Fetch error {pair}: {e}")
        return pair, None

async def main():
    exchange = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "future"}})
    try:
        tasks = [fetch_pair(exchange, p) for p in PAIRS]
        results = await asyncio.gather(*tasks)
        market_data = {p: c for p, c in results if c}
        active_trades = []  # hook up to real open positions if needed
        res = run_signal_scan(market_data, active_trades, BALANCE)
        print("--- Diagnostics ---")
        d = res.diagnostics
        print(f"Raw: {d.raw_count} -> Cooldown: {d.after_cooldown} -> Flood: {d.after_flood} -> Cat: {d.after_category} -> Final: {d.final}")
        if res.drawdown and res.drawdown.paused:
            print(res.drawdown.message)
        if res.btc_macro and res.btc_macro.is_macro_move:
            print(f"BTC macro move {res.btc_macro.magnitude*100:.2f}% {res.btc_macro.direction}")
        print("\n--- Signals (to execute) ---")
        for s in res.signals:
            print(f"{s.pair} {s.side} {s.strategy_name} conf={s.confidence:.2f} price={s.entry_price} tp={s.tp_price} sl={s.sl_price} lev={s.leverage} size={s.trade_size:.2f} RR={s.economics.risk_reward:.2f} fees={s.economics.total_fees:.4f} reason={s.reason}")
        print("\n--- Blocked ---")
        for b in res.filtered:
            print(f"{b.pair} {b.side} via {b.strategy_name} blocked: {b._filter_reason}")
    finally:
        try:
            await exchange.close()
        except Exception:
            pass

if __name__ == "__main__":
    asyncio.run(main())
