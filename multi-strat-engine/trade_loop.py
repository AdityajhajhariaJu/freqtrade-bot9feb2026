#!/usr/bin/env python3
"""
Live trade loop for the multi-strategy engine (Binance USDT-M futures).
- Uses freqtrade config keys.
- Pairs: ETHUSDT, BNBUSDT, SOLUSDT, LINKUSDT, LTCUSDT, DOGEUSDT
- 1m scans, market entries, TP/SL reduce-only.
- One-way mode assumed.
- Max position age per strategy applied (upper bounds), default 25m if unknown.
- On startup: cancels ALL open orders on the configured pairs (does NOT close positions).
- On startup: seeds position metadata from existing positions (updateTime) to enforce max-age across restarts.
- Persists strategy_id on entry for max-age tracking.
"""
import asyncio
import json
import time
import ccxt.async_support as ccxt
from strategies import Candle, ActiveTrade, run_signal_scan, CONFIG as STRAT_CONFIG, set_pair_cooldown

CONFIG_PATH = "/opt/freqtrade/user_data/config.binance_futures_live.json"
PAIRS = ["ETHUSDT", "SOLUSDT", "LTCUSDT", "DOGEUSDT", "AVAXUSDT", "NEARUSDT", "INJUSDT", "LINKUSDT", "BNBUSDT"]
FETCH_LIMIT = 200
TIMEFRAME = "1m"
LOOP_SEC = 60
DEFAULT_MAX_AGE = 25 * 60  # 25 minutes
# Max ages (seconds) per strategy (upper bound minutes)
MAX_AGE = {
    # Trend/Breakout -> 40m
    "ema_scalp": 40 * 60,
    "bb_squeeze": 40 * 60,
    "macd_flip": 40 * 60,
    "atr_breakout": 40 * 60,
    "triple_ema": 40 * 60,
    # Reversion -> 15m
    "rsi_snap": 15 * 60,
    "vwap_bounce": 15 * 60,
    "stoch_cross": 15 * 60,
    "obv_divergence": 15 * 60,
    # Structural -> 35m
    "engulfing_sr": 35 * 60,
}

# In-memory tracker: pair -> {strategy_id, opened_at}
position_meta = {}
_LAST_SWAP_TS = 0



def log_event(event, pair, side, qty=None, price=None, strategy_id="", strategy_name="", pnl=None, note=""):
    from pathlib import Path
    import csv, time
    fpath = Path('/opt/multi-strat-engine/trade_events.csv')
    exists = fpath.exists()
    with fpath.open('a', newline='') as f:
        w=csv.writer(f)
        if not exists:
            w.writerow(["ts","event","pair","side","qty","price","strategy_id","strategy_name","pnl","note"])
        w.writerow([int(time.time()), event, pair, side, qty or "", price or "", strategy_id, strategy_name, pnl or "", note])

def load_keys():
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)
    api_key = cfg.get("exchange", {}).get("key")
    secret = cfg.get("exchange", {}).get("secret")
    if not api_key or not secret:
        raise RuntimeError("API keys not found in freqtrade config")
    return api_key, secret


async def fetch_candles(exchange, pair):
    try:
        ohlcv = await exchange.fetch_ohlcv(pair, timeframe=TIMEFRAME, limit=FETCH_LIMIT)
        candles = [Candle(open=o[1], high=o[2], low=o[3], close=o[4], volume=o[5], timestamp=o[0]) for o in ohlcv]
        return pair, candles
    except Exception as e:
        print(f"Fetch error {pair}: {e}")
        return pair, None


async def fetch_positions(exchange):
    try:
        positions = await exchange.fetch_positions()
    except Exception as e:
        print(f"Fetch positions error: {e}")
        return []
    active = []
    for p in positions:
        amt = float(p.get('contracts') or p.get('contractSize') or 0)
        if amt == 0:
            continue
        pair = p.get('symbol') or p.get('info', {}).get('symbol') or p.get('id')
        side_field = p.get('side') or p.get('info', {}).get('positionSide') or ''
        side = 'LONG' if str(side_field).lower() == 'long' else 'SHORT'
        active.append(ActiveTrade(pair=pair, strategy_id="", side=side))
    return active


async def seed_position_meta(exchange):
    try:
        positions = await exchange.fetch_positions()
    except Exception as e:
        print(f"Seed positions error: {e}")
        return
    now = time.time()
    for p in positions:
        amt = float(p.get('contracts') or p.get('contractSize') or 0)
        if amt == 0:
            continue
        pair = p.get('symbol') or p.get('info', {}).get('symbol') or p.get('id')
        upd_ms = float(p.get('info', {}).get('updateTime') or p.get('timestamp') or now * 1000)
        opened_at = upd_ms / 1000.0
        if pair not in position_meta:
            position_meta[pair] = {"strategy_id": "", "opened_at": opened_at}
            print(f"Seeded meta for {pair} opened_at={opened_at}")


async def set_oneway(exchange):
    try:
        await exchange.set_position_mode(False)  # one-way
    except Exception as e:
        print(f"set_position_mode warning: {e}")


async def ensure_leverage(exchange, pair, lev):
    try:
        await exchange.set_leverage(lev, pair)
    except Exception as e:
        print(f"set_leverage warning {pair} {lev}x: {e}")


async def cancel_all_open_orders(exchange):
    try:
        exchange.options['warnOnFetchOpenOrdersWithoutSymbol'] = False
    except Exception:
        pass
    for sym in PAIRS:
        try:
            await exchange.cancel_all_orders(sym)
            print(f"cancel_all_orders called for {sym}")
        except Exception as e:
            print(f"cancel_all_orders error {sym}: {e}")


async def place_orders(exchange, signal):
    side = 'buy' if signal.side == 'LONG' else 'sell'
    notional = signal.trade_size * signal.leverage
    amount = notional / signal.entry_price
    params = {"type": "MARKET"}
    try:
        await exchange.create_order(signal.pair, 'market', side, amount, None, params)
    except Exception as e:
        print(f"Entry error {signal.pair} {side}: {e}")
        return
    reduce_params = {"reduceOnly": True}
    try:
        await exchange.create_order(signal.pair, 'take_profit_market', 'sell' if signal.side == 'LONG' else 'buy', amount, None, {**reduce_params, "stopPrice": signal.tp_price})
    except Exception as e:
        print(f"TP error {signal.pair}: {e}")
    try:
        await exchange.create_order(signal.pair, 'stop_market', 'sell' if signal.side == 'LONG' else 'buy', amount, None, {**reduce_params, "stopPrice": signal.sl_price})
    except Exception as e:
        print(f"SL error {signal.pair}: {e}")
    position_meta[signal.pair] = {"strategy_id": signal.strategy_id, "opened_at": time.time(), "confidence": signal.confidence}
    log_event("ENTRY", signal.pair, signal.side, qty=amount, price=signal.entry_price, strategy_id=signal.strategy_id, strategy_name=signal.strategy_name, note=signal.reason)
    print(f"EXECUTED {signal.pair} {signal.side} size=${signal.trade_size:.2f} lev={signal.leverage} entry~{signal.entry_price} tp={signal.tp_price} sl={signal.sl_price} strat={signal.strategy_name} reason={signal.reason}")


async def close_position(exchange, pair, side, amount):
    close_side = 'sell' if side == 'LONG' else 'buy'
    try:
        await exchange.create_order(pair, 'market', close_side, amount, None, {"reduceOnly": True})
        log_event("CLOSE_MAX_AGE", pair, side, qty=amount, price=None, pnl=None, note="max_age")
        print(f"Closed {pair} {side} due to max age")
    except Exception as e:
        print(f"Close error {pair}: {e}")




def should_momentum_exit(candles, side):
    if len(candles) < 15: return False
    closes = [c.close for c in candles]
    r = rsi(closes, 14)
    last3 = candles[-3:]
    vols = [c.volume for c in last3]
    vol_down = vols[0] > vols[1] > vols[2]
    if side == "LONG":
        green = all(c.close > c.open for c in last3)
        return r >= 75 and green and vol_down
    else:
        red = all(c.close < c.open for c in last3)
        return r <= 25 and red and vol_down
def position_age_seconds(pair, p_obj):
    meta = position_meta.get(pair)
    strat_id = meta.get('strategy_id') if meta else None
    opened_at = meta.get('opened_at') if meta else None
    if not opened_at:
        upd_ms = float(p_obj.get('info', {}).get('updateTime') or p_obj.get('timestamp') or time.time()*1000)
        opened_at = upd_ms / 1000.0
        position_meta[pair] = {"strategy_id": strat_id or "", "opened_at": opened_at}
    return strat_id, time.time() - opened_at

async def loop():
    api_key, secret = load_keys()
    exchange = ccxt.binance({
        "apiKey": api_key,
        "secret": secret,
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })
    await cancel_all_open_orders(exchange)
    await set_oneway(exchange)
    await seed_position_meta(exchange)
    try:
        while True:
            start = time.time()
            tasks = [fetch_candles(exchange, p) for p in PAIRS]
            results = await asyncio.gather(*tasks)
            market_data = {p: c for p, c in results if c}
            active_trades = await fetch_positions(exchange)
            try:
                positions = await exchange.fetch_positions()
                current_pairs = set()
                for p in positions:
                    amt = float(p.get('contracts') or p.get('contractSize') or 0)
                    if amt == 0:
                        continue
                    pair = p.get('symbol') or p.get('info', {}).get('symbol') or p.get('id')
                    current_pairs.add(pair)
                    strat_id, age_sec = position_age_seconds(pair, p)
                    max_age = MAX_AGE.get(strat_id) if strat_id else DEFAULT_MAX_AGE
                    if max_age and age_sec > max_age:
                        side_field = p.get('side') or p.get('info', {}).get('positionSide') or ''
                        side = 'LONG' if str(side_field).lower() == 'long' else 'SHORT'
                        await close_position(exchange, pair, side, amt)
                        position_meta.pop(pair, None)
                # apply post-close cooldowns for pairs that closed via TP/SL
                post_close = STRAT_CONFIG.get("correlation", {}).get("post_close_cooldown_sec", 0)
                cooldown = STRAT_CONFIG.get("correlation", {}).get("cooldown_sec", 0)
                extra = max(0, post_close - cooldown)
                for pair in list(position_meta.keys()):
                    if pair not in current_pairs:
                        ts = time.time() + extra if extra else time.time()
                        set_pair_cooldown(pair, ts)
                        position_meta.pop(pair, None)
                        print(f"Post-close cooldown set for {pair} ({post_close}s)")
            except Exception as e:
                print(f"Age check error: {e}")
            try:
                balance = await exchange.fetch_balance()
                free_usdt = balance.get('USDT', {}).get('free', 0)
            except Exception as e:
                print(f"Balance fetch error: {e}")
                free_usdt = 100.0
            # fetch funding rates
            funding = {}
            try:
                for sym in PAIRS:
                    data = await exchange.fapiPublicGetPremiumIndex({"symbol": sym})
                    funding[sym] = float(data.get("lastFundingRate", 0.0))
            except Exception as e:
                print(f"Funding fetch error: {e}")
            res = run_signal_scan(market_data, active_trades, free_usdt, funding)
            if res.drawdown and res.drawdown.paused:
                print(res.drawdown.message or "Drawdown breaker active; skipping")
            else:
                # opportunity cost swap (fee-aware)
                global _LAST_SWAP_TS
                if res.signals:
                    slots_available = STRAT_CONFIG["max_concurrent_trades"] - len(active_trades)
                    if slots_available <= 0 and time.time() - _LAST_SWAP_TS > 3600:
                        best = max(res.signals, key=lambda s: s.confidence)
                        try:
                            positions = await exchange.fetch_positions()
                            candidates = []
                            for p in positions:
                                amt = float(p.get('contracts') or p.get('contractSize') or 0)
                                if amt == 0: continue
                                pair = p.get('symbol') or p.get('info', {}).get('symbol') or p.get('id')
                                side_field = p.get('side') or p.get('info', {}).get('positionSide') or ''
                                side = 'LONG' if str(side_field).lower() == 'long' else 'SHORT'
                                pnl = float(p.get('unrealizedPnl') or 0)
                                meta = position_meta.get(pair, {})
                                conf = float(meta.get('confidence') or 0)
                                if pnl <= 0:
                                    candidates.append((conf, pnl, pair, side, amt))
                            if candidates:
                                weakest = min(candidates, key=lambda x: x[0])
                                if best.confidence >= weakest[0] + 0.10:
                                    _, _, pair, side, amt = weakest
                                    await close_position(exchange, pair, side, amt)
                                    position_meta.pop(pair, None)
                                    _LAST_SWAP_TS = time.time()
                        except Exception as e:
                            print(f"Swap check error: {e}")
                for sig in res.signals:
                    await ensure_leverage(exchange, sig.pair, sig.leverage)
                    await place_orders(exchange, sig)
            d = res.diagnostics
            print(f"Cycle diag: Raw {d.raw_count} -> Cooldown {d.after_cooldown} -> Flood {d.after_flood} -> Cat {d.after_category} -> Final {d.final}")
            elapsed = time.time() - start
            await asyncio.sleep(max(5, LOOP_SEC - elapsed))
    finally:
        try:
            await exchange.close()
        except Exception:
            pass

if __name__ == "__main__":
    asyncio.run(loop())
