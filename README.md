# Multi‑Strat Engine (Binance USDT‑M Futures)

A lightweight multi‑strategy crypto futures bot with strict risk filters, TP/SL, and reporting pipelines. Built for short‑term (1m) scalps with configurable safeguards.

> **Repo note:** This repo contains `multi-strat-engine/` (engine code) alongside the existing freqtrade setup.

---

## Features
- 10 strategies (trend / reversion / structural)
- 1m scanning with 200‑bar history
- TP/SL reduce‑only orders (ATR‑scaled + min TP)
- Cooldowns, correlation caps, drawdown circuit breaker
- Funding/volatility/trend filters
- **Adaptive parameters** (RSI Snap + EMA trend filter, slow updates)
- **Divergence boost** (market‑wide RSI/price divergence)
- **Open‑interest boost** (confidence only)
- **Spread gate** (pauses pairs when spread spikes)
- **BTC dump correlated exit** (close longs on 0.5% 1m BTC drop)
- **Momentum exhaustion exit** (early exit on RSI+volume fade)
- **Opportunity‑cost swap** (swap into stronger signal, 1/hr)
- Post‑trade reporting (PnL, heatmap, slippage)
- Systemd service for always‑on operation

---

## Requirements
- Linux host
- Python 3.10+
- Binance USDT‑M futures API key/secret (IP‑restricted recommended)
- CCXT

### Python env
The engine uses the existing freqtrade venv:
```
/opt/freqtrade/.venv
```

If you need to install deps:
```
source /opt/freqtrade/.venv/bin/activate
pip install ccxt
```

---

## Project Layout
```
/opt/multi-strat-engine/
  strategies.py        # all strategies + filters + config
  trade_loop.py        # live runner (trading)
  run_scan.py          # signal‑only runner
  reports/             # reporting scripts
```

---

## Setup
1) **API keys**
Edit:
```
/opt/freqtrade/user_data/config.binance_futures_live.json
```
Ensure:
```
exchange.key
exchange.secret
```

2) **Run the bot (systemd)**
```
systemctl --user enable --now multistrat.service
systemctl --user status multistrat.service
```

3) **Manual run**
```
cd /opt/multi-strat-engine
/opt/freqtrade/.venv/bin/python trade_loop.py
```

---

## Strategies
**Trend/Breakout:**
- ema_scalp
- triple_ema
- macd_flip
- atr_breakout
- bb_squeeze

**Mean‑reversion:**
- rsi_snap
- stoch_cross
- obv_divergence
- vwap_bounce

**Structural:**
- engulfing_sr

Max‑age rules (current):
- Trend/Breakout: **40 min**
- Reversion: **15 min**
- Structural: **35 min**

Extra exits:
- **BTC dump exit:** close longs if BTC drops ≥0.5% in 1m
- **Momentum exhaustion exit:** RSI extreme + 3 candles + volume fade

---

## Risk + Filters (current defaults)
- `confidence_threshold`: 0.64
- `confirm_signal`: true (2‑candle confirmation)
- `min_volatility_pct`: 0.2%
- `trend_ema_fast/slow`: 50 / 200 (adaptive)
- `regime_min_trend_pct`: 0.15%
- `cooldown_sec`: 300
- `post_close_cooldown_sec`: 600
- `max_concurrent_trades`: 4
- `max_drawdown_pause`: 30%
- `signal_decay`: enabled (half‑life 8m)
- `divergence_boost`: +0.04
- `oi_boost`: +0.06
- `spread_gate`: 2× 1‑hour avg

### Pair‑specific strategy limits
- `LINKUSDT`: RSI Snap, Stoch Cross, OBV Divergence only
- `BNBUSDT`: RSI Snap, VWAP Bounce only

---

## Trading Logic (high‑level)
1. Fetch 1m OHLCV for all pairs
2. Evaluate all strategies per pair
3. Apply filters:
   - confidence threshold
   - confirmation
   - volatility + trend alignment
   - funding gate
   - correlation caps + cooldowns
4. Execute market entry
5. Place TP/SL reduce‑only orders
6. Enforce max‑age exits

---

## Reporting & Pipelines
Reports live in:
```
/opt/multi-strat-engine/reports/
```

Scripts:
- `health_check.py` – bot status + TP/SL check
- `post_trade_pipeline.py` – realized PnL + fees + funding
- `pair_performance.py` – per‑coin PnL
- `strategy_performance.py` – per‑strategy PnL (mapped)
- `heatmap_hour.py` – hourly PnL (IST)
- `slippage_fee_report.py` – slippage + fee analysis
- `tp_sl_check.py` – 15‑min TP/SL alerts
- `pipeline_health.py` – read‑only health monitoring

> CSV outputs are ignored by `.gitignore`.

---

## Service
Systemd unit:
```
~/.config/systemd/user/multistrat.service
```
Logs:
```
/home/ubuntu/.openclaw/workspace/logs/multistrat.log
/home/ubuntu/.openclaw/workspace/logs/multistrat.err
```

---

## Safety Notes
- Always use **reduce‑only** TP/SL
- Keep leverage capped in strategy definitions
- Start with small size and monitor PnL
- Avoid running in illiquid hours

---

## Quick Commands
```
# status
systemctl --user status multistrat.service

# restart
systemctl --user restart multistrat.service

# signal scan only
/opt/freqtrade/.venv/bin/python /opt/multi-strat-engine/run_scan.py
```

---

## License
Internal / private use (adjust as needed).
