# Polymarket Sniper v2 — Complete Architecture




## Project Structure

```
polymarket-sniper-v2/
├── .env.template          ← Copy to .env and fill in credentials
├── requirements.txt       ← pip install -r requirements.txt
├── config.yaml            ← All runtime parameters
├── main.py                ← Entry point — run this
├── approve.py             ← ONE-TIME: approve USDC/CTF for EOA wallets
│
├── core/
│   ├── state.py           ← Shared state bus (single source of truth)
│   ├── binance_listener.py← Binance Futures aggTrade WS + EMA velocity
│   ├── poly_book_listener.py ← Polymarket CLOB WS orderbook
│   ├── rtds_listener.py   ← RTDS: Binance + Chainlink feeds (divergence)
│   ├── market_finder.py   ← Gamma API poller + market rotation
│   └── executor.py        ← py-clob-client v0.34.5 order placement
│
├── strategy/
│   ├── signal_engine.py   ← Combined velocity + divergence signals
│   ├── taker_sniper.py    ← FOK market order execution
│   └── maker_quoter.py    ← Post-only limit order quoting
│
├── risk/
│   ├── fee_calculator.py  ← Real Polymarket fee curve
│   └── position_manager.py← Kelly sizing + 20% drawdown kill switch
│
└── utils/
    ├── logger.py          ← JSON + human-readable logging
    └── metrics.py         ← P&L, win rate, FOK rejection tracking
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set credentials
cp .env.template .env
# Edit .env: add POLYGON_PRIVATE_KEY

# 3. (EOA wallets only) One-time contract approval
python approve.py

# 4. Run in paper trade mode (default)
python main.py

# 5. Check logs/
#    bot.log     — structured JSON (every event)
#    trades_*.csv — all signals and outcomes
```

## Signal Types

| Type | Triggers when | Confidence min |
|------|--------------|----------------|
| COMBINED | Velocity + Divergence agree | 0.40 |
| VELOCITY | Binance moving fast + oracle stale ≥ 10s | 0.50 |
| DIVERGENCE | Binance vs Chainlink gap ≥ 0.15% | 0.55 |

## Data Sources Used

| Source | Feed | Purpose |
|--------|------|---------|
| Binance Futures WS | `fstream.binance.com` aggTrade | Raw price + velocity |
| Polymarket RTDS | `crypto_prices` | Binance price via Poly server |
| Polymarket RTDS | `crypto_prices_chainlink` | Settlement oracle price |
| Polymarket CLOB WS | `ws-subscriptions-clob` | Live orderbook bid/ask |
| Polymarket Gamma API | REST | Market rotation / token IDs |

## Fee Zones (trade only in LOW zone)

| Price | Eff. Fee | Zone | Trade? |
|-------|----------|------|--------|
| 0.10 | 2.0% | LOW | ✓ |
| 0.20 | 3.2% | MID | ✗ |
| 0.50 | 3.1% | PEAK | ✗ |
| 0.75 | 1.2% | LOW | ✓ |
| 0.80 | 0.8% | LOW | ✓ |
| 0.90 | 0.2% | VERY LOW | ✓ |

## Go-Live Checklist

- [ ] 48h paper trade with ≥ 10 signals/hour
- [ ] Signal accuracy > 62% on settled trades
- [ ] FOK rejection rate < 50%
- [ ] Set `paper_trade: false` in config.yaml
- [ ] Set `max_position_usdc: 5.0` for first live day
- [ ] Monitor `logs/trades_*.csv` in real time
- [ ] Scale to 50 USDC max only after 3 profitable live days
