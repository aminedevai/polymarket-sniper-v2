"""
backtest_deep.py
================
Deeper analysis of 5-min BTC markets focused on:
1. US market open window (13:00-15:00 UTC)
2. Larger sample by fetching more markets
3. Momentum: did BTC trend UP or DOWN in the previous candle?
"""

import requests
import json
import time
import csv
import os
import re
from datetime import datetime, timezone, timedelta
from collections import defaultdict

SESSION = requests.Session()
SESSION.headers.update({'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'})
GAMMA_API = "https://gamma-api.polymarket.com"

# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_all_btc_5min(max_markets=3000):
    print("Fetching BTC 5-min markets (all available)...")
    markets = []
    offset  = 0
    while len(markets) < max_markets:
        try:
            r = SESSION.get(f"{GAMMA_API}/markets", params={
                "slug_contains": "btc-updown-5m",
                "closed":        "true",
                "limit":         100,
                "offset":        offset,
                "order":         "closedTime",
                "ascending":     "false",
            }, timeout=15)
            batch = r.json()
            if not batch: break
            markets.extend(batch)
            print(f"  {len(markets)} fetched...", end='\r')
            if len(batch) < 100: break
            offset += 100
            time.sleep(0.2)
        except Exception as e:
            print(f"\n  Error: {e}")
            break
    print(f"\n  Done. {len(markets)} total.")
    return markets

def parse_market(m: dict):
    try:
        outcomes = m.get('outcomes', [])
        prices   = m.get('outcomePrices', [])
        if isinstance(outcomes, str): outcomes = json.loads(outcomes)
        if isinstance(prices,   str): prices   = json.loads(prices)
        if len(outcomes) < 2 or len(prices) < 2: return None

        p0, p1 = float(prices[0]), float(prices[1])
        if   abs(p0 - 1.0) < 0.01: winner_idx = 0
        elif abs(p1 - 1.0) < 0.01: winner_idx = 1
        else: return None

        actual_up = 'up' in outcomes[winner_idx].lower() or 'yes' in outcomes[winner_idx].lower()

        last_price = float(m.get('lastTradePrice', 0) or 0)
        if last_price <= 0.01 or last_price >= 0.99: return None

        end_date = m.get('endDate', '')
        dt = None
        if end_date:
            try:
                dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            except: pass

        # Extract unix timestamp from slug for sequencing
        slug = m.get('slug', '')
        slug_ts = None
        m2 = re.search(r'-(\d{9,11})$', slug)
        if m2:
            slug_ts = int(m2.group(1))

        return {
            'slug':       slug,
            'slug_ts':    slug_ts,
            'end_date':   end_date,
            'dt':         dt,
            'hour_utc':   dt.hour if dt else None,
            'minute_utc': dt.minute if dt else None,
            'up_implied': last_price,
            'actual_up':  actual_up,
            'volume':     float(m.get('volumeNum', 0) or 0),
        }
    except: return None

# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze_time_window(records, hour_start, hour_end, label):
    window = [r for r in records
              if r['hour_utc'] is not None
              and hour_start <= r['hour_utc'] < hour_end]
    outside = [r for r in records
               if r['hour_utc'] is not None
               and not (hour_start <= r['hour_utc'] < hour_end)]

    if not window: return

    n_w   = len(window)
    up_w  = sum(1 for r in window if r['actual_up'])
    n_o   = len(outside)
    up_o  = sum(1 for r in outside if r['actual_up'])

    rate_w = up_w / n_w if n_w else 0
    rate_o = up_o / n_o if n_o else 0

    print(f"\n  {label}")
    print(f"  {'─'*55}")
    print(f"  Inside window  ({hour_start:02d}:00-{hour_end:02d}:00 UTC): "
          f"n={n_w:>4}  UP rate={rate_w:.1%}")
    print(f"  Outside window:                          "
          f"n={n_o:>4}  UP rate={rate_o:.1%}")
    diff = rate_w - rate_o
    print(f"  Difference: {diff:+.1%}  "
          f"{'← BEARISH WINDOW' if diff < -0.05 else '← BULLISH WINDOW' if diff > 0.05 else '(no significant bias)'}")

    # Break down by 30-min sub-windows
    print(f"\n  30-min breakdown:")
    for h in range(hour_start, hour_end):
        for half in [0, 30]:
            sub = [r for r in window
                   if r['hour_utc'] == h
                   and r['minute_utc'] is not None
                   and half <= r['minute_utc'] < half + 30]
            if len(sub) < 3: continue
            r_sub = sum(1 for r in sub if r['actual_up']) / len(sub)
            bar   = "█" * int(r_sub * 20)
            print(f"    {h:02d}:{half:02d} UTC  n={len(sub):>3}  UP={r_sub:.1%}  {bar}")

def analyze_momentum(records):
    """
    Does the previous candle's result predict this candle?
    If UP won last candle → does UP tend to win again? (momentum)
    Or does DOWN win? (mean reversion)
    """
    # Sort by slug_ts to get sequential candles
    sorted_r = sorted([r for r in records if r['slug_ts']], key=lambda x: x['slug_ts'])

    # Build consecutive pairs (5-min apart = 300 seconds)
    pairs = []
    for i in range(1, len(sorted_r)):
        prev = sorted_r[i-1]
        curr = sorted_r[i]
        if prev['slug_ts'] and curr['slug_ts']:
            gap = curr['slug_ts'] - prev['slug_ts']
            if 250 <= gap <= 350:  # consecutive 5-min candles
                pairs.append((prev['actual_up'], curr['actual_up']))

    if not pairs:
        print("\n  MOMENTUM ANALYSIS: Not enough consecutive candles found.")
        return

    n_total   = len(pairs)
    # After UP: what happens next?
    after_up  = [(p,c) for p,c in pairs if p]
    after_dn  = [(p,c) for p,c in pairs if not p]

    cont_up   = sum(1 for _,c in after_up if c) / len(after_up) if after_up else 0
    cont_dn   = sum(1 for _,c in after_dn if not c) / len(after_dn) if after_dn else 0

    print(f"\n  MOMENTUM / MEAN REVERSION ANALYSIS")
    print(f"  {'─'*55}")
    print(f"  Consecutive candle pairs: {n_total}")
    print(f"  After UP candle  → UP again: {cont_up:.1%}  "
          f"({'MOMENTUM' if cont_up > 0.55 else 'MEAN REVERT' if cont_up < 0.45 else 'random'})")
    print(f"  After DOWN candle → DOWN again: {cont_dn:.1%}  "
          f"({'MOMENTUM' if cont_dn > 0.55 else 'MEAN REVERT' if cont_dn < 0.45 else 'random'})")

    if cont_up > 0.55:
        print(f"\n  ✓ MOMENTUM SIGNAL: After UP candle → BET UP next candle")
    elif cont_up < 0.45:
        print(f"\n  ✓ MEAN REVERSION SIGNAL: After UP candle → BET DOWN next candle")
    else:
        print(f"\n  No momentum edge detected.")

def analyze_extreme_prices(records):
    """When market is priced >0.90 or <0.10, is there a reliable fade?"""
    print(f"\n  EXTREME PRICE FADE ANALYSIS")
    print(f"  {'─'*55}")

    for lo, hi, label in [
        (0.90, 1.00, "UP priced 90-100% (extreme bullish)"),
        (0.00, 0.10, "UP priced 0-10%   (extreme bearish = DOWN at 90-100%)"),
        (0.85, 0.95, "UP priced 85-95%  (strong bullish)"),
    ]:
        sub = [r for r in records if lo <= r['up_implied'] < hi]
        if len(sub) < 5: continue
        up_r = sum(1 for r in sub if r['actual_up']) / len(sub)
        implied_mid = sum(r['up_implied'] for r in sub) / len(sub)
        edge = up_r - implied_mid

        # For extreme bearish (price near 0), DOWN edge = actual DOWN rate
        if hi <= 0.15:
            down_r    = 1 - up_r
            down_impl = 1 - implied_mid
            edge_dn   = down_r - down_impl
            print(f"  {label}: n={len(sub):>3}  DOWN implied={down_impl:.1%}  "
                  f"DOWN actual={down_r:.1%}  edge={edge_dn:+.1%}  "
                  f"{'← FADE' if edge_dn > 0.05 else ''}")
        else:
            print(f"  {label}: n={len(sub):>3}  UP implied={implied_mid:.1%}  "
                  f"UP actual={up_r:.1%}  edge={edge:+.1%}  "
                  f"{'← FADE DOWN' if edge < -0.05 else '← RIDE UP' if edge > 0.05 else ''}")

def main():
    markets = fetch_all_btc_5min(max_markets=3000)
    if not markets:
        print("No data.")
        return

    print(f"Parsing {len(markets)} markets...")
    records = [r for m in markets for r in [parse_market(m)] if r]
    print(f"  Parsed: {len(records)}  Skipped: {len(markets)-len(records)}")

    if not records:
        print("No parseable records.")
        return

    n    = len(records)
    up_n = sum(1 for r in records if r['actual_up'])
    print(f"\n{'═'*60}")
    print(f"  DEEP BACKTEST  —  {n} markets  —  UP won {up_n/n:.1%} overall")
    print(f"{'═'*60}")

    # Time window analysis
    analyze_time_window(records, 13, 16, "US MARKET OPEN  (13:00-16:00 UTC / 9am-12pm ET)")
    analyze_time_window(records, 9,  12, "LONDON SESSION  (09:00-12:00 UTC)")
    analyze_time_window(records, 17, 20, "US AFTERNOON    (17:00-20:00 UTC / 1pm-4pm ET)")
    analyze_time_window(records, 0,   5, "ASIA SESSION    (00:00-05:00 UTC)")

    # Momentum
    analyze_momentum(records)

    # Extreme price fade
    analyze_extreme_prices(records)

    # Save
    os.makedirs("logs", exist_ok=True)
    path = "logs/deep_backtest.csv"
    with open(path, 'w', newline='', encoding='utf-8') as f:
        keys = ['slug','end_date','hour_utc','minute_utc','up_implied','actual_up','volume']
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in records:
            w.writerow({k: r[k] for k in keys})
    print(f"\n  Data saved → {path}")

    print(f"\n{'═'*60}")
    print(f"  TRADING RULES DERIVED FROM DATA")
    print(f"{'─'*60}")
    print(f"  1. Check time: bearish 10:00/14:00 UTC, bullish 13:00/17-18:00 UTC")
    print(f"  2. Check price: if UP >95%, consider fading DOWN")
    print(f"  3. Check momentum: see above")
    print(f"  → Combine 2+ signals before placing a bet")

if __name__ == "__main__":
    main()