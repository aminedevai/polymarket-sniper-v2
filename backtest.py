"""
backtest_strat.py - Fast version
Fetches each slug directly instead of scanning all markets.
1440 slugs = ~15 min at 1 req/sec, but we batch with threading.
"""

import requests, json, time, csv, os, re
from datetime import datetime, timezone
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

SESSION = requests.Session()
SESSION.headers.update({'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'})
GAMMA_API = "https://gamma-api.polymarket.com"

START_TS = int(datetime(2026, 2, 16, 0, 0, tzinfo=timezone.utc).timestamp())
END_TS   = int(datetime(2026, 2, 20, 23, 55, tzinfo=timezone.utc).timestamp())

FADE_THRESHOLD   = 0.90
STARTING_CAPITAL = 100.0
BET_SIZE         = 10.0
FEE_PCT          = 0.01

GR="\033[92m";RE="\033[91m";YL="\033[93m";CY="\033[96m";GY="\033[90m";B="\033[1m";R="\033[0m"

# ── Generate all slugs ────────────────────────────────────────────────────────
def all_slugs():
    slugs = []
    ts = START_TS
    while ts <= END_TS:
        slugs.append((ts, f"btc-updown-5m-{ts}"))
        ts += 300
    return slugs

# ── Fetch one slug ────────────────────────────────────────────────────────────
def fetch_slug(ts_slug):
    ts, slug = ts_slug
    try:
        r = SESSION.get(f"{GAMMA_API}/markets",
                        params={"slug": slug}, timeout=10)
        data = r.json()
        if data and isinstance(data, list):
            m = data[0]
            m['_slug_ts'] = ts
            return m
    except Exception:
        pass
    return None

# ── Fetch all with threading ──────────────────────────────────────────────────
def fetch_week():
    slugs = all_slugs()
    print(f"Fetching {len(slugs)} markets directly by slug (Feb 16-20)...")
    print(f"Using 20 parallel workers — estimated ~2 minutes...")

    markets  = []
    done     = 0
    total    = len(slugs)

    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = {ex.submit(fetch_slug, s): s for s in slugs}
        for fut in as_completed(futures):
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{total} fetched, {len(markets)} found...", end='\r')
            result = fut.result()
            if result:
                markets.append(result)

    markets.sort(key=lambda m: m.get('_slug_ts', 0))
    print(f"\n  Done. {len(markets)} markets found.")
    return markets

# ── Parse ─────────────────────────────────────────────────────────────────────
def parse(m: dict):
    try:
        outcomes = m.get('outcomes', [])
        prices   = m.get('outcomePrices', [])
        if isinstance(outcomes, str): outcomes = json.loads(outcomes)
        if isinstance(prices,   str): prices   = json.loads(prices)
        if len(outcomes) < 2 or len(prices) < 2: return None

        p0, p1 = float(prices[0]), float(prices[1])
        if   abs(p0 - 1.0) < 0.01: winner_idx = 0
        elif abs(p1 - 1.0) < 0.01: winner_idx = 1
        else: winner_idx = None

        actual_up = None
        if winner_idx is not None:
            actual_up = 'up' in outcomes[winner_idx].lower()

        last_price = float(m.get('lastTradePrice', 0) or 0)
        slug_ts    = m.get('_slug_ts', 0)
        dt         = datetime.fromtimestamp(slug_ts, tz=timezone.utc)

        return {
            'slug':       m.get('slug',''),
            'slug_ts':    slug_ts,
            'date':       dt.strftime('%Y-%m-%d'),
            'time_str':   dt.strftime('%H:%M'),
            'hour':       dt.hour,
            'last_price': last_price,
            'actual_up':  actual_up,
            'resolved':   actual_up is not None,
            'volume':     float(m.get('volumeNum', 0) or 0),
        }
    except: return None

# ── Strategy ──────────────────────────────────────────────────────────────────
def run_strategy(records: list):
    sorted_r = sorted(records, key=lambda x: x['slug_ts'])
    trades, capital, prev = [], STARTING_CAPITAL, None

    for rec in sorted_r:
        up_price  = rec['last_price']
        actual_up = rec['actual_up']
        resolved  = rec['resolved']

        has_price    = 0.01 < up_price < 0.99
        price_signal = has_price and up_price >= FADE_THRESHOLD
        prev_dn      = prev is not None and not prev['actual_up']

        if price_signal and prev_dn:   signal = 'DOWN'
        elif price_signal:             signal = 'DOWN_WEAK'
        else:                          signal = 'SKIP'

        if signal in ('DOWN','DOWN_WEAK') and resolved and capital >= BET_SIZE:
            down_price = 1.0 - up_price
            shares     = BET_SIZE / down_price
            fee        = BET_SIZE * FEE_PCT
            won        = not actual_up
            profit     = (shares - BET_SIZE - fee) if won else (-BET_SIZE - fee)
            capital   += profit
            trades.append({
                'date':       rec['date'],
                'time':       rec['time_str'],
                'slug':       rec['slug'],
                'up_price':   round(up_price, 4),
                'down_price': round(down_price, 4),
                'signal':     signal,
                'prev_down':  prev_dn,
                'actual_up':  actual_up,
                'won':        won,
                'bet':        BET_SIZE,
                'profit':     round(profit, 4),
                'capital':    round(capital, 4),
                'fee':        round(fee, 4),
            })

        if resolved:
            prev = rec

    return trades

# ── Print ─────────────────────────────────────────────────────────────────────
def print_results(trades, records):
    resolved  = [r for r in records if r['resolved']]
    valid     = [r for r in records if 0.01 < r['last_price'] < 0.99]
    fade_opps = [r for r in valid if r['last_price'] >= FADE_THRESHOLD]

    n     = len(trades)
    n_won = sum(1 for t in trades if t['won'])
    pnl   = sum(t['profit'] for t in trades)
    fees  = sum(t['fee'] for t in trades)
    wr    = n_won / n if n else 0
    final = STARTING_CAPITAL + pnl
    pc    = GR if pnl >= 0 else RE

    print(f"\n{'═'*72}")
    print(f"  {B}FADE STRATEGY BACKTEST  —  Feb 16-20 2026{R}")
    print(f"{'═'*72}")
    print(f"\n  Markets fetched:      {len(records)}")
    print(f"  Resolved:             {len(resolved)}")
    print(f"  Valid pre-res price:  {len(valid)}")
    print(f"  Fade opportunities:   {len(fade_opps)}  (UP >= {FADE_THRESHOLD:.0%})")
    print(f"  Signals fired:        {n}")

    if not trades:
        print(f"\n  {YL}No trades fired.{R}")
        if fade_opps:
            print(f"  Sample fade opportunities:")
            for r in fade_opps[:5]:
                print(f"    {r['date']} {r['time_str']}  UP={r['last_price']:.0%}  "
                      f"resolved={r['resolved']}  actual_up={r['actual_up']}")
        return

    print(f"  Won / Lost:           {GR}{n_won}{R} / {RE}{n-n_won}{R}")
    print(f"  Win rate:             {GR if wr>=0.6 else RE}{wr:.1%}{R}  "
          f"(breakeven ~55% at 1:9 odds)")
    print(f"  Total P&L:            {pc}${pnl:+.2f}{R}  (fees: {GY}${fees:.2f}{R})")
    print(f"  Capital: $100 → {CY}${final:.2f}{R}  ({pc}{pnl:+.1f}%{R})")

    # Trade log
    print(f"\n  {B}FULL TRADE LOG{R}")
    print(f"  {'─'*72}")
    print(f"  {'Date':<11} {'Time':>5} {'UP%':>5} {'DN%':>5} "
          f"{'Signal':>9} {'Prev':>4} {'Result':>6} {'P&L':>8} {'Capital':>9}")
    print(f"  {'─'*72}")
    for t in trades:
        rc  = GR if t['won'] else RE
        sc  = YL if t['signal']=='DOWN' else GY
        prv = "DN" if t['prev_down'] else "UP"
        res = "WIN" if t['won'] else "LOSS"
        print(f"  {t['date']:<11} {t['time']:>5} "
              f"{t['up_price']:>4.0%} {t['down_price']:>4.0%} "
              f"{sc}{t['signal']:>9}{R} "
              f"{prv:>4} {rc}{res:>6}{R} "
              f"{rc}${t['profit']:>+7.2f}{R} "
              f"{CY}${t['capital']:>8.2f}{R}")

    # By day
    print(f"\n  {B}BY DAY{R}")
    print(f"  {'─'*45}")
    by_day = defaultdict(list)
    for t in trades: by_day[t['date']].append(t)
    for day in sorted(by_day.keys()):
        dt  = by_day[day]
        p   = sum(x['profit'] for x in dt)
        wr2 = sum(1 for x in dt if x['won'])/len(dt)
        col = GR if p >= 0 else RE
        print(f"  {day}  n={len(dt):>2}  win={wr2:.0%}  "
              f"P&L={col}${p:+.2f}{R}")

    # Strong vs weak breakdown
    strong = [t for t in trades if t['signal']=='DOWN']
    weak   = [t for t in trades if t['signal']=='DOWN_WEAK']
    print(f"\n  {B}SIGNAL QUALITY{R}")
    print(f"  {'─'*45}")
    if strong:
        ws = sum(1 for t in strong if t['won'])/len(strong)
        ps = sum(t['profit'] for t in strong)
        print(f"  Strong (price+momentum):  n={len(strong):>2}  "
              f"win={GR if ws>=0.6 else RE}{ws:.0%}{R}  P&L=${ps:+.2f}")
    if weak:
        ww = sum(1 for t in weak if t['won'])/len(weak)
        pw = sum(t['profit'] for t in weak)
        print(f"  Weak   (price only):      n={len(weak):>2}  "
              f"win={GR if ww>=0.6 else RE}{ww:.0%}{R}  P&L=${pw:+.2f}")
    if strong and weak:
        ws = sum(1 for t in strong if t['won'])/len(strong)
        ww = sum(1 for t in weak   if t['won'])/len(weak)
        print(f"\n  → {'USE STRONG SIGNAL ONLY' if ws > ww else 'BOTH SIMILAR PERFORMANCE'}")

    # Save CSV
    os.makedirs("logs", exist_ok=True)
    path = "logs/strat_backtest_feb16_20.csv"
    with open(path,'w',newline='',encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(trades[0].keys()))
        w.writeheader(); w.writerows(trades)
    print(f"\n  Saved → {path}")

def main():
    markets = fetch_week()
    if not markets:
        print("No markets returned.")
        return
    print(f"Parsing...")
    records = [r for m in markets for r in [parse(m)] if r]
    print(f"  Parsed: {len(records)}")
    trades = run_strategy(records)
    print_results(trades, records)

if __name__ == "__main__":
    main()