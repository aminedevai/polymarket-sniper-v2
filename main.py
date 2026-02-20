"""
main.py - Polymarket Copy Trader with Persistent Memory + Manual Close
"""

import asyncio
import json
import logging
import os
import re
import sys
import time
import threading
import requests
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime, timezone

import yaml
from dotenv import load_dotenv

load_dotenv()

def load_config(path="config.yaml"):
    try:
        with open(path) as f: return yaml.safe_load(f)
    except FileNotFoundError: return {}

cfg      = load_config()
LOG_FILE = cfg.get("log_file", "logs/bot.log")

# ── CONFIG ───────────────────────────────────────────────────────────────────
TARGET_WALLET   = "0x63ce342161250d705dc0b16df89036c8e5f9ba9a"
DATA_API        = "https://data-api.polymarket.com"
GAMMA_API       = "https://gamma-api.polymarket.com"
STARTING_BUDGET = 100.0
COPY_SCALE      = 0.5
POLL_INTERVAL   = 5
MEMORY_FILE     = "logs/trader_memory.json"

# ── COLORS ───────────────────────────────────────────────────────────────────
class C:
    R="\033[0m"; B="\033[1m"; CY="\033[96m"; MG="\033[95m"
    GR="\033[92m"; RE="\033[91m"; YL="\033[93m"; BL="\033[94m"
    WH="\033[97m"; GY="\033[90m"; OR="\033[38;5;208m"

def _c(t,c): return f"{c}{t}{C.R}"
def green(t):  return _c(t, C.GR)
def red(t):    return _c(t, C.RE)
def yel(t):    return _c(t, C.YL)
def cyan(t):   return _c(t, C.CY)
def gray(t):   return _c(t, C.GY)
def bold(t):   return _c(t, C.B)
def blue(t):   return _c(t, C.BL)
def orange(t): return _c(t, C.OR)
def mg(t):     return _c(t, C.MG)
def pnlc(v,t): return green(t) if v >= 0 else red(t)

def trunc(t, n): return t[:n-2]+".." if len(t) > n else t

def strip_ansi(t):
    return re.sub(r'\033\[[0-9;]*m', '', t)

def pad(t, n, align='<'):
    raw   = strip_ansi(t)
    extra = len(t) - len(raw)
    w     = n + extra
    if align == '>': return t.rjust(w)
    if align == '^': return t.center(w)
    return t.ljust(w)

# ── HELPERS ──────────────────────────────────────────────────────────────────

def end_date_from_slug(slug: str) -> str:
    """
    Extract end date directly from slug timestamp.
    e.g. btc-updown-5m-1771595700  ->  2026-02-20T13:55:00+00:00
    No API call needed — the timestamp IS the market close time.
    """
    m = re.search(r'-(\d{9,11})$', slug)
    if m:
        try:
            ts = int(m.group(1))
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            return dt.isoformat()
        except Exception:
            pass
    return ""

def fetch_end_date_gamma(slug: str) -> str:
    """Fallback: fetch end date from Gamma API."""
    if not slug: return ""
    try:
        r = requests.get(
            f"{GAMMA_API}/events",
            params={"slug": slug},
            timeout=5,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        data = r.json()
        if isinstance(data, list) and data:
            # Try event-level endDate
            ev = data[0]
            if ev.get("endDate"): return ev["endDate"]
            markets = ev.get("markets", [])
            if markets: return markets[0].get("endDate", "")
    except Exception:
        pass
    return ""

def get_end_date(slug: str, raw_end: str) -> str:
    """Get best end date: from raw API > slug parse > Gamma fallback."""
    if raw_end: return raw_end
    from_slug = end_date_from_slug(slug)
    if from_slug: return from_slug
    return fetch_end_date_gamma(slug)

def time_left(end_str: str):
    """Returns (colored string, total_seconds). -1 = expired."""
    if not end_str: return gray("Unknown"), 0
    try:
        end = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        if end <= now: return red("ENDED"), -1
        tot = int((end - now).total_seconds())
        d, rem = divmod(tot, 86400)
        h, rem = divmod(rem, 3600)
        m, s   = divmod(rem, 60)
        if d > 0:   return yel(f"{d}d {h}h"), tot
        elif h > 0: return yel(f"{h}h {m}m"), tot
        elif m > 0: return orange(f"{m}m {s}s"), tot
        else:       return red(f"{s}s"), tot
    except Exception:
        return gray("Unknown"), 0

# ── MEMORY ───────────────────────────────────────────────────────────────────

def load_memory() -> dict:
    os.makedirs("logs", exist_ok=True)
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE) as f:
                data = json.load(f)
            bal = data.get('balance', STARTING_BUDGET)
            n   = len(data.get('closed_trades', []))
            print(f"  {green('Memory loaded')}  balance={cyan(f'${bal:.2f}')}  "
                  f"history={yel(str(n))} closed trades")
            return data
        except Exception as e:
            print(f"  {red('Memory error')}: {e} — starting fresh")
    return {}

def save_memory(m: dict):
    try:
        with open(MEMORY_FILE, 'w') as f:
            json.dump(m, f, indent=2)
    except Exception as e:
        logging.getLogger("memory").error(f"Save failed: {e}")

# ── DATA CLASSES ─────────────────────────────────────────────────────────────

@dataclass
class Position:
    key:          str
    market_title: str
    outcome:      str
    slug:         str
    condition_id: str
    entry_price:  float
    cur_price:    float
    shares:       float
    entry_amount: float
    cur_value:    float
    end_date:     str
    opened_at:    float = field(default_factory=time.time)

    @property
    def url(self):
        return f"https://polymarket.com/event/{self.slug}" if self.slug else "N/A"
    @property
    def pnl(self):
        return self.cur_value - self.entry_amount
    @property
    def roi_pct(self):
        return ((self.cur_price - self.entry_price) / self.entry_price * 100) if self.entry_price > 0 else 0.0

@dataclass
class ClosedTrade:
    key:          str
    market_title: str
    outcome:      str
    slug:         str
    entry_price:  float
    exit_price:   float
    entry_amount: float
    exit_amount:  float
    realized_pnl: float
    closed_at:    float = field(default_factory=time.time)
    reason:       str   = "closed"

    @property
    def url(self):
        return f"https://polymarket.com/event/{self.slug}" if self.slug else "N/A"
    @property
    def roi_pct(self):
        return ((self.exit_price - self.entry_price) / self.entry_price * 100) if self.entry_price > 0 else 0.0

    def to_dict(self):
        return {
            'key': self.key, 'market_title': self.market_title,
            'outcome': self.outcome, 'slug': self.slug,
            'entry_price': self.entry_price, 'exit_price': self.exit_price,
            'entry_amount': self.entry_amount, 'exit_amount': self.exit_amount,
            'realized_pnl': self.realized_pnl, 'closed_at': self.closed_at,
            'reason': self.reason,
        }

    @classmethod
    def from_dict(cls, d):
        keys = ['key','market_title','outcome','slug','entry_price','exit_price',
                'entry_amount','exit_amount','realized_pnl','closed_at','reason']
        return cls(**{k: d[k] for k in keys if k in d})

# ── COPY TRADER ──────────────────────────────────────────────────────────────

class CopyTrader:
    def __init__(self, wallet: str, memory: dict):
        self.wallet    = wallet
        self.scale     = COPY_SCALE
        self.available = memory.get('balance', STARTING_BUDGET)
        self.invested  = memory.get('invested', 0.0)
        self.returned  = memory.get('returned', 0.0)
        self.realized  = memory.get('realized', 0.0)
        self.session_start = self.available
        self.closed_trades: List[ClosedTrade] = [
            ClosedTrade.from_dict(d) for d in memory.get('closed_trades', [])
        ]
        self.positions:       Dict[str, Position] = {}
        self._prev_target:    Dict[str, dict]     = {}
        self.manual_queue:    List[str]            = []  # keys to manually close

        self.session = requests.Session()
        self.session.headers.update({'Accept': 'application/json', 'User-Agent': 'Mozilla/5.0'})
        self.logger = logging.getLogger("trader")

    @property
    def budget(self):
        return self.available + self.invested

    def to_memory(self):
        return {
            'balance':      self.available,
            'invested':     self.invested,
            'returned':     self.returned,
            'realized':     self.realized,
            'closed_trades': [t.to_dict() for t in self.closed_trades[-500:]],
            'saved_at':     datetime.now().isoformat(),
        }

    def _fetch_raw(self) -> Dict[str, dict]:
        try:
            r = self.session.get(
                f"{DATA_API}/positions",
                params={"user": self.wallet, "sizeThreshold": 0.01, "limit": 500},
                timeout=15,
            )
            r.raise_for_status()
            out = {}
            for p in r.json():
                cid = p.get('conditionId', '')
                oc  = p.get('outcome', 'Unknown')
                key = f"{cid}_{oc}"
                # Try all possible end date field names
                raw_end = (p.get('endDate') or p.get('end_date') or
                           p.get('endDateIso') or p.get('expirationDate') or '')
                slug = p.get('slug', '')
                out[key] = {
                    'condition_id':  cid,
                    'market_title':  p.get('title', 'Unknown'),
                    'outcome':       oc,
                    'avg_price':     float(p.get('avgPrice', 0) or 0),
                    'cur_price':     float(p.get('curPrice', 0) or 0),
                    'shares':        float(p.get('size', 0) or 0),
                    'current_value': float(p.get('currentValue', 0) or 0),
                    'slug':          slug,
                    'end_date':      get_end_date(slug, raw_end),
                }
            return out
        except Exception as e:
            self.logger.error(f"Fetch: {e}")
            return self._prev_target

    def _do_close(self, key: str, reason: str = "closed") -> Optional[ClosedTrade]:
        if key not in self.positions:
            return None
        pos = self.positions[key]
        exit_val = pos.cur_price * pos.shares
        profit   = exit_val - pos.entry_amount
        self.available       += exit_val
        self.invested         = max(0.0, self.invested - pos.entry_amount)
        self.returned        += exit_val
        self.realized        += profit
        ct = ClosedTrade(
            key=key, market_title=pos.market_title, outcome=pos.outcome,
            slug=pos.slug, entry_price=pos.entry_price, exit_price=pos.cur_price,
            entry_amount=pos.entry_amount, exit_amount=exit_val,
            realized_pnl=profit, reason=reason,
        )
        self.closed_trades.append(ct)
        del self.positions[key]
        return ct

    def close_all(self, reason: str = "shutdown") -> List[ClosedTrade]:
        return [ct for key in list(self.positions.keys())
                for ct in [self._do_close(key, reason)] if ct]

    def sync(self) -> List[tuple]:
        events  = []
        current = self._fetch_raw()

        # ── NEW positions ────────────────────────────────────────────────
        for key, raw in current.items():
            if key not in self._prev_target:
                needed = raw['avg_price'] * raw['shares'] * self.scale
                if needed < 0.01:
                    continue
                if needed > self.available:
                    events.append(("skip",
                        f"SKIP {trunc(raw['market_title'], 35)} | "
                        f"Need ${needed:.2f}, have ${self.available:.2f}"))
                    continue
                our_shares = raw['shares'] * self.scale
                self.available -= needed
                self.invested  += needed
                pos = Position(
                    key=key, market_title=raw['market_title'],
                    outcome=raw['outcome'], slug=raw['slug'],
                    condition_id=raw['condition_id'],
                    entry_price=raw['avg_price'], cur_price=raw['cur_price'],
                    shares=our_shares, entry_amount=needed,
                    cur_value=raw['current_value'] * self.scale,
                    end_date=raw['end_date'],
                )
                self.positions[key] = pos
                tl, _ = time_left(pos.end_date)
                events.append(("new",
                    f"NEW BET  {raw['market_title']}\n"
                    f"         Side={raw['outcome']}  "
                    f"Entry=${raw['avg_price']:.4f}  "
                    f"Invested=${needed:.2f} ({our_shares:.1f} shares)\n"
                    f"         Time Left={tl}\n"
                    f"         URL={pos.url}"))
                self.logger.info(
                    f"COPIED: {raw['market_title']} | {raw['outcome']} | ${needed:.2f}")

        # ── UPDATE + AUTO-CLOSE ──────────────────────────────────────────
        for key, pos in list(self.positions.items()):
            if key in current:
                pos.cur_price = current[key]['cur_price']
                pos.cur_value = current[key]['current_value'] * self.scale
                # Refresh end_date if still missing
                if not pos.end_date and pos.slug:
                    pos.end_date = get_end_date(pos.slug, '')
                # Auto-close expired
                _, secs = time_left(pos.end_date)
                if secs == -1:
                    ct = self._do_close(key, reason="expired")
                    if ct:
                        events.append(("close",
                            f"EXPIRED  {pos.market_title}\n"
                            f"         Side={pos.outcome}  "
                            f"Profit=${ct.realized_pnl:+.2f}  ROI={ct.roi_pct:+.1f}%"))
            elif key in self._prev_target:
                # Target closed — we close too
                ct = self._do_close(key, reason="closed")
                if ct:
                    events.append(("close",
                        f"TARGET CLOSED  {pos.market_title}\n"
                        f"               Side={pos.outcome}  "
                        f"Profit=${ct.realized_pnl:+.2f}  ROI={ct.roi_pct:+.1f}%"))

        # ── MANUAL CLOSE ────────────────────────────────────────────────
        for key in list(self.manual_queue):
            if key in self.positions:
                pos = self.positions[key]
                ct  = self._do_close(key, reason="manual")
                if ct:
                    events.append(("close",
                        f"MANUAL CLOSE  {pos.market_title}\n"
                        f"              Side={pos.outcome}  "
                        f"Profit=${ct.realized_pnl:+.2f}  ROI={ct.roi_pct:+.1f}%"))
        self.manual_queue.clear()

        self._prev_target = current
        return events

    def summary(self):
        unreal = sum(p.pnl for p in self.positions.values())
        total_val = self.available + sum(p.cur_value for p in self.positions.values())
        return {
            'budget':       self.budget,
            'available':    self.available,
            'invested':     self.invested,
            'returned':     self.returned,
            'realized':     self.realized,
            'unrealized':   unreal,
            'value':        total_val,
            'n_open':       len(self.positions),
            'n_closed':     len(self.closed_trades),
            'session_start':self.session_start,
        }

# ── DASHBOARD ────────────────────────────────────────────────────────────────

W = 122

def div(n=None): return gray("─" * (n or W - 4))

def render(trader: CopyTrader, start_time: float, alerts: List[str]):
    s  = trader.summary()
    ps = sorted(trader.positions.values(), key=lambda p: p.entry_amount, reverse=True)
    ct = trader.closed_trades[-6:]
    up = time.time() - start_time
    h  = int(up // 3600)
    m  = int((up % 3600) // 60)
    sc = int(up % 60)

    pct    = s['invested'] / s['budget'] * 100 if s['budget'] else 0
    bar_w  = 25
    filled = int(pct / 100 * bar_w)
    bar    = C.YL + "█" * filled + C.GY + "░" * (bar_w - filled) + C.R
    spnl   = s['available'] - s['session_start']

    bgt = bold(f"${s['budget']:.2f}")
    avl = green(f"${s['available']:.2f}")
    inv = yel(f"${s['invested']:.2f}")
    ret = blue(f"${s['returned']:.2f}")
    val = cyan(f"${s['value']:.2f}")
    rlz = pnlc(s['realized'],   f"${s['realized']:+.2f}")
    unr = pnlc(s['unrealized'], f"${s['unrealized']:+.2f}")
    sp  = pnlc(spnl,            f"${spnl:+.2f}")

    # Column widths
    CM, CS, CE, CN, CI, CP, CR, CT = 28, 5, 8, 8, 8, 10, 7, 12

    L = []

    # Header
    L += [
        f"{C.CY}╔{'═'*(W-2)}╗{C.R}",
        (f"{C.CY}║{C.R}  {bold('POLYMARKET COPY TRADER  ─  PAPER MODE'):<46}"
         f"  uptime {gray(f'{h:02d}:{m:02d}:{sc:02d}')}"
         f"  {yel(TARGET_WALLET[:20]+'...')}"
         f"  {C.CY}║{C.R}"),
        f"{C.CY}╚{'═'*(W-2)}╝{C.R}", "",
    ]

    # Budget
    L += [
        f"  {green('▼ BUDGET')}",
        f"  {div(80)}",
        f"  Portfolio {bgt}   Available {avl}   Invested {inv} ({pct:.0f}%)  {bar}",
        f"  Realized  {rlz}   Unrealized {unr}   Returned {ret}   Session {sp}",
        "",
    ]

    # Open positions table
    close_hint = gray("type number + Enter to close")
    L += [
        f"  {C.YL}▼ OPEN POSITIONS  ({s['n_open']})  {close_hint}{C.R}",
        f"  {div()}",
        (f"  {bold('#'):>4} "
         f"{bold('Market'):<{CM}} "
         f"{bold('Side'):>{CS}} "
         f"{bold('Entry'):>{CE}} "
         f"{bold('Now'):>{CN}} "
         f"{bold('In$'):>{CI}} "
         f"{bold('P&L'):>{CP}} "
         f"{bold('ROI'):>{CR}} "
         f"{bold('Time Left'):>{CT}}  "
         f"{bold('Full Bet URL')}"),
        f"  {div()}",
    ]

    if ps:
        for i, p in enumerate(ps[:10]):
            tl_str, _ = time_left(p.end_date)
            roi_val   = p.roi_pct
            roi_str   = pnlc(roi_val, f"{roi_val:+.1f}%")
            pnl_str   = pnlc(p.pnl,  f"${p.pnl:+.2f}")
            side_c    = green(p.outcome) if p.outcome.lower() in ("yes","up") else red(p.outcome)
            num_c     = cyan(f"[{i}]")
            L.append(
                f"  {pad(num_c, 4, '>')} "
                f"{trunc(p.market_title, CM):<{CM}} "
                f"{pad(side_c,  CS, '>')} "
                f"{pad(gray(f'${p.entry_price:.3f}'), CE, '>')} "
                f"{pad(cyan(f'${p.cur_price:.3f}'),   CN, '>')} "
                f"{pad(yel(f'${p.entry_amount:.2f}'), CI, '>')} "
                f"{pad(pnl_str, CP, '>')} "
                f"{pad(roi_str, CR, '>')} "
                f"{pad(tl_str,  CT, '>')}  "
                f"{blue(p.url)}"
            )
    else:
        L.append(f"  {gray('No open positions.')}")
    L.append("")

    # Closed trades table
    L += [
        f"  {C.GR}▼ CLOSED TRADES  ({s['n_closed']} total){C.R}",
        f"  {div()}",
        (f"  {bold('Market'):<{CM}} "
         f"{bold('Side'):>{CS}} "
         f"{bold('Entry'):>{CE}} "
         f"{bold('Exit'):>{CN}} "
         f"{bold('In$'):>{CI}} "
         f"{bold('Out$'):>{CP}} "
         f"{bold('Profit'):>{CR}} "
         f"{bold('ROI'):>7}  "
         f"{bold('How'):<8}  "
         f"{bold('Full Bet URL')}"),
        f"  {div()}",
    ]

    if ct:
        for t in reversed(ct):
            roi_str = pnlc(t.roi_pct,      f"{t.roi_pct:+.1f}%")
            pnl_str = pnlc(t.realized_pnl, f"${t.realized_pnl:+.2f}")
            side_c  = green(t.outcome) if t.outcome.lower() in ("yes","up") else red(t.outcome)
            rsn_c   = (orange(t.reason) if t.reason == "manual"
                       else red(t.reason)    if t.reason == "expired"
                       else gray(t.reason))
            L.append(
                f"  {trunc(t.market_title, CM):<{CM}} "
                f"{pad(side_c,  CS, '>')} "
                f"{pad(gray(f'${t.entry_price:.3f}'), CE, '>')} "
                f"{pad(cyan(f'${t.exit_price:.3f}'),  CN, '>')} "
                f"{pad(yel(f'${t.entry_amount:.2f}'), CI, '>')} "
                f"{pad(cyan(f'${t.exit_amount:.2f}'), CP, '>')} "
                f"{pad(pnl_str, CR, '>')} "
                f"{pad(roi_str, 7,  '>')}  "
                f"{rsn_c:<8}  "
                f"{blue(t.url)}"
            )
    else:
        L.append(f"  {gray('No closed trades yet.')}")
    L.append("")

    # Alerts
    L += [f"  {orange('▼ ALERTS')}", f"  {div(80)}"]
    if alerts:
        for a in alerts[-6:]:
            c = (C.GR if "NEW" in a
                 else C.RE if any(x in a for x in ("CLOSED","EXPIRED","MANUAL"))
                 else C.YL)
            L.append(f"  {c}{a}{C.R}")
    else:
        L.append(f"  {gray('Monitoring target wallet...')}")
    L.append("")
    L.append(
        f"  {gray('Refreshes every')} {yel(str(POLL_INTERVAL)+'s')} "
        f"{gray('|')} {mg('Type position [number] + Enter to close it')} "
        f"{gray('| Ctrl+C = close all + save')}"
    )

    os.system('cls' if os.name == 'nt' else 'clear')
    print("\n".join(L))
    sys.stdout.flush()

# ── INPUT THREAD ─────────────────────────────────────────────────────────────

def input_thread_fn(trader: CopyTrader):
    """Background thread: reads input for manual close commands."""
    while True:
        try:
            line = input().strip()
            if line.isdigit():
                idx = int(line)
                ps  = list(trader.positions.values())
                if 0 <= idx < len(ps):
                    key = ps[idx].key
                    trader.manual_queue.append(key)
                    print(f"\n  {orange(f'Queued manual close for [{idx}] {trunc(ps[idx].market_title, 40)}')}")
                else:
                    print(f"\n  {red(f'Invalid: [{idx}] — valid range is 0-{len(ps)-1}')}")
        except (EOFError, KeyboardInterrupt):
            break
        except Exception:
            pass

# ── LOGGING ──────────────────────────────────────────────────────────────────

def setup_logging():
    os.makedirs("logs", exist_ok=True)
    fh = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for h in list(root.handlers):
        if not isinstance(h, logging.FileHandler):
            root.removeHandler(h)
    root.addHandler(fh)

# ── MAIN ─────────────────────────────────────────────────────────────────────

async def main():
    # Enable ANSI on Windows
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleMode(
            ctypes.windll.kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass

    setup_logging()
    logger     = logging.getLogger("main")
    start_time = time.time()
    alerts: List[str] = []

    memory = load_memory()
    trader = CopyTrader(TARGET_WALLET, memory)

    bal_s = cyan(f"${trader.available:.2f}")
    hist  = yel(str(len(trader.closed_trades)))
    print(f"\n  {bold('POLYMARKET COPY TRADER')}")
    print(f"  Balance: {bal_s}   History: {hist} closed trades\n")

    # Snapshot existing positions WITHOUT copying them
    raw = trader._fetch_raw()
    trader._prev_target = raw
    n_missing = sum(1 for v in raw.values() if not v.get('end_date'))
    print(f"  Target has {C.WH}{len(raw)}{C.R} existing positions "
          f"({gray(str(n_missing)+' missing end_date')})")
    print(f"  {green('Watching for NEW bets...')}\n")
    await asyncio.sleep(2)

    # Start background input thread for manual closes
    t = threading.Thread(target=input_thread_fn, args=(trader,), daemon=True)
    t.start()

    try:
        while True:
            events = trader.sync()

            for kind, msg in events:
                ts = datetime.now().strftime('%H:%M:%S')
                alerts.append(f"[{ts}] {msg.split(chr(10))[0]}")
                alerts = alerts[-20:]
                logger.info(msg.replace('\n', ' | '))

                # Flash full message before next render
                if kind in ("new", "close"):
                    c = C.GR if kind == "new" else C.RE
                    print(f"\n{c}{'─'*65}{C.R}")
                    for line in msg.split('\n'):
                        print(f"{c}{line}{C.R}")
                    print(f"{c}{'─'*65}{C.R}\n")
                    if kind == "new":
                        await asyncio.sleep(3)

            save_memory(trader.to_memory())
            render(trader, start_time, alerts)
            await asyncio.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print(f"\n{C.YL}Shutting down — closing all positions at current price...{C.R}")
        closed = trader.close_all(reason="shutdown")
        for ct in closed:
            pnl_s = pnlc(ct.realized_pnl, f"${ct.realized_pnl:+.2f}")
            print(f"  Closed: {trunc(ct.market_title, 45)} | {pnl_s}")
            logger.info(f"SHUTDOWN: {ct.market_title} | ${ct.realized_pnl:+.2f}")

        save_memory(trader.to_memory())
        s    = trader.summary()
        spnl = s['available'] - trader.session_start

        avl_f  = green(f"${s['available']:.2f}")
        bgt_f  = cyan(f"${s['budget']:.2f}")
        sp_f   = pnlc(spnl, f"${spnl:+.2f}")
        rlz_f  = pnlc(s['realized'], f"${s['realized']:+.2f}")
        ncl_f  = yel(str(s['n_closed']))
        nxt_f  = cyan(f"${s['available']:.2f}")

        print(f"\n{'─'*55}")
        print(bold("FINAL REPORT  (memory saved)"))
        print(f"{'─'*55}")
        print(f"Portfolio:    {bgt_f}")
        print(f"Available:    {avl_f}")
        print(f"Session P&L:  {sp_f}")
        print(f"All Realized: {rlz_f}")
        print(f"Trades closed:{ncl_f}")
        print(f"\n{green('Next run starts with')} {nxt_f}")


if __name__ == "__main__":
    asyncio.run(main())