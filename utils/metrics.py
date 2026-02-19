"""
utils/metrics.py
================
Live terminal dashboard + CSV trade log.

Dashboard refreshes every second showing:
  - Live BTC price, Chainlink price, divergence, oracle staleness
  - Current market token + seconds remaining
  - All open/settled paper trades with outcomes
  - Running P&L, win rate, signal counts by type
  - Recent event log (last 12 lines)
"""

import csv
import os
import sys
import time
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ── ANSI color codes (work on Windows 10+ with VT100 enabled) ─────────────
R  = "\033[0m"          # reset
BOLD = "\033[1m"
DIM  = "\033[2m"

BLK = "\033[30m"
RED = "\033[31m"
GRN = "\033[32m"
YLW = "\033[33m"
BLU = "\033[34m"
MAG = "\033[35m"
CYN = "\033[36m"
WHT = "\033[37m"

B_RED = "\033[41m"
B_GRN = "\033[42m"
B_YLW = "\033[43m"
B_BLU = "\033[44m"

def _enable_windows_ansi():
    """Enable VT100 ANSI codes on Windows."""
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

_enable_windows_ansi()


@dataclass
class TradeRecord:
    ts: str
    mode: str
    signal_type: str
    side: str
    entry_price: float
    size_usdc: float
    fee: float
    confidence: float
    outcome: str = "OPEN"   # OPEN | WIN | LOSS | REJECTED
    pnl: float = 0.0
    settled_price: float = 0.0


class MetricsTracker:
    DASHBOARD_INTERVAL_S = 1.0   # redraw every second

    def __init__(self, log_dir: str = ".", state=None):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.csv_path = f"{log_dir}/trades_{ts_str}.csv"
        self.state = state                       # SharedState reference for live feed data

        self._trades: list[TradeRecord] = []
        self._fok_rejections = 0
        self._maker_quotes = 0
        self._start_time = time.monotonic()
        self._last_draw = 0.0
        self._event_log: deque = deque(maxlen=12)  # last 12 terminal events

        # CSV
        self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._csv_file)
        self._writer.writerow([
            "ts", "mode", "signal_type", "side", "entry_price",
            "size_usdc", "fee", "confidence", "outcome", "pnl"
        ])
        self._csv_file.flush()

        self._log_event(f"Trade log → {self.csv_path}", color=DIM)

    # ── Public record methods ─────────────────────────────────────────────

    def record_paper_trade(self, signal, size_usdc: float, fee: float):
        rec = TradeRecord(
            ts=datetime.now(timezone.utc).strftime("%H:%M:%S"),
            mode="PAPER",
            signal_type=signal.signal_type,
            side=signal.side,
            entry_price=signal.target_price,
            size_usdc=size_usdc,
            fee=fee,
            confidence=signal.confidence,
        )
        self._trades.append(rec)
        self._flush_csv(rec)
        self._log_event(
            f"📋 PAPER {signal.side:<3} | "
            f"${size_usdc:.2f} @ {signal.target_price:.3f} | "
            f"conf={signal.confidence:.2f} [{signal.signal_type}]",
            color=CYN,
        )
        self._draw()

    def record_live_trade(self, signal, size_usdc: float, fee: float, resp: dict):
        rec = TradeRecord(
            ts=datetime.now(timezone.utc).strftime("%H:%M:%S"),
            mode="LIVE",
            signal_type=signal.signal_type,
            side=signal.side,
            entry_price=signal.target_price,
            size_usdc=size_usdc,
            fee=fee,
            confidence=signal.confidence,
        )
        self._trades.append(rec)
        self._flush_csv(rec)
        self._log_event(
            f"🔴 LIVE  {signal.side:<3} | "
            f"${size_usdc:.2f} @ {signal.target_price:.3f} | "
            f"conf={signal.confidence:.2f} [{signal.signal_type}]",
            color=YLW,
        )
        self._draw()

    def record_fok_rejection(self, signal):
        self._fok_rejections += 1
        self._log_event(
            f"⚡ FOK REJECTED | {signal.side} [{signal.signal_type}] "
            f"conf={signal.confidence:.2f}",
            color=YLW,
        )

    def record_maker_quote(self, bid: float, ask: float, size: float):
        self._maker_quotes += 1
        self._log_event(
            f"📌 MAKER QUOTE  | bid={bid:.3f} ask={ask:.3f} size={size:.0f}sh",
            color=DIM,
        )

    def settle_trade(self, index: int, won: bool, payout: float):
        if index >= len(self._trades):
            return
        rec = self._trades[index]
        rec.outcome = "WIN" if won else "LOSS"
        rec.pnl = payout - rec.size_usdc - rec.fee
        rec.settled_price = payout / (rec.size_usdc / rec.entry_price) if rec.size_usdc > 0 else 0
        self._flush_csv(rec)

        icon = "✅" if won else "❌"
        color = GRN if won else RED
        self._log_event(
            f"{icon} SETTLED {rec.side:<3} | "
            f"PnL={rec.pnl:+.2f} | "
            f"{'WIN' if won else 'LOSS'} @ {rec.entry_price:.3f}",
            color=color,
        )
        self._draw()

    # ── Dashboard draw ────────────────────────────────────────────────────

    def _log_event(self, msg: str, color: str = ""):
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self._event_log.append(f"{DIM}{ts}{R}  {color}{msg}{R}")

    def tick(self):
        """Call from main loop every iteration to trigger periodic redraw."""
        now = time.monotonic()
        if (now - self._last_draw) >= self.DASHBOARD_INTERVAL_S:
            self._draw()
            self._last_draw = now

    def _draw(self):
        self._last_draw = time.monotonic()
        lines = self._build_dashboard()
        # Move cursor to top-left, clear screen
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()

    def _build_dashboard(self) -> list[str]:
        W = 72   # dashboard width
        lines = []

        def bar(label, content):
            lines.append(f"  {BOLD}{label:<18}{R}{content}")

        def divider(title=""):
            if title:
                pad = (W - len(title) - 4) // 2
                lines.append(f"  {DIM}{'─'*pad} {WHT}{title}{R}{DIM} {'─'*pad}{R}")
            else:
                lines.append(f"  {DIM}{'─'*W}{R}")

        uptime = int(time.monotonic() - self._start_time)
        h, m, s = uptime // 3600, (uptime % 3600) // 60, uptime % 60

        # ── Header ──────────────────────────────────────────────────────
        lines.append("")
        lines.append(
            f"  {B_BLU}{BOLD}  POLYMARKET SNIPER v2  —  PAPER TRADE MODE  {R}"
            f"  {DIM}uptime {h:02d}:{m:02d}:{s:02d}{R}"
        )
        lines.append("")

        # ── Live feed data ───────────────────────────────────────────────
        divider("LIVE FEEDS")
        if self.state:
            s_ = self.state

            # Binance price + velocity
            v = s_.btc_velocity_1s
            v_color = GRN if v > 0.001 else (RED if v < -0.001 else WHT)
            v_arrow = "▲" if v > 0 else ("▼" if v < 0 else "–")
            bar("BTC (Binance)",
                f"{WHT}${s_.btc_price:,.2f}{R}  "
                f"vel1s={v_color}{v_arrow}{abs(v):.4%}{R}  "
                f"vel5s={s_.btc_velocity_5s:+.4%}")

            # Chainlink price + staleness
            stale = s_.oracle_stale_ms / 1000
            stale_color = GRN if stale < 10 else (YLW if stale < 30 else RED)
            bar("BTC (Chainlink)",
                f"{WHT}${s_.rtds_chainlink_price:,.2f}{R}  "
                f"stale={stale_color}{stale:.1f}s{R}")

            # Divergence
            div = s_.price_divergence_pct
            div_color = GRN if abs(div) > 0.0015 else DIM
            bar("Divergence",
                f"{div_color}{div:+.4%}{R}  "
                f"({'SIGNAL ZONE' if abs(div) > 0.0015 else 'below threshold'})")

            # Orderbook
            spread = s_.poly_best_ask - s_.poly_best_bid
            bar("Orderbook",
                f"bid={GRN}{s_.poly_best_bid:.4f}{R}  "
                f"ask={RED}{s_.poly_best_ask:.4f}{R}  "
                f"spread={spread:.4f}")

            # Market
            t_rem = s_.seconds_remaining
            t_color = GRN if t_rem > 60 else (YLW if t_rem > 20 else RED)
            bar("Market",
                f"{s_.active_token_id[:16]}...  "
                f"T-{t_color}{t_rem:.0f}s{R}  "
                f"open={s_.candle_open_price:.2f}")
        else:
            lines.append(f"  {DIM}Waiting for state...{R}")

        # ── Trade summary ────────────────────────────────────────────────
        divider("TRADE SUMMARY")
        total    = len(self._trades)
        settled  = [t for t in self._trades if t.outcome in ("WIN", "LOSS")]
        wins     = [t for t in settled if t.outcome == "WIN"]
        open_tr  = [t for t in self._trades if t.outcome == "OPEN"]
        total_pnl = sum(t.pnl for t in settled)
        total_fees = sum(t.fee for t in self._trades)
        win_rate = len(wins) / len(settled) if settled else 0.0
        rej_total = total + self._fok_rejections
        rej_rate  = self._fok_rejections / rej_total if rej_total > 0 else 0.0

        pnl_color = GRN if total_pnl >= 0 else RED
        wr_color  = GRN if win_rate >= 0.62 else (YLW if win_rate >= 0.50 else RED)

        bar("Trades",
            f"total={WHT}{total}{R}  "
            f"open={YLW}{len(open_tr)}{R}  "
            f"settled={len(settled)}  "
            f"rejected={self._fok_rejections} ({rej_rate:.0%})")
        bar("P&L",
            f"{pnl_color}${total_pnl:+.2f}{R}  "
            f"fees=${total_fees:.2f}  "
            f"win_rate={wr_color}{win_rate:.1%}{R}  "
            f"({len(wins)}W / {len(settled)-len(wins)}L)")
        bar("Maker quotes", f"{self._maker_quotes}")

        # Per-signal-type breakdown
        by_type: dict[str, list] = {}
        for t in settled:
            by_type.setdefault(t.signal_type, []).append(t)
        for stype, trades in by_type.items():
            w = sum(1 for t in trades if t.outcome == "WIN")
            wr = w / len(trades)
            c = GRN if wr >= 0.62 else (YLW if wr >= 0.50 else RED)
            bar(f"  [{stype}]",
                f"n={len(trades)}  wins={w}  "
                f"wr={c}{wr:.1%}{R}  "
                f"avg_conf={sum(t.confidence for t in trades)/len(trades):.3f}")

        # ── Recent trades table ──────────────────────────────────────────
        divider("RECENT TRADES")
        recent = self._trades[-8:]  # last 8
        if not recent:
            lines.append(f"  {DIM}No trades yet.{R}")
        else:
            lines.append(
                f"  {DIM}{'TIME':<10}{'SIDE':<6}{'TYPE':<12}"
                f"{'PRICE':<8}{'SIZE':>8}{'CONF':>7}{'STATUS':<10}{'PNL':>8}{R}"
            )
            for t in reversed(recent):
                if t.outcome == "WIN":
                    status_str = f"{GRN}WIN{R}"
                    pnl_str = f"{GRN}${t.pnl:+.2f}{R}"
                elif t.outcome == "LOSS":
                    status_str = f"{RED}LOSS{R}"
                    pnl_str = f"{RED}${t.pnl:+.2f}{R}"
                else:
                    status_str = f"{YLW}OPEN{R}"
                    pnl_str = f"{DIM}—{R}"
                side_color = GRN if t.side == "YES" else RED
                lines.append(
                    f"  {t.ts:<10}"
                    f"{side_color}{t.side:<6}{R}"
                    f"{t.signal_type:<12}"
                    f"{t.entry_price:<8.3f}"
                    f"${t.size_usdc:>6.2f}"
                    f"{t.confidence:>7.2f}"
                    f"  {status_str:<10}"
                    f"{pnl_str:>8}"
                )

        # ── Event log ────────────────────────────────────────────────────
        divider("EVENT LOG")
        for evt in list(self._event_log):
            lines.append(f"  {evt}")

        divider()
        lines.append(
            f"  {DIM}CSV → {self.csv_path}   "
            f"Ctrl+C to stop{R}"
        )
        lines.append("")
        return lines

    # ── CSV ───────────────────────────────────────────────────────────────

    def _flush_csv(self, rec: TradeRecord):
        self._writer.writerow([
            rec.ts, rec.mode, rec.signal_type, rec.side,
            f"{rec.entry_price:.4f}", f"{rec.size_usdc:.2f}",
            f"{rec.fee:.4f}", f"{rec.confidence:.4f}",
            rec.outcome, f"{rec.pnl:.4f}",
        ])
        self._csv_file.flush()

    def _maybe_print_summary(self):
        pass  # replaced by live dashboard

    def print_summary(self):
        self._draw()

    def close(self):
        self._draw()
        self._csv_file.close()