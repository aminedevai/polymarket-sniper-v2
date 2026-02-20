"""
core/state.py
=============
Single shared state object passed to every module.
All fields are written by one owner and read by many — no locks needed
because Python's GIL makes float/int assignments atomic in practice,
and we're single-process async (no threads).
"""

from dataclasses import dataclass, field
from collections import deque
import time


@dataclass
class SharedState:
    # ── Binance feed (written by BinanceListener) ──────────────────────────
    btc_price: float = 0.0
    btc_velocity_1s: float = 0.0        # EMA-smoothed 1s velocity
    btc_velocity_5s: float = 0.0        # raw 5s window velocity
    btc_last_update: float = 0.0        # monotonic time of last update
    btc_price_history: deque = field(   # (source_ts_ms, price)
        default_factory=lambda: deque(maxlen=300)
    )

    # ── Polymarket CLOB orderbook (written by PolyBookListener) ───────────
    poly_best_bid: float = 0.0
    poly_best_ask: float = 1.0
    poly_mid: float = 0.5
    poly_spread: float = 1.0
    poly_last_trade_price: float = 0.5
    poly_book_last_update: float = 0.0  # monotonic

    # ── RTDS Binance feed (written by RTDSListener - binance topic) ────────
    rtds_binance_price: float = 0.0
    rtds_binance_ts_ms: int = 0         # source timestamp ms
    rtds_binance_recv_ts_ms: int = 0    # polymarket server sent ts

    # ── RTDS Chainlink feed (written by RTDSListener - chainlink topic) ───
    rtds_chainlink_price: float = 0.0
    rtds_chainlink_ts_ms: int = 0       # source timestamp ms (oracle round)
    rtds_chainlink_recv_ts_ms: int = 0  # polymarket server sent ts

    # ── Computed divergence metrics (written by RTDSListener) ─────────────
    price_divergence_pct: float = 0.0   # (binance - chainlink) / chainlink
    oracle_lag_ms: float = 0.0          # binance_ts - chainlink_ts
    oracle_stale_ms: float = 0.0        # ms since chainlink last updated

    # ── Market metadata (written by MarketFinder) ─────────────────────────
    active_token_id: str = ""           # UP token (YES equivalent)
    active_token_down: str = ""         # DOWN token (NO equivalent)
    active_condition_id: str = ""
    market_end_ts: int = 0
    seconds_remaining: float = 300.0
    candle_open_price: float = 0.0      # Chainlink price at market start

    # ── Position tracking (written by RiskManager / executor) ─────────────
    open_position_side: str = ""        # "YES", "NO", or ""
    open_position_size: float = 0.0     # USDC spent
    open_position_price: float = 0.0    # entry price
    current_balance: float = 0.0

    # ── Runtime flags ──────────────────────────────────────────────────────
    feeds_ready: bool = False           # True once both binance+chainlink live

    def is_binance_fresh(self, max_age_s: float = 2.0) -> bool:
        return (time.monotonic() - self.btc_last_update) < max_age_s

    def is_book_fresh(self, max_age_s: float = 5.0) -> bool:
        return (time.monotonic() - self.poly_book_last_update) < max_age_s

    def is_chainlink_fresh(self, max_age_ms: float = 90_000) -> bool:
        import time as _t
        return (_t.time() * 1000 - self.rtds_chainlink_ts_ms) < max_age_ms