"""
strategy/signal_engine.py
==========================
Combines four signal sources:
  1. Binance price velocity (1s EMA + 5s raw)
  2. Chainlink oracle divergence (Binance - Chainlink gap)
  3. Oracle staleness (how long since last Chainlink update)
  4. Time remaining in interval (signal decay filter)

Entry condition (CORRECTED from v1.0):
  - Target p < 0.25 or p > 0.75 only (low-fee zones)
  - Trade window: 30s – 240s remaining
"""

import time
import logging
from dataclasses import dataclass

from core.state import SharedState
from risk.fee_calculator import FeeCalculator

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    should_trade: bool
    side: str           # "YES" or "NO"
    target_price: float # entry price (ask for YES, 1-bid for NO)
    confidence: float   # 0.0 – 1.0
    signal_type: str    # "VELOCITY", "DIVERGENCE", "COMBINED"
    reason: str


class SignalEngine:
    # Velocity thresholds
    VELOCITY_THRESHOLD_1S = 0.0015     # 0.15%/s EMA
    VELOCITY_THRESHOLD_5S = 0.0030     # 0.30% over 5s

    # Divergence thresholds
    MIN_DIVERGENCE_PCT = 0.0015        # 0.15% gap Binance vs Chainlink
    MAX_DIVERGENCE_PCT = 0.008         # > 0.8% → data issue, skip

    # Oracle staleness
    MIN_STALE_MS = 10_000              # Chainlink must be ≥ 10s stale for Type 1
    MAX_STALE_MS = 55_000              # > 55s suggests heartbeat failure

    # Time window (seconds remaining in the 5-min candle)
    MIN_SECONDS_REMAINING = 30
    MAX_SECONDS_REMAINING = 240

    # Fee filter
    MAX_FEE_RATE = 0.012               # 1.2% max effective fee rate

    # Data freshness
    MAX_BINANCE_AGE_S = 2.0
    MAX_BOOK_AGE_S = 5.0

    def evaluate(self, state: SharedState) -> TradeSignal:
        no_signal = lambda r: TradeSignal(False, "", 0.0, 0.0, "", r)

        # ── Feed freshness ──────────────────────────────────────────────────
        if not state.is_binance_fresh(self.MAX_BINANCE_AGE_S):
            return no_signal("Binance feed stale")
        if not state.is_book_fresh(self.MAX_BOOK_AGE_S):
            return no_signal("Poly orderbook stale")
        if not state.feeds_ready:
            return no_signal("RTDS feeds not ready yet")

        # ── Time filter ─────────────────────────────────────────────────────
        t = state.seconds_remaining
        if not (self.MIN_SECONDS_REMAINING <= t <= self.MAX_SECONDS_REMAINING):
            return no_signal(f"Outside time window: {t:.0f}s")

        v1s = state.btc_velocity_1s
        v5s = state.btc_velocity_5s
        div = state.price_divergence_pct
        stale_ms = state.oracle_stale_ms

        # ── TYPE 1: Velocity + Stale Oracle ────────────────────────────────
        type1 = (
            abs(v1s) >= self.VELOCITY_THRESHOLD_1S
            and self.MIN_STALE_MS <= stale_ms <= self.MAX_STALE_MS
        )

        # ── TYPE 2: Chainlink Divergence ────────────────────────────────────
        type2 = (
            self.MIN_DIVERGENCE_PCT <= abs(div) <= self.MAX_DIVERGENCE_PCT
        )

        if not type1 and not type2:
            return no_signal(
                f"No signal | v1s={v1s:.4%} div={div:.4%} stale={stale_ms:.0f}ms"
            )

        # ── Determine direction ─────────────────────────────────────────────
        # Type 1: follow Binance velocity
        vel_bullish = v1s > 0
        vel_bearish = v1s < 0

        # Type 2: Binance > Chainlink → Chainlink will correct UP → YES
        div_bullish = div > 0
        div_bearish = div < 0

        if type1 and type2:
            if vel_bullish and div_bullish:
                side = "YES"
                signal_type = "COMBINED"
            elif vel_bearish and div_bearish:
                side = "NO"
                signal_type = "COMBINED"
            else:
                return no_signal(
                    f"Signal conflict: vel={'UP' if vel_bullish else 'DN'} "
                    f"div={'UP' if div_bullish else 'DN'}"
                )
        elif type1:
            side = "YES" if vel_bullish else "NO"
            signal_type = "VELOCITY"
        else:
            side = "YES" if div_bullish else "NO"
            signal_type = "DIVERGENCE"

        # ── Get target price from orderbook ─────────────────────────────────
        if side == "YES":
            target_price = state.poly_best_ask    # we buy YES at ask
        else:
            target_price = 1.0 - state.poly_best_bid  # NO price = 1 - YES bid

        if target_price <= 0 or target_price >= 1:
            return no_signal(f"Invalid target price: {target_price:.4f}")

        # ── Fee zone check (CORRECTED — target extremes only) ───────────────
        if not FeeCalculator.is_tradeable_zone(target_price, self.MAX_FEE_RATE):
            eff = FeeCalculator.effective_fee_rate(target_price)
            return no_signal(
                f"Fee too high at p={target_price:.2f}: {eff:.2%} "
                f"(target p<0.25 or p>0.75)"
            )

        # ── Sanity: required win prob check ─────────────────────────────────
        required_prob = FeeCalculator.required_win_probability(target_price)
        if required_prob > 0.93:
            return no_signal(f"Required win prob too high: {required_prob:.2%}")

        # ── Confidence scoring ──────────────────────────────────────────────
        v_score = min(1.0, abs(v1s) / (self.VELOCITY_THRESHOLD_1S * 4))
        v5_score = min(1.0, abs(v5s) / (self.VELOCITY_THRESHOLD_5S * 3))
        d_score = min(1.0, abs(div) / (self.MIN_DIVERGENCE_PCT * 5))
        stale_score = min(1.0, stale_ms / 30_000)
        # Time score: peaks at 150s (2.5 min remaining)
        time_score = max(0.0, 1.0 - abs(t - 150) / 150)

        if signal_type == "COMBINED":
            confidence = (
                v_score * 0.25
                + v5_score * 0.10
                + d_score * 0.30
                + stale_score * 0.15
                + time_score * 0.20
            )
        elif signal_type == "VELOCITY":
            confidence = (
                v_score * 0.40
                + v5_score * 0.15
                + stale_score * 0.25
                + time_score * 0.20
            )
        else:  # DIVERGENCE
            confidence = d_score * 0.65 + time_score * 0.35

        confidence = min(1.0, confidence)

        reason = (
            f"[{signal_type}] side={side} conf={confidence:.3f} | "
            f"p={target_price:.3f} | "
            f"v1s={v1s:.4%} v5s={v5s:.4%} | "
            f"div={div:.4%} stale={stale_ms:.0f}ms | "
            f"t={t:.0f}s"
        )

        return TradeSignal(
            should_trade=True,
            side=side,
            target_price=target_price,
            confidence=confidence,
            signal_type=signal_type,
            reason=reason,
        )
