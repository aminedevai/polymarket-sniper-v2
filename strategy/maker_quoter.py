"""
strategy/maker_quoter.py
========================
Maker Mode strategy: post-only limit orders inside the spread.
  - Pays ZERO taker fees
  - Earns daily USDC rebates from Polymarket's maker rebate pool
  - Risk: orders may not fill (no fill = no P&L, but also no loss)

Logic:
  - When no strong taker signal exists, quote both sides 1 tick inside spread
  - Cancel and requote every REQUOTE_INTERVAL_S or when market rotates
  - Never quote inside last 60s of candle (settlement risk)
  - Position size limited to MAKER_MAX_SHARES per side
"""

import asyncio
import logging
import time

from core.state import SharedState
from core.executor import PolyExecutor
from utils.metrics import MetricsTracker

logger = logging.getLogger(__name__)


class MakerQuoter:
    TICK_SIZE = 0.01                  # Minimum price increment on Polymarket
    REQUOTE_INTERVAL_S = 15.0         # Refresh quotes every 15 seconds
    MIN_SPREAD_TO_QUOTE = 0.03        # Only quote if spread >= 3 cents
    MAKER_MAX_SHARES = 20.0           # Max shares per side
    MIN_SECONDS_REMAINING = 60        # Stop quoting inside last 60s

    def __init__(
        self,
        state: SharedState,
        executor: PolyExecutor,
        metrics: MetricsTracker,
        paper_trade: bool = True,
    ):
        self.state = state
        self.executor = executor
        self.metrics = metrics
        self.paper_trade = paper_trade
        self._active_bid_id: str | None = None
        self._active_ask_id: str | None = None
        self._last_requote = 0.0
        self._quoted_token_id: str = ""

    def _should_quote(self) -> tuple[bool, str]:
        t = self.state.seconds_remaining
        if t < self.MIN_SECONDS_REMAINING:
            return False, f"Too close to expiry: {t:.0f}s"

        spread = self.state.poly_spread
        if spread < self.MIN_SPREAD_TO_QUOTE:
            return False, f"Spread too tight: {spread:.4f}"

        if not self.state.active_token_id:
            return False, "No active market"

        if not self.state.is_book_fresh():
            return False, "Orderbook stale"

        return True, "OK"

    def _compute_quotes(self) -> tuple[float, float]:
        """
        Place bids/asks 1 tick INSIDE the current best bid/ask.
        This ensures we're the maker (never cross the spread).
        """
        bid_price = round(self.state.poly_best_bid + self.TICK_SIZE, 2)
        ask_price = round(self.state.poly_best_ask - self.TICK_SIZE, 2)

        # Sanity: bid must be below ask
        if bid_price >= ask_price:
            bid_price = round(self.state.poly_best_bid, 2)
            ask_price = round(self.state.poly_best_ask, 2)

        return bid_price, ask_price

    async def _cancel_existing(self):
        if self.paper_trade:
            self._active_bid_id = None
            self._active_ask_id = None
            return

        if self._active_bid_id:
            try:
                self.executor.cancel_order(self._active_bid_id)
            except Exception as e:
                logger.debug(f"Cancel bid failed: {e}")
            self._active_bid_id = None

        if self._active_ask_id:
            try:
                self.executor.cancel_order(self._active_ask_id)
            except Exception as e:
                logger.debug(f"Cancel ask failed: {e}")
            self._active_ask_id = None

    async def _place_quotes(self):
        ok, reason = self._should_quote()
        if not ok:
            logger.debug(f"Maker skip: {reason}")
            return

        bid_price, ask_price = self._compute_quotes()
        token_id = self.state.active_token_id

        if self.paper_trade:
            logger.info(
                f"[PAPER MAKER] BID {self.MAKER_MAX_SHARES:.0f}sh @ {bid_price:.4f} | "
                f"ASK {self.MAKER_MAX_SHARES:.0f}sh @ {ask_price:.4f} | "
                f"spread={ask_price - bid_price:.4f}"
            )
            self._active_bid_id = "paper_bid"
            self._active_ask_id = "paper_ask"
            self.metrics.record_maker_quote(bid_price, ask_price, self.MAKER_MAX_SHARES)
            return

        # Live: place both sides
        try:
            bid_resp = self.executor.execute_maker_limit(
                token_id=token_id,
                price=bid_price,
                size=self.MAKER_MAX_SHARES,
                side="BUY",
            )
            self._active_bid_id = bid_resp.get("orderID")
        except Exception as e:
            logger.error(f"Maker bid failed: {e}")

        try:
            ask_resp = self.executor.execute_maker_limit(
                token_id=token_id,
                price=ask_price,
                size=self.MAKER_MAX_SHARES,
                side="SELL",
            )
            self._active_ask_id = ask_resp.get("orderID")
        except Exception as e:
            logger.error(f"Maker ask failed: {e}")

    async def run(self):
        """
        Continuous maker loop. Requotes on interval or market rotation.
        Should be run as an asyncio task alongside the taker loop.
        """
        logger.info("Maker quoter started.")
        while True:
            now = time.monotonic()
            token_rotated = self.state.active_token_id != self._quoted_token_id

            if token_rotated or (now - self._last_requote) >= self.REQUOTE_INTERVAL_S:
                if token_rotated:
                    logger.info("Maker: market rotated, cancelling old quotes.")
                    await self._cancel_existing()
                    self._quoted_token_id = self.state.active_token_id

                await self._cancel_existing()
                await self._place_quotes()
                self._last_requote = now

            await asyncio.sleep(1.0)

    async def cancel_all_quotes(self):
        """Emergency cancel — call before shutdown."""
        await self._cancel_existing()
        if not self.paper_trade:
            self.executor.cancel_all()
