"""
core/poly_book_listener.py
==========================
Subscribes to Polymarket CLOB WebSocket market channel.
Maintains real-time local orderbook state.
"""

import asyncio
import json
import logging
import time

import websockets

from core.state import SharedState

logger = logging.getLogger(__name__)

POLY_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


class PolyBookListener:
    def __init__(self, state: SharedState):
        self.state = state
        self._last_seq = -1
        self._reconnect_delay = 1.0
        self._current_token_id = ""

    def _build_subscription(self, token_id: str) -> str:
        return json.dumps({
            "assets_ids": [token_id],
            "type": "market",
            "custom_feature_enabled": True,  # Critical for best_bid_ask events
        })

    def _handle_book_snapshot(self, msg: dict):
        """Full orderbook snapshot received on initial subscribe."""
        bids = msg.get("bids", [])
        asks = msg.get("asks", [])

        best_bid = max((float(o["price"]) for o in bids), default=0.0)
        best_ask = min((float(o["price"]) for o in asks), default=1.0)

        self.state.poly_best_bid = best_bid
        self.state.poly_best_ask = best_ask
        self.state.poly_mid = (best_bid + best_ask) / 2
        self.state.poly_spread = best_ask - best_bid
        self.state.poly_book_last_update = time.monotonic()
        logger.info(f"Book snapshot: bid={best_bid:.4f} ask={best_ask:.4f} spread={best_ask - best_bid:.4f}")

    def _handle_price_change(self, msg: dict):
        """Incremental price level update."""
        price = float(msg.get("price", 0))
        side = msg.get("side", "")
        size = float(msg.get("size", 0))

        if side == "BUY":
            if size > 0:
                self.state.poly_best_bid = max(self.state.poly_best_bid, price)
        elif side == "SELL":
            if size > 0:
                self.state.poly_best_ask = min(self.state.poly_best_ask, price)

        # Recalculate mid and spread
        if self.state.poly_best_ask > self.state.poly_best_bid:
            self.state.poly_mid = (self.state.poly_best_bid + self.state.poly_best_ask) / 2
            self.state.poly_spread = self.state.poly_best_ask - self.state.poly_best_bid
            self.state.poly_book_last_update = time.monotonic()

    def _handle_best_bid_ask(self, msg: dict):
        """Best bid/ask update (requires custom_feature_enabled=true)."""
        bid = msg.get("best_bid")
        ask = msg.get("best_ask")

        if bid is not None:
            self.state.poly_best_bid = float(bid)
        if ask is not None:
            self.state.poly_best_ask = float(ask)

        if self.state.poly_best_ask > self.state.poly_best_bid:
            self.state.poly_mid = (self.state.poly_best_bid + self.state.poly_best_ask) / 2
            self.state.poly_spread = self.state.poly_best_ask - self.state.poly_best_bid
            self.state.poly_book_last_update = time.monotonic()
            logger.debug(f"Best bid/ask update: {self.state.poly_best_bid:.4f} / {self.state.poly_best_ask:.4f}")

    def _handle_last_trade(self, msg: dict):
        price = msg.get("price")
        if price is not None:
            self.state.poly_last_trade_price = float(price)

    def _process_message(self, msg: dict):
        # Sequence gap detection
        seq = msg.get("seq", -1)
        if self._last_seq >= 0 and seq > 0 and seq > self._last_seq + 1:
            logger.warning(f"Sequence gap: {self._last_seq} -> {seq}. Possible missed update!")
        if seq > 0:
            self._last_seq = seq

        event_type = msg.get("event_type", "")

        handlers = {
            "book": self._handle_book_snapshot,
            "price_change": self._handle_price_change,
            "best_bid_ask": self._handle_best_bid_ask,
            "last_trade_price": self._handle_last_trade,
            "market_resolved": lambda m: logger.info(f"Market resolved: {m}"),
            "tick_size_change": lambda m: logger.info(f"Tick size change: {m.get('tick_size')}"),
        }

        handler = handlers.get(event_type)
        if handler:
            handler(msg)
        else:
            logger.debug(f"Unknown event type: {event_type}")

    async def _run_once(self, token_id: str):
        if not token_id:
            logger.error("Cannot connect to CLOB: empty token_id")
            return

        async with websockets.connect(
                POLY_WS_URL,
                ping_interval=10,
                ping_timeout=5,
        ) as ws:
            await ws.send(self._build_subscription(token_id))
            self._reconnect_delay = 1.0
            self._last_seq = -1
            logger.info(f"Poly CLOB WS subscribed: token={token_id[:12]}...")

            async for raw in ws:
                if raw == "PONG":
                    continue
                try:
                    msgs = json.loads(raw)
                    if not isinstance(msgs, list):
                        msgs = [msgs]
                    for msg in msgs:
                        self._process_message(msg)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse CLOB message: {e}")

    async def start(self, token_id: str):
        self._current_token_id = token_id
        while True:
            try:
                # Use latest token from state if available
                tid = self.state.active_token_id or token_id
                if tid:
                    await self._run_once(tid)
                else:
                    logger.warning("No token_id available, waiting...")
                    await asyncio.sleep(1.0)
            except Exception as e:
                logger.warning(
                    f"Poly CLOB WS error: {e}. "
                    f"Retrying in {self._reconnect_delay:.1f}s..."
                )
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 30)