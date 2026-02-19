"""
core/binance_listener.py
========================
Connects to Binance Futures aggTrade WebSocket.
Computes EMA-smoothed 1s and raw 5s velocity.
Writes to SharedState. Reconnects with exponential backoff.
"""

import asyncio
import json
import logging
import time

import websockets

from core.state import SharedState

logger = logging.getLogger(__name__)


class BinanceListener:
    URL = "wss://fstream.binance.com/ws/btcusdt@aggTrade"
    EMA_ALPHA = 0.3
    VELOCITY_WINDOW_1S_MS = 1_000
    VELOCITY_WINDOW_5S_MS = 5_000

    def __init__(self, state: SharedState):
        self.state = state
        self._ema_velocity = 0.0
        self._reconnect_delay = 1.0

    def _compute_velocity(self, new_price: float, now_ms: float):
        history = self.state.btc_price_history

        # 1-second EMA velocity
        cutoff_1s = now_ms - self.VELOCITY_WINDOW_1S_MS
        prices_1s = [p for t, p in history if t >= cutoff_1s]
        if prices_1s:
            raw_v1 = (new_price - prices_1s[0]) / prices_1s[0]
            self._ema_velocity = (
                self.EMA_ALPHA * raw_v1 + (1 - self.EMA_ALPHA) * self._ema_velocity
            )
            self.state.btc_velocity_1s = self._ema_velocity

        # 5-second raw velocity
        cutoff_5s = now_ms - self.VELOCITY_WINDOW_5S_MS
        prices_5s = [p for t, p in history if t >= cutoff_5s]
        if prices_5s:
            self.state.btc_velocity_5s = (new_price - prices_5s[0]) / prices_5s[0]

    async def _run_once(self):
        async with websockets.connect(
            self.URL,
            ping_interval=20,
            ping_timeout=10,
        ) as ws:
            self._reconnect_delay = 1.0
            logger.info("Binance WS connected.")
            async for raw in ws:
                data = json.loads(raw)
                new_price = float(data["p"])
                now_ms = float(data["T"])  # use exchange timestamp

                self.state.btc_price_history.append((now_ms, new_price))
                self._compute_velocity(new_price, now_ms)

                self.state.btc_price = new_price
                self.state.btc_last_update = time.monotonic()

    async def start(self):
        while True:
            try:
                await self._run_once()
            except Exception as e:
                logger.warning(
                    f"Binance WS error: {e}. "
                    f"Retrying in {self._reconnect_delay:.1f}s..."
                )
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 30)
