"""
core/rtds_listener.py
=====================
Connects to Polymarket RTDS WebSocket.

Confirmed message format (Feb 2026):
  - Both topics send: {"payload": {"data": [...], "symbol": "..."}, "timestamp": ..., "topic": "...", "type": "..."}
  - payload.data = [{"timestamp": ms, "value": float}, ...]
  - topic = "crypto_prices" or "crypto_prices_chainlink"
  - Empty string sent as first message (keepalive) — skip it
"""

import asyncio
import json
import logging
import time

import websockets

from core.state import SharedState

logger = logging.getLogger(__name__)

RTDS_URL = "wss://ws-live-data.polymarket.com"


class RTDSListener:
    def __init__(self, state: SharedState, symbol: str = "btc"):
        self.state = state
        self.binance_symbol = f"{symbol}usdt"
        self.chainlink_symbol = f"{symbol}/usd"
        self._reconnect_delay = 1.0

    def _build_subscription(self) -> str:
        return json.dumps({
            "action": "subscribe",
            "subscriptions": [
                {
                    "topic": "crypto_prices",
                    "type": "update",
                    "filters": self.binance_symbol,
                },
                {
                    "topic": "crypto_prices_chainlink",
                    "type": "*",
                    "filters": json.dumps({"symbol": self.chainlink_symbol}),
                },
            ]
        })

    def _extract_latest(self, payload: dict) -> tuple[float, int]:
        """Extract latest price and timestamp from bulk data array or single value."""
        data = payload.get("data", [])
        if data and isinstance(data, list):
            latest = max(data, key=lambda x: x.get("timestamp", 0))
            return float(latest.get("value", 0)), int(latest.get("timestamp", 0))
        # Fallback: single value format
        value = payload.get("value", 0)
        ts = payload.get("timestamp", int(time.time() * 1000))
        return float(value), int(ts)

    def _update_divergence(self):
        b = self.state.rtds_binance_price
        cl = self.state.rtds_chainlink_price
        if b > 0 and cl > 0:
            self.state.price_divergence_pct = (b - cl) / cl
            self.state.oracle_lag_ms = float(
                self.state.rtds_binance_ts_ms - self.state.rtds_chainlink_ts_ms
            )
            self.state.oracle_stale_ms = (
                time.time() * 1000 - self.state.rtds_chainlink_ts_ms
            )
            if not self.state.feeds_ready:
                self.state.feeds_ready = True
                logger.info(
                    f"Both RTDS feeds live | "
                    f"Binance={b:.2f} Chainlink={cl:.2f} "
                    f"div={self.state.price_divergence_pct:.4%}"
                )

    def _handle_message(self, msg: dict):
        topic = msg.get("topic", "")
        payload = msg.get("payload", {})
        recv_ts = int(msg.get("timestamp", time.time() * 1000))

        if not payload or not isinstance(payload, dict):
            return

        price, source_ts = self._extract_latest(payload)
        if price <= 0:
            return

        if source_ts == 0:
            source_ts = recv_ts

        if topic == "crypto_prices":
            self.state.rtds_binance_price = price
            self.state.rtds_binance_ts_ms = source_ts
            self.state.rtds_binance_recv_ts_ms = recv_ts
            logger.debug(f"Binance RTDS: ${price:.2f}")

        elif topic == "crypto_prices_chainlink":
            old = self.state.rtds_chainlink_price
            self.state.rtds_chainlink_price = price
            self.state.rtds_chainlink_ts_ms = source_ts
            self.state.rtds_chainlink_recv_ts_ms = recv_ts
            if old != price:
                logger.debug(f"Chainlink: ${old:.2f} -> ${price:.2f}")

        self._update_divergence()

    async def _run_once(self):
        async with websockets.connect(
            RTDS_URL,
            ping_interval=10,
            ping_timeout=5,
        ) as ws:
            self._reconnect_delay = 1.0
            await ws.send(self._build_subscription())
            logger.info(
                f"RTDS connected | "
                f"Binance: {self.binance_symbol} | "
                f"Chainlink: {self.chainlink_symbol}"
            )
            async for raw in ws:
                try:
                    if not raw:
                        continue
                    msg = json.loads(raw)
                    self._handle_message(msg)
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.debug(f"RTDS parse error: {e} | raw={str(raw)[:100]}")

    async def start(self):
        while True:
            try:
                await self._run_once()
            except Exception as e:
                logger.warning(
                    f"RTDS disconnected: {e}. "
                    f"Retrying in {self._reconnect_delay:.1f}s..."
                )
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 30)