"""
core/rtds_listener.py
=====================
Subscribes to Polymarket RTDS WebSocket.
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
        self._running = False

    def _build_subscription(self) -> str:
        # Subscribe to both feeds
        sub = {
            "action": "subscribe",
            "subscriptions": [
                {
                    "topic": "crypto_prices",
                    "type": "update",
                    "filters": json.dumps({"symbol": self.binance_symbol})
                },
                {
                    "topic": "crypto_prices_chainlink",
                    "type": "update",
                    "filters": json.dumps({"symbol": self.chainlink_symbol})
                },
            ]
        }
        return json.dumps(sub)

    def _update_divergence(self):
        b = self.state.rtds_binance_price
        cl = self.state.rtds_chainlink_price
        if b > 0 and cl > 0:
            self.state.price_divergence_pct = (b - cl) / cl
            self.state.oracle_lag_ms = float(
                self.state.rtds_binance_ts_ms - self.state.rtds_chainlink_ts_ms
            )
            self.state.oracle_stale_ms = time.time() * 1000 - self.state.rtds_chainlink_ts_ms

            if not self.state.feeds_ready:
                self.state.feeds_ready = True
                logger.info(
                    f"✓ RTDS feeds ready | "
                    f"Binance=${b:.2f} Chainlink=${cl:.2f} | "
                    f"div={self.state.price_divergence_pct:.4%}"
                )

    def _handle_message(self, raw_msg: str):
        """Parse RTDS message."""
        if not raw_msg or raw_msg.strip() == "":
            logger.debug("Empty message received")
            return

        if raw_msg == "PONG":
            logger.debug("PONG received")
            return

        try:
            msg = json.loads(raw_msg)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON: {e} | msg={raw_msg[:100]}")
            return

        topic = msg.get("topic", "")
        payload = msg.get("payload", {})
        msg_type = msg.get("type", "")

        # Skip subscription confirmation messages
        if msg_type == "subscribe":
            logger.debug(f"Subscription confirmed: {topic}")
            return

        if not payload or not isinstance(payload, dict):
            return

        recv_ts = int(msg.get("timestamp", time.time() * 1000))

        # Handle crypto_prices (Binance)
        if topic == "crypto_prices":
            try:
                # Direct value format
                price = float(payload.get("value", 0))
                source_ts = int(payload.get("timestamp", recv_ts))

                if price > 0:
                    self.state.rtds_binance_price = price
                    self.state.rtds_binance_ts_ms = source_ts
                    logger.debug(f"Binance: ${price:.2f}")
                    self._update_divergence()
            except Exception as e:
                logger.debug(f"Binance parse error: {e}")

        # Handle crypto_prices_chainlink
        elif topic == "crypto_prices_chainlink":
            try:
                data = payload.get("data", [])

                if data and isinstance(data, list):
                    # Get latest from batch array
                    latest = max(data, key=lambda x: x.get("timestamp", 0))
                    price = float(latest.get("value", 0))
                    source_ts = int(latest.get("timestamp", recv_ts))

                    if price > 0:
                        old = self.state.rtds_chainlink_price
                        self.state.rtds_chainlink_price = price
                        self.state.rtds_chainlink_ts_ms = source_ts

                        if abs(old - price) > 0.1 or old == 0:
                            lag_ms = recv_ts - source_ts
                            logger.info(
                                f"Chainlink: ${price:.2f} "
                                f"({len(data)} pts, {lag_ms}ms lag)"
                            )
                        self._update_divergence()
                else:
                    # Single value fallback
                    val = payload.get("value")
                    if val is not None:
                        price = float(val)
                        if price > 0:
                            self.state.rtds_chainlink_price = price
                            self.state.rtds_chainlink_ts_ms = recv_ts
                            logger.info(f"Chainlink(single): ${price:.2f}")
                            self._update_divergence()
            except Exception as e:
                logger.debug(f"Chainlink parse error: {e}")

    async def _ping_loop(self, ws):
        """Send PING every 5 seconds."""
        while self._running:
            try:
                await ws.send("PING")
                await asyncio.sleep(5.0)
            except:
                break

    async def _receive_loop(self, ws):
        """Receive messages."""
        while self._running:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
                self._handle_message(msg)
            except asyncio.TimeoutError:
                logger.warning("RTDS timeout - no data for 30s")
                break
            except websockets.exceptions.ConnectionClosed:
                break
            except Exception as e:
                logger.error(f"RTDS receive error: {e}")
                break

    async def _run_once(self):
        self._running = True

        try:
            logger.info("Connecting to RTDS...")

            async with websockets.connect(
                    RTDS_URL,
                    ping_interval=None,
                    ping_timeout=None,
                    close_timeout=10,
            ) as ws:
                logger.info("RTDS connected")

                await ws.send(self._build_subscription())
                logger.info(f"Subscribed to {self.binance_symbol} + {self.chainlink_symbol}")

                self._reconnect_delay = 1.0

                ping_task = asyncio.create_task(self._ping_loop(ws))
                receive_task = asyncio.create_task(self._receive_loop(ws))

                done, pending = await asyncio.wait(
                    [ping_task, receive_task],
                    return_when=asyncio.FIRST_COMPLETED
                )

                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        except websockets.exceptions.InvalidStatusCode as e:
            logger.error(f"RTDS rejected: HTTP {e.status_code}")
            if e.status_code == 403:
                logger.error("Geographic restriction suspected")
            self._reconnect_delay = min(self._reconnect_delay * 2, 60)
        except Exception as e:
            logger.error(f"RTDS error: {type(e).__name__}: {e}")
        finally:
            self._running = False

    async def start(self):
        while True:
            try:
                await self._run_once()
            except Exception as e:
                logger.critical(f"RTDS fatal: {e}")

            if not self._running:
                logger.warning(f"RTDS reconnect in {self._reconnect_delay:.1f}s...")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 30)