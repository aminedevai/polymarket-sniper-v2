"""
core/chainlink_listener.py
==========================
Fetches BTC/USD price directly from Chainlink Data Streams REST API.
Polls every 2 seconds (Data Streams supports up to 10 req/s).

Feed ID: 0x00039d9e45394f473ab1f050a1b963e6b05351e52d71e507509ada0c95ed75b8
Docs: https://docs.chain.link/data-streams

Authentication: HMAC-SHA256 signed requests.
Set in .env:
  CHAINLINK_CLIENT_ID=your_client_id
  CHAINLINK_CLIENT_SECRET=your_client_secret

FREE alternative: if no API key, we fall back to polling the Chainlink
  on-chain aggregator via a public RPC — no key needed.
"""

import asyncio
import hashlib
import hmac
import logging
import os
import time

import aiohttp

from core.state import SharedState

logger = logging.getLogger(__name__)

# Chainlink Data Streams
STREAMS_URL = "https://api.chain.link/v1/reports/latest"
BTC_FEED_ID = "0x00039d9e45394f473ab1f050a1b963e6b05351e52d71e507509ada0c95ed75b8"

# Fallback: Chainlink BTC/USD aggregator on Polygon (no key needed)
# Contract: 0xc907E116054Ad103354f2D350FD2514433D57F6f
POLYGON_RPC = "https://1rpc.io/matic"
AGGREGATOR_ADDRESS = "0xc907E116054Ad103354f2D350FD2514433D57F6f"
LATESTANSWER_CALLDATA = "0x50d25bcd"  # latestAnswer() selector
DECIMALS = 8

POLL_INTERVAL_S = 2.0


class ChainlinkListener:
    def __init__(self, state: SharedState):
        self.state = state
        self.client_id = os.environ.get("CHAINLINK_CLIENT_ID", "")
        self.client_secret = os.environ.get("CHAINLINK_CLIENT_SECRET", "")
        self._use_streams = bool(self.client_id and self.client_secret)
        self._reconnect_delay = 1.0

    # ── Chainlink Data Streams (with API key) ────────────────────────────
    def _sign_request(self, method: str, path: str, body: str = "") -> dict:
        ts = str(int(time.time() * 1000))
        msg = f"{method.upper()}\n{path}\n{ts}\n{body}"
        sig = hmac.new(
            self.client_secret.encode(),
            msg.encode(),
            hashlib.sha256,
        ).hexdigest()
        return {
            "Authorization": f"Bearer {self.client_id}",
            "X-Authorization-Timestamp": ts,
            "X-Authorization-Signature-SHA256": sig,
        }

    async def _fetch_streams(self, session: aiohttp.ClientSession) -> float:
        path = f"/v1/reports/latest?feedID={BTC_FEED_ID}"
        headers = self._sign_request("GET", path)
        async with session.get(
            f"https://api.chain.link{path}",
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=5),
        ) as r:
            data = await r.json()
            # Price is in report.benchmarkPrice (18 decimals)
            price_raw = int(data["report"]["benchmarkPrice"], 16)
            return price_raw / 1e18

    # ── Fallback: on-chain aggregator via public RPC ──────────────────────
    async def _fetch_onchain(self, session: aiohttp.ClientSession) -> float:
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_call",
            "params": [
                {"to": AGGREGATOR_ADDRESS, "data": LATESTANSWER_CALLDATA},
                "latest",
            ],
            "id": 1,
        }
        async with session.post(
            POLYGON_RPC,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=5),
        ) as r:
            data = await r.json()
            result = data.get("result", "0x0")
            price_raw = int(result, 16)
            return price_raw / (10 ** DECIMALS)

    async def _fetch_price(self, session: aiohttp.ClientSession) -> float:
        if self._use_streams:
            return await self._fetch_streams(session)
        return await self._fetch_onchain(session)

    async def start(self):
        mode = "Data Streams" if self._use_streams else "on-chain RPC fallback"
        logger.info(f"Chainlink listener starting ({mode})")

        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    price = await self._fetch_price(session)
                    if price > 0:
                        now_ms = int(time.time() * 1000)
                        old = self.state.rtds_chainlink_price

                        self.state.rtds_chainlink_price = price
                        self.state.rtds_chainlink_ts_ms = now_ms
                        self.state.rtds_chainlink_recv_ts_ms = now_ms

                        # Update divergence
                        b = self.state.rtds_binance_price
                        if b > 0:
                            self.state.price_divergence_pct = (b - price) / price
                            self.state.oracle_stale_ms = 0
                            self.state.oracle_lag_ms = (
                                self.state.rtds_binance_ts_ms - now_ms
                            )
                            if not self.state.feeds_ready:
                                self.state.feeds_ready = True
                                logger.info(
                                    f"Chainlink live: ${price:.2f} | "
                                    f"Binance: ${b:.2f} | "
                                    f"div={self.state.price_divergence_pct:.4%}"
                                )

                        if old != price and old > 0:
                            logger.debug(
                                f"Chainlink: ${old:.2f} -> ${price:.2f}"
                            )
                        self._reconnect_delay = 1.0

                except Exception as e:
                    logger.warning(f"Chainlink fetch error: {e}. "
                                   f"Retry in {self._reconnect_delay:.1f}s")
                    await asyncio.sleep(self._reconnect_delay)
                    self._reconnect_delay = min(self._reconnect_delay * 2, 30)
                    continue

                await asyncio.sleep(POLL_INTERVAL_S)