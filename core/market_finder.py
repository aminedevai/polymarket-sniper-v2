"""
core/market_finder.py - MANUAL TOKEN OVERRIDE VERSION
"""

import asyncio
import logging
import time
import os

import aiohttp

from core.state import SharedState

logger = logging.getLogger(__name__)

GAMMA_URL = "https://gamma-api.polymarket.com/markets"


class MarketFinder:
    POLL_INTERVAL_S = 15

    def __init__(self, state: SharedState):
        self.state = state
        self.on_rotation = asyncio.Event()

        # Get manual token from environment
        self.manual_token = os.environ.get("POLY_TOKEN_ID", "")
        self.manual_condition = os.environ.get("POLY_CONDITION_ID", "")
        self.manual_question = os.environ.get("POLY_QUESTION", "Manual 5-min BTC")

        # Track rotation timing for 5-min markets
        self._current_token = ""
        self._market_start_time = 0

    def _calculate_5min_rotation(self):
        """
        Calculate when the next 5-minute rotation should happen.
        Returns seconds until next rotation.
        """
        now = int(time.time())
        # 5-minute blocks: :00, :05, :10, :15, :20, :25, :30, :35, :40, :45, :50, :55
        next_rotation = ((now // 300) + 1) * 300
        seconds_until = next_rotation - now
        return seconds_until, next_rotation

    async def _fetch_fallback_market(self) -> dict | None:
        """
        Fetch any available BTC market as fallback.
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                        GAMMA_URL,
                        params={"active": "true", "closed": "false", "_c": "bitcoin", "limit": 10},
                        timeout=10
                ) as resp:
                    if resp.status == 200:
                        markets = await resp.json()
                        for m in markets:
                            if m.get("active") and not m.get("closed"):
                                tokens = m.get("clobTokenIds", [])
                                if tokens:
                                    return m
        except Exception as e:
            logger.debug(f"Fallback fetch failed: {e}")
        return None

    async def start(self):
        """Main loop with manual token support"""

        # If manual token is set, use it exclusively
        if self.manual_token:
            logger.info(f"Using MANUAL token: {self.manual_token[:20]}...")
            self._current_token = self.manual_token

            self.state.active_token_id = self.manual_token
            self.state.active_condition_id = self.manual_condition or "manual"
            self.state.seconds_remaining = 300  # 5 minutes default
            self.state.market_end_ts = int(time.time()) + 300

            # Set open price if available
            if self.state.rtds_chainlink_price > 0:
                self.state.candle_open_price = self.state.rtds_chainlink_price
            elif self.state.btc_price > 0:
                self.state.candle_open_price = self.state.btc_price

            logger.info(
                f"MANUAL MARKET SET: {self.manual_question} | "
                f"Token: {self.manual_token[:16]}... | "
                f"Open: ${self.state.candle_open_price:.2f}"
            )
            self.on_rotation.set()
            self.on_rotation.clear()

            # Keep alive with countdown
            while True:
                now = int(time.time())
                elapsed = now - self._market_start_time if self._market_start_time > 0 else 0
                remaining = max(0, 300 - (elapsed % 300))
                self.state.seconds_remaining = remaining

                # Log countdown every minute
                if remaining % 60 == 0 and remaining > 0:
                    logger.info(f"Manual market T-{remaining}s remaining")

                await asyncio.sleep(self.POLL_INTERVAL_S)

            return  # Never reached but for clarity

        # Otherwise try to auto-discover (will likely fail for 5-min markets)
        logger.warning("No manual token set. Attempting auto-discovery...")
        logger.warning("5-minute markets are geo-restricted. Set POLY_TOKEN_ID env var.")

        while True:
            try:
                # Try to find any market (fallback to long-term BTC)
                market = await self._fetch_fallback_market()

                if market:
                    token_ids = market.get("clobTokenIds", [])
                    if token_ids:
                        token_id = token_ids[0]

                        if token_id != self._current_token:
                            self._current_token = token_id

                            end_ts = 0
                            val = market.get("endDateIso") or market.get("endDate")
                            if val:
                                try:
                                    from datetime import datetime
                                    dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
                                    end_ts = int(dt.timestamp())
                                except:
                                    pass

                            now_ts = int(time.time())
                            seconds_remaining = max(0, end_ts - now_ts) if end_ts > 0 else 300

                            self.state.active_token_id = token_id
                            self.state.active_condition_id = market.get("conditionId", "")
                            self.state.seconds_remaining = seconds_remaining
                            self.state.market_end_ts = end_ts

                            if self.state.rtds_chainlink_price > 0:
                                self.state.candle_open_price = self.state.rtds_chainlink_price
                            elif self.state.btc_price > 0:
                                self.state.candle_open_price = self.state.btc_price

                            logger.info(
                                f"Auto-selected: {market.get('question', '')[:40]}... | "
                                f"T-{seconds_remaining}s"
                            )
                            self.on_rotation.set()
                            self.on_rotation.clear()
                else:
                    logger.warning("No markets found via API")

            except Exception as e:
                logger.error(f"MarketFinder error: {e}", exc_info=True)

            await asyncio.sleep(self.POLL_INTERVAL_S)

    async def fetch_once(self) -> dict | None:
        if self.manual_token:
            return {
                "clobTokenIds": [self.manual_token],
                "conditionId": self.manual_condition or "manual",
                "question": self.manual_question
            }
        return await self._fetch_fallback_market()