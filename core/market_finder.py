"""
core/market_finder.py
=====================
Finds active 5-minute BTC markets using slug pattern:
  btc-updown-5m-{UNIX_TIMESTAMP}

Market structure (confirmed Feb 2026):
  - outcomes: ["Up", "Down"]
  - clobTokenIds: JSON string (not a list) containing array of token IDs
  - clobTokenIds[0] = UP token (YES equivalent)
  - clobTokenIds[1] = DOWN token (NO equivalent)
  - endDate: ISO string e.g. "2026-02-20T08:20:00Z"
  - conditionId: hex string for CLOB subscription
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone

import aiohttp

from core.state import SharedState

logger = logging.getLogger(__name__)

GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"


def get_5min_boundaries(n: int = 4) -> list[int]:
    """Returns current + next N-1 five-minute boundary timestamps."""
    now = int(time.time())
    current = (now // 300) * 300
    return [current + (i * 300) for i in range(n)]


def parse_clob_token_ids(raw) -> list[str]:
    """
    clobTokenIds comes back as a JSON string like:
    '["443...abc", "221...def"]'
    Parse it into a proper list.
    """
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(t) for t in parsed]
        except Exception:
            pass
    return []


def parse_end_ts(market: dict) -> int:
    """Parse endDate ISO string → Unix timestamp."""
    for field in ("endDate", "endDateIso", "end_date"):
        val = market.get(field)
        if val and isinstance(val, str) and "T" in val:
            try:
                dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
                return int(dt.timestamp())
            except Exception:
                pass
        elif isinstance(val, (int, float)) and val > 1_000_000_000:
            return int(val)
    return 0


class MarketFinder:
    POLL_INTERVAL_S = 10

    def __init__(self, state: SharedState):
        self.state = state
        self.on_rotation = asyncio.Event()

    async def _fetch_event_by_slug(
        self, session: aiohttp.ClientSession, slug: str
    ) -> dict | None:
        try:
            async with session.get(
                GAMMA_EVENTS_URL,
                params={"slug": slug},
                timeout=aiohttp.ClientTimeout(total=8),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                if isinstance(data, list) and data:
                    return data[0]
        except Exception as e:
            logger.debug(f"Slug {slug} failed: {e}")
        return None

    async def _fetch_active_market(self) -> dict | None:
        boundaries = get_5min_boundaries(n=4)
        now = int(time.time())

        async with aiohttp.ClientSession() as session:
            for ts in boundaries:
                slug = f"btc-updown-5m-{ts}"
                event = await self._fetch_event_by_slug(session, slug)
                if not event:
                    continue

                markets = event.get("markets", [])
                if not markets:
                    continue

                # Find active, non-closed market with CLOB enabled
                active = [
                    m for m in markets
                    if m.get("active", False)
                    and not m.get("closed", False)
                    and m.get("enableOrderBook", False)
                ]
                if not active:
                    continue

                market = active[0]

                # Parse token IDs from JSON string
                token_ids = parse_clob_token_ids(market.get("clobTokenIds", []))
                if len(token_ids) < 2:
                    logger.warning(
                        f"Market {slug} has {len(token_ids)} tokens — expected ≥2"
                    )
                    continue

                end_ts = parse_end_ts(market)
                seconds_remaining = max(0, end_ts - now)

                if seconds_remaining < 5:
                    logger.debug(f"{slug} expires in {seconds_remaining}s — skip")
                    continue

                # Inject parsed data back onto market dict
                market["_token_up"] = token_ids[0]    # UP = YES
                market["_token_down"] = token_ids[1]  # DOWN = NO
                market["_end_ts"] = end_ts
                market["_slug"] = slug

                logger.info(
                    f"Market: {market.get('question','')[:55]} | "
                    f"ends in {seconds_remaining}s | "
                    f"Up={token_ids[0][:10]}... "
                    f"Down={token_ids[1][:10]}..."
                )
                return market

        logger.warning(
            f"No active 5-min BTC market found. "
            f"Checked: {[f'btc-updown-5m-{ts}' for ts in boundaries]}"
        )
        return None

    async def start(self):
        while True:
            try:
                market = await self._fetch_active_market()
                if market:
                    token_up   = market["_token_up"]
                    token_down = market["_token_down"]
                    condition_id = market.get("conditionId", "")
                    end_ts = market["_end_ts"]
                    now_ts = int(time.time())
                    seconds_remaining = max(0, end_ts - now_ts)

                    self.state.seconds_remaining = seconds_remaining

                    # Store both tokens so strategy can trade either side
                    if token_up != self.state.active_token_id:
                        if self.state.rtds_chainlink_price > 0:
                            self.state.candle_open_price = self.state.rtds_chainlink_price
                        elif self.state.btc_price > 0:
                            self.state.candle_open_price = self.state.btc_price

                        self.state.active_token_id   = token_up    # UP token
                        self.state.active_token_down = token_down  # DOWN token
                        self.state.active_condition_id = condition_id
                        self.state.market_end_ts = end_ts

                        logger.info(
                            f"Market rotated → "
                            f"{market.get('question','')[:55]} | "
                            f"ends in {seconds_remaining}s | "
                            f"open={self.state.candle_open_price:.2f}"
                        )
                        self.on_rotation.set()
                        self.on_rotation.clear()

            except Exception as e:
                logger.error(f"MarketFinder error: {e}", exc_info=True)

            await asyncio.sleep(self.POLL_INTERVAL_S)

    async def fetch_once(self) -> dict | None:
        return await self._fetch_active_market()