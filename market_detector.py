# src/market_detector.py
from datetime import datetime, timezone, timedelta
import requests
import logging

logger = logging.getLogger(__name__)


class BTC5MinMarketDetector:
    """
    Detects active BTC 5-minute up/down markets on Polymarket.
    """

    GAMMA_API_URL = "https://gamma-api.polymarket.com"

    def __init__(self):
        self.session = requests.Session()

    def get_current_5min_timestamps(self, window_range=3):
        """
        Generate timestamps for current and nearby 5-minute windows.

        Args:
            window_range: How many windows before/after current to generate

        Returns:
            List of Unix timestamps
        """
        now = datetime.now(timezone.utc)

        # Round down to nearest 5 minutes
        minutes = (now.minute // 5) * 5
        current_window = now.replace(minute=minutes, second=0, microsecond=0)

        timestamps = []
        for offset in range(-window_range, window_range + 1):
            window_time = current_window + timedelta(minutes=offset * 5)
            timestamps.append(int(window_time.timestamp()))

        return timestamps

    def generate_market_slugs(self):
        """Generate possible market slugs for BTC 5-min markets."""
        timestamps = self.get_current_5min_timestamps()
        return [f"btc-updown-5m-{ts}" for ts in timestamps]

    def find_active_markets_via_api(self):
        """
        Query Gamma API for active BTC 5-minute markets.
        More reliable than guessing timestamps.
        """
        try:
            url = f"{self.GAMMA_API_URL}/events"
            params = {
                "active": "true",
                "closed": "false",
                "archived": "false",
                "limit": "100",
                "order": "startTime",
                "ascending": "true"
            }

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            events = response.json()

            btc_markets = []
            for event in events:
                slug = event.get("slug", "")
                if "btc-updown-5m-" in slug:
                    btc_markets.append({
                        "slug": slug,
                        "title": event.get("title"),
                        "conditionId": event.get("conditionId"),
                        "marketId": event.get("markets", [{}])[0].get("id"),
                        "startTime": event.get("startTime"),
                        "endTime": event.get("endTime"),
                        "timestamp": self._extract_timestamp(slug)
                    })

            return btc_markets

        except Exception as e:
            logger.error(f"Error fetching markets from API: {e}")
            return []

    def _extract_timestamp(self, slug):
        """Extract timestamp from slug like 'btc-updown-5m-1771511700'."""
        try:
            return int(slug.split("-")[-1])
        except (ValueError, IndexError):
            return None

    def get_tradeable_market(self):
        """
        Get the currently tradeable BTC 5-min market.
        Returns the market that is currently active and open for trading.
        """
        markets = self.find_active_markets_via_api()

        now = datetime.now(timezone.utc)
        now_ts = int(now.timestamp())

        for market in markets:
            start_time = market.get("startTime", 0) // 1000  # Convert ms to s
            end_time = market.get("endTime", 0) // 1000

            # Market is tradeable if we're within its time window
            if start_time <= now_ts <= end_time:
                logger.info(f"Found tradeable market: {market['slug']}")
                return market

        logger.warning("No tradeable BTC 5-min market found")
        return None


# Convenience function for quick usage
def get_btc_5min_market():
    """Quick function to get current BTC 5-min market."""
    detector = BTC5MinMarketDetector()
    return detector.get_tradeable_market()