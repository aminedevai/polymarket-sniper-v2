# test_detection.py
from src.market_detector import BTC5MinMarketDetector
import json


def test_detection():
    detector = BTC5MinMarketDetector()

    print("Generated timestamps:", detector.get_current_5min_timestamps())
    print("Generated slugs:", detector.generate_market_slugs())

    print("\nFetching active markets from API...")
    markets = detector.find_active_markets_via_api()
    print(f"Found {len(markets)} BTC 5-min markets")

    for m in markets:
        print(f"  - {m['slug']} | Start: {m['startTime']} | End: {m['endTime']}")

    print("\nGetting tradeable market...")
    tradeable = detector.get_tradeable_market()
    if tradeable:
        print(f"TRADEABLE: {json.dumps(tradeable, indent=2)}")
    else:
        print("No tradeable market found")


if __name__ == "__main__":
    test_detection()