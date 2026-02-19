#!/usr/bin/env python3
"""
test_5min_discovery.py - Test finding 5-min markets by timestamp
"""
import asyncio
import aiohttp
import time
import datetime


async def find_by_timestamp(target_ts: int):
    """Try to find market for specific timestamp"""
    slug = f"btc-updown-5m-{target_ts}"
    print(f"Searching for: {slug}")
    print(f"Target time: {datetime.datetime.fromtimestamp(target_ts, tz=datetime.timezone.utc)}")

    # Try Gamma API with various search patterns
    urls = [
        f"https://gamma-api.polymarket.com/markets?active=true&closed=false&_c={slug}&limit=10",
        f"https://gamma-api.polymarket.com/markets?active=true&closed=false&_c=btc-updown-5m&limit=20",
        f"https://gamma-api.polymarket.com/events?active=true&closed=false&_c={slug}&limit=10",
    ]

    async with aiohttp.ClientSession() as session:
        for url in urls:
            print(f"\nTrying: {url[:80]}...")
            try:
                async with session.get(url, timeout=10) as resp:
                    print(f"Status: {resp.status}")
                    if resp.status == 200:
                        data = await resp.json()
                        print(f"Results: {len(data)}")

                        # Print first few items
                        for item in data[:3]:
                            if isinstance(item, dict):
                                print(f"  - {item.get('question', item.get('title', 'N/A'))[:60]}")
                                print(f"    Slug: {item.get('slug', 'N/A')}")
                                tokens = item.get("clobTokenIds", [])
                                if tokens:
                                    print(f"    Token: {tokens[0][:20]}...")
            except Exception as e:
                print(f"Error: {e}")


async def main():
    # Calculate current and next 5-min timestamps
    now = int(time.time())
    current_5min = (now // 300) * 300
    next_5min = current_5min + 300

    print(f"Current time: {datetime.datetime.now(datetime.timezone.utc)}")
    print(f"Current 5-min block: {current_5min}")
    print(f"Next 5-min block: {next_5min}")

    print("\n" + "=" * 60)
    print("Searching for CURRENT 5-min market")
    print("=" * 60)
    await find_by_timestamp(current_5min)

    print("\n" + "=" * 60)
    print("Searching for NEXT 5-min market")
    print("=" * 60)
    await find_by_timestamp(next_5min)


if __name__ == "__main__":
    asyncio.run(main())