"""
test_tokens.py - inspect the token structure of a 5-min BTC market
"""
import asyncio
import aiohttp
import time
import json


async def main():
    now = int(time.time())
    ts = (now // 300) * 300  # current 5-min boundary
    slug = f"btc-updown-5m-{ts}"

    async with aiohttp.ClientSession() as session:
        async with session.get(
            "https://gamma-api.polymarket.com/events",
            params={"slug": slug},
            timeout=aiohttp.ClientTimeout(total=8)
        ) as r:
            data = await r.json()

    event = data[0]
    market = event["markets"][0]

    print(f"Event title: {event.get('title')}")
    print(f"Market question: {market.get('question')}")
    print(f"conditionId: {market.get('conditionId', 'N/A')}")
    print(f"Total clobTokenIds: {len(market.get('clobTokenIds', []))}")
    print()

    # Print first 5 tokens
    tokens = market.get("clobTokenIds", [])
    print(f"First 5 tokens:")
    for t in tokens[:5]:
        print(f"  {t}")
    print()

    # Check outcomes field
    outcomes = market.get("outcomes", [])
    print(f"Outcomes: {outcomes}")

    # Check outcomePrices
    prices = market.get("outcomePrices", [])
    print(f"outcomePrices: {prices}")

    # Print full market keys
    print(f"\nAll market keys: {list(market.keys())}")

    # Print raw for inspection
    print(f"\nFull market (truncated):")
    safe = {k: v for k, v in market.items() if k != "clobTokenIds"}
    print(json.dumps(safe, indent=2)[:2000])

asyncio.run(main())