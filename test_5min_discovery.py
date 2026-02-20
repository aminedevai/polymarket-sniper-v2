"""
find_5min_markets.py
====================
Run this to find exactly what 5-minute BTC markets exist on Polymarket right now.
Tries multiple API endpoints and search strategies.

Usage:
    python find_5min_markets.py
"""
import asyncio
import aiohttp
import json

GAMMA_URL = "https://gamma-api.polymarket.com/markets"
GAMMA_EVENTS = "https://gamma-api.polymarket.com/events"


async def try_endpoint(session, url, params, label):
    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as r:
            data = await r.json()
            if isinstance(data, list):
                print(f"\n[{label}] → {len(data)} results")
                for m in data[:5]:
                    q = m.get("question", m.get("title", ""))[:100]
                    active = m.get("active")
                    closed = m.get("closed")
                    end = m.get("endDate", "")[:20]
                    cids = m.get("clobTokenIds", [])
                    print(f"  active={active} closed={closed} end={end} clobIds={len(cids)} | {q}")
                if len(data) > 5:
                    print(f"  ... and {len(data)-5} more")
            else:
                print(f"\n[{label}] → non-list response: {str(data)[:200]}")
    except Exception as e:
        print(f"\n[{label}] ERROR: {e}")


async def main():
    async with aiohttp.ClientSession() as session:

        print("=" * 70)
        print("STRATEGY 1: Search by keywords")
        print("=" * 70)
        for kw in ["5-minute", "5 minute", "5min", "BTC", "Bitcoin", "crypto"]:
            await try_endpoint(session, GAMMA_URL, {
                "_c": kw, "active": "true", "closed": "false", "limit": 5
            }, f"keyword={kw}")

        print("\n" + "=" * 70)
        print("STRATEGY 2: Search by tag")
        print("=" * 70)
        for tag in ["crypto", "bitcoin", "finance", "5-minute"]:
            await try_endpoint(session, GAMMA_URL, {
                "tag": tag, "active": "true", "closed": "false", "limit": 5
            }, f"tag={tag}")

        print("\n" + "=" * 70)
        print("STRATEGY 3: Events endpoint")
        print("=" * 70)
        for kw in ["BTC", "Bitcoin", "5-minute"]:
            await try_endpoint(session, GAMMA_EVENTS, {
                "_c": kw, "active": "true", "closed": "false", "limit": 5
            }, f"events keyword={kw}")

        print("\n" + "=" * 70)
        print("STRATEGY 4: Raw dump — first 20 active markets NO filter")
        print("=" * 70)
        await try_endpoint(session, GAMMA_URL, {
            "active": "true", "closed": "false", "limit": 20
        }, "no filter")

        print("\n" + "=" * 70)
        print("STRATEGY 5: Check if 5-min markets exist at all (including closed)")
        print("=" * 70)
        for kw in ["5-minute", "5 minute", "BTC up", "BTC down"]:
            await try_endpoint(session, GAMMA_URL, {
                "_c": kw, "limit": 5
            }, f"no active filter, keyword={kw}")

    print("\n" + "=" * 70)
    print("DONE — paste the output above so we can fix market_finder.py")
    print("=" * 70)


asyncio.run(main())