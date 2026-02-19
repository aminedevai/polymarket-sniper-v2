#!/usr/bin/env python3
"""
find_5min_markets.py - Diagnostic tool to locate 5-minute BTC markets
"""
import asyncio
import aiohttp
import json
from datetime import datetime

GAMMA_URL = "https://gamma-api.polymarket.com/markets"


async def try_endpoint(session, params, description):
    """Try a specific API endpoint configuration"""
    print(f"\n=== {description} ===")
    print(f"Params: {json.dumps(params, indent=2)}")

    try:
        async with session.get(GAMMA_URL, params=params, timeout=10) as resp:
            print(f"Status: {resp.status}")
            if resp.status != 200:
                print(f"Error: HTTP {resp.status}")
                return []

            markets = await resp.json()
            print(f"Total markets returned: {len(markets)}")

            # Filter for BTC-related markets
            btc_markets = []
            for m in markets:
                q = (m.get("question", "") or "").lower()
                if "btc" in q or "bitcoin" in q:
                    btc_markets.append(m)

            print(f"BTC-related markets: {len(btc_markets)}")

            for m in btc_markets[:5]:
                q = m.get("question", "N/A")
                active = m.get("active", False)
                closed = m.get("closed", False)
                end_date = m.get("endDateIso", "N/A")
                tokens = m.get("clobTokenIds", [])

                print(f"  - {q[:60]}")
                print(f"    Active: {active}, Closed: {closed}")
                print(f"    End: {end_date}")
                print(f"    Tokens: {len(tokens)}")
                if tokens:
                    print(f"    First token: {tokens[0][:20]}...")

                # Check if it's a 5-minute market
                if "5" in q and ("minute" in q or "min" in q):
                    print(f"    *** POTENTIAL 5-MIN MARKET ***")

            return btc_markets

    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        return []


async def main():
    print("=" * 60)
    print("Searching for 5-minute BTC markets in Polymarket Gamma API")
    print("=" * 60)

    async with aiohttp.ClientSession() as session:
        # Strategy 1: Active crypto markets
        await try_endpoint(
            session,
            {"active": "true", "closed": "false", "tag": "crypto", "limit": 100},
            "Strategy 1: Active crypto markets (tag=crypto)"
        )

        # Strategy 2: Search for "5" in content
        await try_endpoint(
            session,
            {"active": "true", "closed": "false", "_c": "5", "limit": 50},
            "Strategy 2: Content search for '5'"
        )

        # Strategy 3: Search for "minute"
        await try_endpoint(
            session,
            {"active": "true", "closed": "false", "_c": "minute", "limit": 50},
            "Strategy 3: Content search for 'minute'"
        )

        # Strategy 4: Bitcoin specific
        await try_endpoint(
            session,
            {"active": "true", "closed": "false", "_c": "bitcoin", "limit": 50},
            "Strategy 4: Content search for 'bitcoin'"
        )

        # Strategy 5: No filter, just active
        await try_endpoint(
            session,
            {"active": "true", "closed": "false", "limit": 200},
            "Strategy 5: All active markets (no filter)"
        )

        # Strategy 6: Try events endpoint instead
        print("\n=== Strategy 6: Events endpoint ===")
        events_url = "https://gamma-api.polymarket.com/events"
        try:
            async with session.get(
                    events_url,
                    params={"active": "true", "closed": "false", "tag": "crypto", "limit": 50},
                    timeout=10
            ) as resp:
                print(f"Status: {resp.status}")
                events = await resp.json()
                print(f"Events found: {len(events)}")

                for e in events[:10]:
                    title = e.get("title", "N/A")
                    if "btc" in title.lower() or "bitcoin" in title.lower():
                        print(f"  - Event: {title[:60]}")
                        markets = e.get("markets", [])
                        print(f"    Markets in event: {len(markets)}")
                        for m in markets:
                            mq = m.get("question", "")
                            if "5" in mq:
                                print(f"      *** 5-MIN MARKET: {mq[:50]}...")
                                print(f"      Token: {m.get('clobTokenIds', ['none'])[0][:20]}...")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())