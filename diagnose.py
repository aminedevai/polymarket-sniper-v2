"""
diagnose.py - run this once to check API connectivity
"""
import asyncio
import aiohttp
import websockets
import json


async def check_gamma():
    print("\n=== GAMMA API - BTC Markets ===")
    url = "https://gamma-api.polymarket.com/markets"
    params = {"tag": "crypto", "active": "true", "closed": "false", "limit": 20}
    async with aiohttp.ClientSession() as s:
        async with s.get(url, params=params) as r:
            markets = await r.json()

    btc = [m for m in markets if "btc" in m.get("question", "").lower()
           or "bitcoin" in m.get("question", "").lower()]

    if not btc:
        print("NO BTC MARKETS FOUND - printing all questions:")
        for m in markets:
            print(f"  {m.get('question','')[:100]}")
    else:
        print(f"Found {len(btc)} BTC markets:")
        for m in btc:
            print(f"  active={m.get('active')} closed={m.get('closed')} "
                  f"q={m.get('question','')[:80]}")


async def check_chainlink():
    print("\n=== RTDS CHAINLINK FEED ===")
    try:
        async with websockets.connect(
            "wss://ws-live-data.polymarket.com",
            ping_interval=None
        ) as ws:
            sub = json.dumps({
                "action": "subscribe",
                "subscriptions": [{
                    "topic": "crypto_prices_chainlink",
                    "type": "*",
                    "filters": json.dumps({"symbol": "btc/usd"})
                }]
            })
            await ws.send(sub)
            print("Subscribed, waiting for messages (5 seconds)...")
            for i in range(5):
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
                    print(f"  msg {i+1}: {msg[:200]}")
                except asyncio.TimeoutError:
                    print(f"  msg {i+1}: TIMEOUT - no data received")
                    break
    except Exception as e:
        print(f"  ERROR: {e}")


async def main():
    await check_gamma()
    await check_chainlink()
    print("\nDone.")

asyncio.run(main())
