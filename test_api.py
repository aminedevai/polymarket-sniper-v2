# test_api.py - Diagnostic script for Polymarket APIs
import asyncio
import aiohttp
import websockets
import json
import sys


# Test 1: Gamma API with proper filtering
async def test_gamma():
    print("\n=== Testing Gamma API ===")
    url = "https://gamma-api.polymarket.com/markets"

    # Try multiple search strategies
    searches = [
        # Strategy 1: Look for active crypto markets
        {"active": "true", "closed": "false", "tag": "crypto", "limit": 50},
        # Strategy 2: Direct search for Bitcoin
        {"active": "true", "closed": "false", "_c": "bitcoin", "limit": 50},
        # Strategy 3: Search for 5-minute specifically
        {"active": "true", "closed": "false", "_c": "5-minute", "limit": 50},
    ]

    async with aiohttp.ClientSession() as session:
        for i, params in enumerate(searches):
            print(f"\nSearch {i + 1}: {params}")
            try:
                async with session.get(url, params=params, timeout=10) as r:
                    if r.status != 200:
                        print(f"  HTTP Error: {r.status}")
                        continue
                    markets = await r.json()
                    print(f"  Found {len(markets)} markets")

                    # Look for BTC markets
                    for m in markets[:5]:
                        q = m.get("question", "N/A")
                        if "btc" in q.lower() or "bitcoin" in q.lower():
                            print(f"  -> {q[:60]}")
                            print(f"     Active: {m.get('active')}, Closed: {m.get('closed')}")
                            print(f"     End: {m.get('endDateIso', 'N/A')}")
                            token_ids = m.get("clobTokenIds", [])
                            print(f"     Tokens: {len(token_ids)}")
                            if token_ids:
                                print(f"     First token: {token_ids[0][:20]}...")

            except Exception as e:
                print(f"  Error: {e}")


# Test 2: RTDS WebSocket with proper ping/pong
async def test_rtds():
    print("\n=== Testing RTDS WebSocket ===")
    print("Connecting to wss://ws-live-data.polymarket.com...")

    try:
        # Try with explicit subprotocols and headers
        async with websockets.connect(
                "wss://ws-live-data.polymarket.com",
                subprotocols=["realtime"],
                ping_interval=None,  # We'll handle ping manually
                ping_timeout=None,
        ) as ws:
            print("Connected! Sending subscription...")

            # Send subscription
            sub = {
                "action": "subscribe",
                "subscriptions": [
                    {
                        "topic": "crypto_prices_chainlink",
                        "type": "update",
                        "filters": json.dumps({"symbol": "btc/usd"})
                    }
                ]
            }
            await ws.send(json.dumps(sub))
            print(f"Sent: {json.dumps(sub, indent=2)}")

            # Start ping task
            async def send_pings():
                while True:
                    try:
                        await ws.send("PING")
                        await asyncio.sleep(5)
                    except:
                        break

            ping_task = asyncio.create_task(send_pings())

            # Listen for messages
            try:
                for i in range(10):  # Try to get 10 messages
                    msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    print(f"\n  Raw message {i + 1}: {msg[:200]}")

                    # Try to parse
                    try:
                        data = json.loads(msg)
                        print(f"  Parsed: topic={data.get('topic')}, type={data.get('type')}")
                        if data.get('topic') == 'crypto_prices_chainlink':
                            payload = data.get('payload', {})
                            if 'data' in payload:
                                latest = max(payload['data'], key=lambda x: x.get('timestamp', 0))
                                print(f"  Chainlink BTC: ${latest.get('value')} at {latest.get('timestamp')}")
                    except json.JSONDecodeError as e:
                        print(f"  JSON Error: {e}")

            except asyncio.TimeoutError:
                print("  Timeout waiting for messages")
            finally:
                ping_task.cancel()

    except websockets.exceptions.InvalidStatusCode as e:
        print(f"  Connection rejected with status: {e.status_code}")
        print("  This may indicate geographic blocking or authentication required")
    except Exception as e:
        print(f"  Connection error: {type(e).__name__}: {e}")


# Test 3: Check if we're being blocked
async def test_connection():
    print("\n=== Testing Basic Connectivity ===")
    import socket
    try:
        # Try to resolve the hostname
        addr = socket.getaddrinfo("ws-live-data.polymarket.com", None)
        print(f"DNS resolution: OK ({len(addr)} records)")

        # Try HTTP endpoint
        async with aiohttp.ClientSession() as s:
            async with s.get("https://ws-live-data.polymarket.com", timeout=5) as r:
                print(f"HTTP test: Status {r.status}")
                text = await r.text()
                print(f"Response preview: {text[:200]}")
    except Exception as e:
        print(f"Connectivity test failed: {e}")


async def main():
    await test_connection()
    await test_gamma()
    await test_rtds()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTests interrupted")