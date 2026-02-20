"""Test multiple free Polygon RPCs to find one that works"""
import asyncio, aiohttp

RPCS = [
    "https://rpc.ankr.com/polygon",
    "https://polygon.llamarpc.com",
    "https://1rpc.io/matic",
    "https://rpc-mainnet.matic.quiknode.pro",
    "https://polygon.drpc.org",
]

PAYLOAD = {
    "jsonrpc": "2.0",
    "method": "eth_call",
    "params": [
        {"to": "0xc907E116054Ad103354f2D350FD2514433D57F6f", "data": "0x50d25bcd"},
        "latest"
    ],
    "id": 1
}

async def test_rpc(session, url):
    try:
        async with session.post(url, json=PAYLOAD, timeout=aiohttp.ClientTimeout(total=5)) as r:
            data = await r.json()
            result = data.get("result", "")
            if result and result != "0x":
                price = int(result, 16) / 1e8
                print(f"  ✓ {url} → BTC=${price:.2f}")
                return url, price
            else:
                print(f"  ✗ {url} → {str(data)[:80]}")
    except Exception as e:
        print(f"  ✗ {url} → {e}")
    return None, 0

async def main():
    async with aiohttp.ClientSession() as session:
        for rpc in RPCS:
            url, price = await test_rpc(session, rpc)
            if price > 0:
                print(f"\nBest RPC: {url}")
                break

asyncio.run(main())