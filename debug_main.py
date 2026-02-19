#!/usr/bin/env python3
"""
debug_main.py - Find exactly where main.py crashes
"""
import sys
import traceback


def check_imports():
    """Check all imports step by step"""
    modules = [
        ("asyncio", None),
        ("logging", None),
        ("os", None),
        ("sys", None),
        ("time", None),
        ("yaml", "PyYAML"),
        ("dotenv", "python-dotenv"),
        ("websockets", "websockets"),
        ("aiohttp", "aiohttp"),
    ]

    for mod_name, pip_name in modules:
        try:
            __import__(mod_name)
            print(f"✓ {mod_name}")
        except ImportError as e:
            print(f"✗ {mod_name} MISSING (pip install {pip_name or mod_name})")
            return False

    # Check local imports
    print("\nChecking local modules...")
    try:
        from core.state import SharedState
        print("✓ core.state")
    except Exception as e:
        print(f"✗ core.state: {e}")
        traceback.print_exc()
        return False

    try:
        from core.binance_listener import BinanceListener
        print("✓ core.binance_listener")
    except Exception as e:
        print(f"✗ core.binance_listener: {e}")
        traceback.print_exc()
        return False

    try:
        from core.rtds_listener import RTDSListener
        print("✓ core.rtds_listener")
    except Exception as e:
        print(f"✗ core.rtds_listener: {e}")
        traceback.print_exc()
        return False

    try:
        from core.market_finder import MarketFinder
        print("✓ core.market_finder")
    except Exception as e:
        print(f"✗ core.market_finder: {e}")
        traceback.print_exc()
        return False

    try:
        from core.executor import PolyExecutor
        print("✓ core.executor")
    except Exception as e:
        print(f"✗ core.executor: {e}")
        traceback.print_exc()
        return False

    try:
        from strategy.signal_engine import SignalEngine
        print("✓ strategy.signal_engine")
    except Exception as e:
        print(f"✗ strategy.signal_engine: {e}")
        return False

    try:
        from strategy.taker_sniper import TakerSniper
        print("✓ strategy.taker_sniper")
    except Exception as e:
        print(f"✗ strategy.taker_sniper: {e}")
        return False

    try:
        from strategy.maker_quoter import MakerQuoter
        print("✓ strategy.maker_quoter")
    except Exception as e:
        print(f"✗ strategy.maker_quoter: {e}")
        return False

    try:
        from risk.position_manager import RiskManager
        print("✓ risk.position_manager")
    except Exception as e:
        print(f"✗ risk.position_manager: {e}")
        return False

    try:
        from risk.fee_calculator import FeeCalculator
        print("✓ risk.fee_calculator")
    except Exception as e:
        print(f"✗ risk.fee_calculator: {e}")
        return False

    try:
        from utils.logger import setup_logging
        print("✓ utils.logger")
    except Exception as e:
        print(f"✗ utils.logger: {e}")
        return False

    try:
        from utils.metrics import MetricsTracker
        print("✓ utils.metrics")
    except Exception as e:
        print(f"✗ utils.metrics: {e}")
        return False

    return True


def check_environment():
    """Check environment variables"""
    print("\nChecking environment...")
    import os
    required = ["POLYGON_PRIVATE_KEY"]
    for var in required:
        val = os.environ.get(var)
        if val:
            masked = val[:6] + "..." + val[-4:] if len(val) > 10 else "set"
            print(f"✓ {var} = {masked}")
        else:
            print(f"⚠ {var} NOT SET (needed for live trading)")


def test_asyncio():
    """Test asyncio can run basic tasks"""
    print("\nTesting asyncio...")
    import asyncio

    async def simple_task():
        return "asyncio works"

    try:
        result = asyncio.run(simple_task())
        print(f"✓ {result}")
        return True
    except Exception as e:
        print(f"✗ asyncio failed: {e}")
        traceback.print_exc()
        return False


def test_state():
    """Test SharedState creation"""
    print("\nTesting SharedState...")
    try:
        from core.state import SharedState
        state = SharedState()
        print(f"✓ SharedState created")
        print(f"  - btc_price: {state.btc_price}")
        print(f"  - feeds_ready: {state.feeds_ready}")
        return True
    except Exception as e:
        print(f"✗ SharedState failed: {e}")
        traceback.print_exc()
        return False


def main():
    print("=" * 50)
    print("Polymarket Sniper v2 - Diagnostic Tool")
    print("=" * 50)

    ok = True

    ok = check_imports() and ok
    check_environment()
    ok = test_asyncio() and ok
    ok = test_state() and ok

    print("\n" + "=" * 50)
    if ok:
        print("ALL CHECKS PASSED")
        print("Try running: python main.py")
    else:
        print("SOME CHECKS FAILED")
        print("Fix the errors above before running main.py")
    print("=" * 50)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())