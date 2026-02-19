"""
main.py - DEBUG VERSION
"""
import sys

print(f"Python: {sys.version}", flush=True)
print("Starting import...", flush=True)

try:
    import asyncio

    print("✓ asyncio", flush=True)
    import logging

    print("✓ logging", flush=True)
    import os

    print("✓ os", flush=True)
    import time

    print("✓ time", flush=True)
    import yaml

    print("✓ yaml", flush=True)
    from dotenv import load_dotenv

    print("✓ dotenv", flush=True)
    from core.state import SharedState

    print("✓ SharedState", flush=True)
    from core.binance_listener import BinanceListener

    print("✓ BinanceListener", flush=True)
    from core.poly_book_listener import PolyBookListener

    print("✓ PolyBookListener", flush=True)
    from core.rtds_listener import RTDSListener

    print("✓ RTDSListener", flush=True)
    from core.market_finder import MarketFinder

    print("✓ MarketFinder", flush=True)
    from core.executor import PolyExecutor

    print("✓ PolyExecutor", flush=True)
    from strategy.signal_engine import SignalEngine

    print("✓ SignalEngine", flush=True)
    from strategy.taker_sniper import TakerSniper

    print("✓ TakerSniper", flush=True)
    from strategy.maker_quoter import MakerQuoter

    print("✓ MakerQuoter", flush=True)
    from risk.position_manager import RiskManager

    print("✓ RiskManager", flush=True)
    from risk.fee_calculator import FeeCalculator

    print("✓ FeeCalculator", flush=True)
    from utils.logger import setup_logging

    print("✓ setup_logging", flush=True)
    from utils.metrics import MetricsTracker

    print("✓ MetricsTracker", flush=True)
except Exception as e:
    print(f"IMPORT FAILED: {type(e).__name__}: {e}", flush=True)
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\nAll imports successful!", flush=True)

load_dotenv()


def load_config(path: str = "config.yaml") -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}


cfg = load_config()
PAPER_TRADE = cfg.get("paper_trade", True)
TRADING_MODE = cfg.get("strategy", {}).get("mode", "dual")
LOG_LEVEL = cfg.get("log_level", "INFO")
LOG_FILE = cfg.get("log_file", "logs/bot.log")

print(f"Config: paper={PAPER_TRADE}, mode={TRADING_MODE}", flush=True)

# Setup logging with immediate console output
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("main")
logger.info("Logging initialized")


async def main():
    logger.info("=== Polymarket Sniper v2 Starting ===")

    try:
        state = SharedState()
        logger.info("State created")

        finder = MarketFinder(state)
        logger.info("MarketFinder created")

        binance = BinanceListener(state)
        logger.info("BinanceListener created")

        rtds = RTDSListener(state, symbol="btc")
        logger.info("RTDSListener created")

        poly_book = PolyBookListener(state)
        logger.info("PolyBookListener created")

        try:
            executor = PolyExecutor()
            logger.info("PolyExecutor created")
        except Exception as e:
            logger.error(f"PolyExecutor failed: {e}")
            logger.info("Continuing with paper trade mode...")
            executor = None

        metrics = MetricsTracker(log_dir="logs", state=state)
        logger.info("MetricsTracker created")

        balance = 1000.0 if PAPER_TRADE else (executor.get_balance() if executor else 1000.0)
        state.current_balance = balance
        logger.info(f"Balance: ${balance:.2f}")

        risk_mgr = RiskManager(starting_balance=balance)
        logger.info("RiskManager created")

        signal_engine = SignalEngine()
        taker = TakerSniper(state, executor, risk_mgr, metrics, paper_trade=PAPER_TRADE)
        maker = MakerQuoter(state, executor, metrics, paper_trade=PAPER_TRADE)
        logger.info("Strategy modules created")

        # Start core tasks
        logger.info("Starting core tasks...")

        async def market_finder_task():
            logger.info("Market finder starting...")
            await finder.start()

        async def binance_task():
            logger.info("Binance listener starting...")
            await binance.start()

        async def rtds_task():
            logger.info("RTDS listener starting...")
            await rtds.start()

        tasks = [
            asyncio.create_task(market_finder_task()),
            asyncio.create_task(binance_task()),
            asyncio.create_task(rtds_task()),
        ]

        logger.info("All core tasks created, gathering...")
        await asyncio.gather(*tasks)

    except Exception as e:
        logger.critical(f"Fatal error in main: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        print("Starting asyncio.run(main())...", flush=True)
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown by user", flush=True)
    except Exception as e:
        print(f"\nFatal error: {type(e).__name__}: {e}", flush=True)
        import traceback

        traceback.print_exc()