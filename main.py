"""
main.py
=======
Async orchestrator for Polymarket Sniper v2.
"""

import asyncio
import logging
import os
import sys
import time

import yaml
from dotenv import load_dotenv

from core.state import SharedState
from core.binance_listener import BinanceListener
from core.poly_book_listener import PolyBookListener
from core.rtds_listener import RTDSListener
from core.chainlink_listener import ChainlinkListener
from core.market_finder import MarketFinder
from core.executor import PolyExecutor
from strategy.signal_engine import SignalEngine
from strategy.taker_sniper import TakerSniper
from strategy.maker_quoter import MakerQuoter
from risk.position_manager import RiskManager
from risk.fee_calculator import FeeCalculator
from utils.logger import setup_logging
from utils.metrics import MetricsTracker

load_dotenv()


def load_config(path: str = "config.yaml") -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}


cfg = load_config()
PAPER_TRADE  = cfg.get("paper_trade", True)
TRADING_MODE = cfg.get("strategy", {}).get("mode", "dual")
LOG_LEVEL    = cfg.get("log_level", "INFO")
LOG_FILE     = cfg.get("log_file", "logs/bot.log")
LOOP_MS      = 0.05   # 50ms decision loop

setup_logging(level=LOG_LEVEL, log_file=LOG_FILE)
logger = logging.getLogger("main")


async def trading_loop(
    state: SharedState,
    signal_engine: SignalEngine,
    taker: TakerSniper,
    maker: MakerQuoter,
    risk_mgr: RiskManager,
    executor: PolyExecutor,
    metrics: MetricsTracker,
):
    logger.info(f"Trading loop started | mode={TRADING_MODE} | paper={PAPER_TRADE}")

    balance_refresh_interval = 30.0
    last_balance_refresh = 0.0

    while True:
        await asyncio.sleep(LOOP_MS)

        if not risk_mgr.is_active():
            logger.critical("Risk manager halted. Stopping.")
            break

        if not state.active_token_id:
            metrics.tick()
            continue

        now = time.monotonic()
        if (now - last_balance_refresh) >= balance_refresh_interval:
            if not PAPER_TRADE:
                bal = executor.get_balance()
                risk_mgr.update_balance(bal)
                state.current_balance = bal
            else:
                if not state.current_balance:
                    state.current_balance = 1000.0
            last_balance_refresh = now

        available = state.current_balance

        if TRADING_MODE in ("taker", "dual"):
            signal = signal_engine.evaluate(state)
            if signal.should_trade:
                taker.execute(signal, available)

        metrics.tick()


async def rotating_book_listener(state: SharedState, poly_book: PolyBookListener):
    """Restart CLOB WS subscription whenever market token rotates."""
    last_token = ""
    while True:
        current_token = state.active_token_id
        if current_token and current_token != last_token:
            last_token = current_token
            logger.info(f"Book listener subscribing: {current_token[:16]}...")
            try:
                await poly_book.start(current_token)
            except Exception as e:
                logger.warning(f"Book listener error: {e}")
        await asyncio.sleep(1.0)


async def main():
    logger.info("=" * 50)
    logger.info("  POLYMARKET SNIPER v2 STARTING")
    logger.info(f"  Mode: {TRADING_MODE} | Paper: {PAPER_TRADE}")
    logger.info("=" * 50)

    state   = SharedState()
    finder  = MarketFinder(state)
    binance = BinanceListener(state)
    rtds    = RTDSListener(state, symbol="btc")
    chainlink = ChainlinkListener(state)
    poly_book = PolyBookListener(state)
    executor  = PolyExecutor()

    metrics  = MetricsTracker(log_dir="logs", state=state)
    balance  = executor.get_balance() if not PAPER_TRADE else 1000.0
    state.current_balance = balance
    risk_mgr = RiskManager(starting_balance=balance)

    logger.info(f"Balance: ${balance:.2f} USDC")

    signal_engine = SignalEngine()
    taker = TakerSniper(state, executor, risk_mgr, metrics, paper_trade=PAPER_TRADE)
    maker = MakerQuoter(state, executor, metrics, paper_trade=PAPER_TRADE)

    # Bootstrap: get initial market
    logger.info("Fetching initial market...")
    initial = await finder.fetch_once()
    if initial:
        state.active_token_id   = initial["_token_up"]
        state.active_token_down = initial["_token_down"]
        state.active_condition_id = initial.get("conditionId", "")
        state.market_end_ts = initial["_end_ts"]
        state.seconds_remaining = max(0, initial["_end_ts"] - int(time.time()))
        if state.rtds_chainlink_price > 0:
            state.candle_open_price = state.rtds_chainlink_price
        logger.info(
            f"Initial market: {initial.get('question','')[:55]} | "
            f"ends in {state.seconds_remaining:.0f}s | "
            f"UP={state.active_token_id[:12]}... "
            f"DOWN={state.active_token_down[:12]}..."
        )
    else:
        logger.warning("No initial market found — waiting for MarketFinder poll.")

    # Print fee table
    logger.info("Fee curve:")
    FeeCalculator.print_fee_table()

    tasks = [
        asyncio.create_task(finder.start(),                           name="market_finder"),
        asyncio.create_task(binance.start(),                          name="binance_ws"),
        asyncio.create_task(rtds.start(),                             name="rtds_ws"),
        asyncio.create_task(chainlink.start(),                        name="chainlink"),
        asyncio.create_task(rotating_book_listener(state, poly_book), name="book_listener"),
        asyncio.create_task(
            trading_loop(state, signal_engine, taker, maker, risk_mgr, executor, metrics),
            name="trading_loop",
        ),
    ]

    if TRADING_MODE in ("maker", "dual"):
        tasks.append(asyncio.create_task(maker.run(), name="maker_quoter"))

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    except Exception as e:
        logger.critical(f"Fatal: {e}", exc_info=True)
    finally:
        logger.info("Shutting down...")
        try:
            await maker.cancel_all_quotes()
        except Exception:
            pass
        metrics.close()
        logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())