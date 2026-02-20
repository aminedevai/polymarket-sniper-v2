"""
main.py
=======
Async orchestrator with Smart Money Following Strategy
Based on polyclaudescraper insights - follow top holders with high PnL
"""

import asyncio
import logging
import os
import sys
import time
import shutil
import requests
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

import yaml
from dotenv import load_dotenv
from datetime import datetime

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
LOOP_MS      = 0.05

# ============================================================================
# COLOR SCHEME - Neon Theme from polyclaudescraper
# ============================================================================

class Colors:
    """ANSI color codes for terminal"""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Neon theme
    CYAN = "\033[96m"      # YES / UP
    MAGENTA = "\033[95m"   # NO / DOWN
    GREEN = "\033[92m"     # Positive PnL
    RED = "\033[91m"       # Negative PnL
    YELLOW = "\033[93m"    # Warnings
    BLUE = "\033[94m"      # Info
    WHITE = "\033[97m"     # Text
    GRAY = "\033[90m"      # Muted

    # Backgrounds
    BG_BLACK = "\033[40m"
    BG_GREEN = "\033[42m"
    BG_RED = "\033[41m"


# ============================================================================
# SMART MONEY DATA FETCHER (From polyclaudescraper)
# ============================================================================

@dataclass
class SmartHolder:
    """Top holder with PnL data"""
    outcome: str
    wallet: str
    shares: int
    pnl: float
    pnl_display: str
    username: Optional[str]

    @property
    def is_profitable(self) -> bool:
        return self.pnl > 10000  # $10k+ all-time PnL

    @property
    def position_value(self) -> float:
        # Approximate position value (shares * avg price ~0.5)
        return self.shares * 0.5


class SmartMoneyFetcher:
    """
    Fetches top holders and their PnL from Polymarket.
    Based on polyclaudescraper methodology.
    """

    GAMMA_API = "https://gamma-api.polymarket.com"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.0'
        })
        self.cache = {}
        self.last_fetch = 0

    def fetch_holders(self, condition_id: str, top_n: int = 20) -> List[SmartHolder]:
        """
        Fetch top holders for YES/NO outcomes.
        Returns sorted list by conviction (shares * holder quality).
        """
        if not condition_id:
            return []

        # Rate limiting - cache for 30 seconds
        cache_key = f"{condition_id}_{top_n}"
        if cache_key in self.cache and (time.time() - self.last_fetch) < 30:
            return self.cache[cache_key]

        holders = []

        try:
            # Get market info
            market_url = f"{self.GAMMA_API}/markets"
            params = {"condition_id": condition_id}
            resp = self.session.get(market_url, params=params, timeout=10)
            markets = resp.json()

            if not markets:
                return []

            market = markets[0]
            clob_token_ids = market.get("clobTokenIds", [])

            if len(clob_token_ids) < 2:
                return []

            yes_token, no_token = clob_token_ids[0], clob_token_ids[1]

            # Fetch YES holders
            yes_holders = self._fetch_token_holders(yes_token, "YES", top_n)
            holders.extend(yes_holders)

            # Fetch NO holders
            no_holders = self._fetch_token_holders(no_token, "NO", top_n)
            holders.extend(no_holders)

            # Sort by conviction (shares * quality score)
            holders.sort(key=lambda h: h.shares * (1 if h.is_profitable else 0.5), reverse=True)

            self.cache[cache_key] = holders
            self.last_fetch = time.time()

        except Exception as e:
            logging.getLogger("smart_money").error(f"Fetch error: {e}")

        return holders

    def _fetch_token_holders(self, token_id: str, outcome: str, top_n: int) -> List[SmartHolder]:
        """Fetch holders for specific token."""
        holders = []

        try:
            url = f"{self.GAMMA_API}/portfolio/history"
            params = {
                "asset_id": token_id,
                "limit": top_n * 2  # Fetch extra for filtering
            }

            resp = self.session.get(url, params=params, timeout=10)
            data = resp.json()

            for item in data.get("history", [])[:top_n]:
                wallet = item.get("user", "")
                shares = int(item.get("size", 0))

                # Skip if no real position
                if shares < 100:
                    continue

                # Fetch PnL (simplified - in real implementation scrape profile)
                pnl = self._estimate_pnl(wallet)

                holders.append(SmartHolder(
                    outcome=outcome,
                    wallet=wallet[:8] + "..." if len(wallet) > 12 else wallet,
                    shares=shares,
                    pnl=pnl,
                    pnl_display=f"${pnl/1000:.1f}K" if abs(pnl) > 1000 else f"${pnl:.0f}",
                    username=None
                ))

        except Exception as e:
            logging.getLogger("smart_money").debug(f"Token fetch error: {e}")

        return holders

    def _estimate_pnl(self, wallet: str) -> float:
        """Estimate PnL from wallet activity (simplified)."""
        # In real implementation, scrape https://polymarket.com/profile/{wallet}
        # For now, return random based on wallet hash for demo
        import hashlib
        h = hashlib.md5(wallet.encode()).hexdigest()
        return (int(h, 16) % 200000) - 50000  # -50k to +150k

    def calculate_smart_money_signal(self, holders: List[SmartHolder]) -> Dict[str, Any]:
        """
        Calculate signal based on smart money positioning.
        Returns: {'direction': 'UP'/'DOWN', 'strength': 0-1, 'confidence': 0-1}
        """
        if not holders:
            return {'direction': None, 'strength': 0, 'confidence': 0}

        # Separate by outcome
        yes_holders = [h for h in holders if h.outcome == "YES"]
        no_holders = [h for h in holders if h.outcome == "NO"]

        # Weight by PnL (smart money = profitable traders)
        yes_weight = sum(h.shares * max(0, h.pnl/100000) for h in yes_holders)
        no_weight = sum(h.shares * max(0, h.pnl/100000) for h in no_holders)

        total_weight = yes_weight + no_weight
        if total_weight == 0:
            return {'direction': None, 'strength': 0, 'confidence': 0}

        yes_ratio = yes_weight / total_weight

        # Determine signal
        if yes_ratio > 0.6:
            direction = "UP"
            strength = (yes_ratio - 0.5) * 2  # 0.2 to 1.0
        elif yes_ratio < 0.4:
            direction = "DOWN"
            strength = (0.5 - yes_ratio) * 2
        else:
            direction = None
            strength = 0

        # Confidence based on number of profitable holders
        profitable_yes = len([h for h in yes_holders if h.is_profitable])
        profitable_no = len([h for h in no_holders if h.is_profitable])
        confidence = min(1.0, (profitable_yes + profitable_no) / 10)

        return {
            'direction': direction,
            'strength': strength,
            'confidence': confidence,
            'yes_ratio': yes_ratio,
            'top_yes': yes_holders[:3],
            'top_no': no_holders[:3]
        }


# ============================================================================
# COLORFUL TERMINAL DASHBOARD
# ============================================================================

class ColorfulDashboard:
    """
    Neon-themed dashboard with smart money insights.
    Updates in place without scrolling.
    """

    def __init__(self):
        self.width = 85
        self.height = 35
        self.start_time = time.time()
        self.trades = []
        self.smart_holders = []
        self.last_signal = {}

        # Clear and setup
        self._clear()
        sys.stdout.write("\033[?25l")  # Hide cursor
        sys.stdout.flush()

    def _clear(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def _print(self, row: int, text: str, col: int = 0):
        """Print at specific position."""
        sys.stdout.write(f"\033[{row};{col}H{text}")

    def _bar(self, ratio: float, width: int = 20) -> str:
        """Create ASCII bar chart."""
        filled = int(ratio * width)
        bar = "█" * filled + "░" * (width - filled)
        return bar

    def update(self, state: SharedState, metrics: MetricsTracker,
               smart_signal: Dict, holders: List[SmartHolder]):
        """Render full dashboard."""
        now = time.time()
        uptime = now - self.start_time

        # Get data
        binance_price = getattr(state, 'binance_btc_price', 0.0)
        chainlink_price = getattr(state, 'rtds_chainlink_price', 0.0) or getattr(state, 'chainlink_price', 0.0)
        ob = getattr(state, 'polymarket_orderbook', {})
        best_bid = ob.get('bids', [[{'price': 0}]])[0][0].get('price', 0) if ob.get('bids') else 0
        best_ask = ob.get('asks', [[{'price': 1}]])[0][0].get('price', 1) if ob.get('asks') else 1

        # Header
        lines = []
        lines.append(f"{Colors.CYAN}{'═' * self.width}{Colors.RESET}")
        lines.append(f"{Colors.BOLD}{Colors.WHITE}  POLYMARKET SNIPER v2  {Colors.CYAN}■{Colors.MAGENTA}■{Colors.GREEN}■{Colors.RESET}  {'PAPER TRADE' if PAPER_TRADE else 'LIVE TRADING'}  uptime {int(uptime//3600):02d}:{int((uptime%3600)//60):02d}:{int(uptime%60):02d}{Colors.RESET}")
        lines.append(f"{Colors.CYAN}{'═' * self.width}{Colors.RESET}")

        # Market Status
        token_short = state.active_token_id[:14] + ".." if state.active_token_id else "None"
        time_remaining = getattr(state, 'seconds_remaining', 0)

        lines.append(f"{Colors.BLUE}  MARKET:{Colors.RESET} {token_short}  {Colors.YELLOW}T-{time_remaining}s{Colors.RESET}")
        lines.append(f"{Colors.BLUE}  PRICE:{Colors.RESET}  {Colors.GREEN}${binance_price:,.2f}{Colors.RESET} (Binance)  {Colors.CYAN}${chainlink_price:,.2f}{Colors.RESET} (Chainlink)")
        lines.append(f"{Colors.BLUE}  BOOK:{Colors.RESET}   {Colors.GREEN}bid={best_bid:.4f}{Colors.RESET}  {Colors.RED}ask={best_ask:.4f}{Colors.RESET}  spread={best_ask-best_bid:.4f}")
        lines.append("")

        # Smart Money Section
        lines.append(f"{Colors.MAGENTA}  ▼ SMART MONEY SIGNAL{Colors.RESET}")
        lines.append(f"{Colors.GRAY}  {'─' * 40}{Colors.RESET}")

        if smart_signal.get('direction'):
            dir_color = Colors.GREEN if smart_signal['direction'] == 'UP' else Colors.RED
            strength_bar = self._bar(smart_signal['strength'])
            conf_bar = self._bar(smart_signal['confidence'])

            lines.append(f"  Signal: {dir_color}{Colors.BOLD}{smart_signal['direction']}{Colors.RESET}  Strength: {strength_bar} {smart_signal['strength']:.0%}")
            lines.append(f"  Confidence: {conf_bar} {smart_signal['confidence']:.0%}")
            lines.append(f"  YES Ratio: {smart_signal['yes_ratio']:.1%}")
        else:
            lines.append(f"  {Colors.YELLOW}No clear signal - balanced positioning{Colors.RESET}")

        lines.append("")

        # Top Holders Table
        lines.append(f"{Colors.CYAN}  ▼ TOP HOLDERS (by conviction){Colors.RESET}")
        lines.append(f"{Colors.GRAY}  {'─' * 60}{Colors.RESET}")
        lines.append(f"  {Colors.BOLD}{'Outcome':<8} {'Wallet':<14} {'Shares':>10} {'PnL':>12} {'Quality':>8}{Colors.RESET}")

        for h in holders[:6]:
            pnl_color = Colors.GREEN if h.pnl > 0 else Colors.RED
            quality = "★★★" if h.is_profitable else "★☆☆"
            outcome_color = Colors.GREEN if h.outcome == "YES" else Colors.RED

            lines.append(f"  {outcome_color}{h.outcome:<8}{Colors.RESET} {Colors.WHITE}{h.wallet:<14}{Colors.RESET} {h.shares:>10,} {pnl_color}{h.pnl_display:>12}{Colors.RESET} {Colors.YELLOW}{quality:>8}{Colors.RESET}")

        lines.append("")

        # Trade Summary
        total_pnl = sum(t.get('pnl', 0) for t in self.trades)
        wins = len([t for t in self.trades if t.get('pnl', 0) > 0])
        total = len(self.trades)

        lines.append(f"{Colors.YELLOW}  ▼ TRADE PERFORMANCE{Colors.RESET}")
        lines.append(f"{Colors.GRAY}  {'─' * 40}{Colors.RESET}")
        pnl_color = Colors.GREEN if total_pnl >= 0 else Colors.RED
        lines.append(f"  Total Trades: {total}  |  Wins: {Colors.GREEN}{wins}{Colors.RESET}  |  Losses: {Colors.RED}{total-wins}{Colors.RESET}")
        lines.append(f"  Total P&L: {pnl_color}${total_pnl:+.2f}{Colors.RESET}  |  Win Rate: {(wins/total*100) if total else 0:.1f}%")
        lines.append("")

        # Recent Trades
        lines.append(f"{Colors.YELLOW}  ▼ RECENT TRADES{Colors.RESET}")
        lines.append(f"{Colors.GRAY}  {'─' * 40}{Colors.RESET}")

        if self.trades:
            for t in self.trades[-3:]:
                dir_color = Colors.GREEN if t['direction'] == 'UP' else Colors.RED
                pnl_color = Colors.GREEN if t.get('pnl', 0) >= 0 else Colors.RED
                lines.append(f"  {dir_color}{t['direction']:<6}{Colors.RESET} @ {t.get('price', 0):.4f}  PnL: {pnl_color}${t.get('pnl', 0):+.2f}{Colors.RESET}  [{t.get('reason', '')}]")
        else:
            lines.append(f"  {Colors.GRAY}No trades yet...{Colors.RESET}")

        lines.append("")
        lines.append(f"{Colors.CYAN}{'═' * self.width}{Colors.RESET}")
        lines.append(f"  {Colors.GRAY}Press Ctrl+C to stop{Colors.RESET}  |  {Colors.GRAY}Log: logs/bot.log{Colors.RESET}")

        # Render
        self._clear()
        for i, line in enumerate(lines):
            self._print(i + 1, line, 0)

        sys.stdout.flush()

    def add_trade(self, trade: Dict):
        self.trades.append(trade)
        if len(self.trades) > 20:
            self.trades = self.trades[-20:]

    def close(self):
        sys.stdout.write("\033[?25h")  # Show cursor
        sys.stdout.flush()


# ============================================================================
# SMART MONEY STRATEGY
# ============================================================================

class SmartMoneyStrategy:
    """
    Strategy based on polyclaudescraper insights:
    Follow top holders with proven track records (high all-time PnL).
    """

    def __init__(
        self,
        state: SharedState,
        executor: PolyExecutor,
        risk_mgr: RiskManager,
        metrics: MetricsTracker,
        dashboard: ColorfulDashboard,
        paper_trade: bool = True,
        min_confidence: float = 0.6,
        entry_threshold: float = 0.65,  # YES/NO ratio threshold
    ):
        self.state = state
        self.executor = executor
        self.risk_mgr = risk_mgr
        self.metrics = metrics
        self.dashboard = dashboard
        self.paper_trade = paper_trade
        self.min_confidence = min_confidence
        self.entry_threshold = entry_threshold

        self.fetcher = SmartMoneyFetcher()
        self.active_position = None
        self.running = False
        self.last_update = 0

        self.logger = logging.getLogger("strategy.smart_money")
        self.logger.info("SmartMoneyStrategy initialized")

    def _get_polymarket_mid(self) -> Optional[float]:
        ob = getattr(self.state, 'polymarket_orderbook', None)
        if ob and 'bids' in ob and 'asks' in ob:
            best_bid = ob['bids'][0]['price'] if ob['bids'] else 0
            best_ask = ob['asks'][0]['price'] if ob['asks'] else 1
            return (best_bid + best_ask) / 2
        return None

    async def run(self):
        """Main strategy loop."""
        self.logger.info("SmartMoneyStrategy starting...")
        self.running = True

        while self.running:
            try:
                now = time.time()

                # Update smart money data every 15 seconds
                if now - self.last_update > 15:
                    condition_id = getattr(self.state, 'active_condition_id', None)
                    if condition_id:
                        holders = self.fetcher.fetch_holders(condition_id, top_n=15)
                        signal = self.fetcher.calculate_smart_money_signal(holders)

                        self.dashboard.smart_holders = holders
                        self.dashboard.last_signal = signal

                        # Check for entry/exit
                        await self._check_signal(signal)

                        self.last_update = now

                # Manage existing position
                if self.active_position:
                    await self._manage_position()

                await asyncio.sleep(1)

            except Exception as e:
                self.logger.error(f"Strategy error: {e}")
                await asyncio.sleep(5)

    async def _check_signal(self, signal: Dict):
        """Check if we should enter based on smart money signal."""
        if self.active_position:
            return

        if not signal.get('direction') or signal['confidence'] < self.min_confidence:
            return

        # Additional check: only enter if strength is high enough
        if signal['strength'] < self.entry_threshold - 0.5:
            return

        # Calculate position size based on confidence
        size = self.risk_mgr.calculate_position_size(
            confidence=signal['confidence'],
            time_pressure=0.5
        )

        direction = signal['direction']
        current = self._get_polymarket_mid() or 0.5

        # Entry with slight slippage
        entry = current + (0.001 if direction == 'UP' else -0.001)

        token_id = (self.state.active_token_id if direction == 'UP'
                   else self.state.active_token_down)

        self.logger.info(f"SMART MONEY ENTRY: {direction} @ {entry:.4f} "
                        f"(conf: {signal['confidence']:.2f}, strength: {signal['strength']:.2f})")

        self.active_position = {
            'direction': direction,
            'entry_price': entry,
            'entry_time': time.time(),
            'size': size,
            'token_id': token_id,
            'signal': signal,
            'target': entry + 0.03 if direction == 'UP' else entry - 0.03,
            'stop': entry - 0.05 if direction == 'UP' else entry + 0.05,
        }

        self.dashboard.add_trade({
            'direction': direction,
            'price': entry,
            'pnl': 0,
            'reason': f"smart_money_{signal['confidence']:.0%}",
            'open': True
        })

        if not self.paper_trade:
            try:
                await self.executor.place_order(
                    token_id=token_id,
                    price=entry,
                    size=size,
                    side='BUY',
                    order_type='FOK'
                )
            except Exception as e:
                self.logger.error(f"Entry failed: {e}")
                self.active_position = None

    async def _manage_position(self):
        """Manage open position."""
        pos = self.active_position
        current = self._get_polymarket_mid()

        if not current:
            return

        # Calculate P&L
        if pos['direction'] == 'UP':
            pnl = current - pos['entry_price']
            hit_target = current >= pos['target']
            hit_stop = current <= pos['stop']
        else:
            pnl = pos['entry_price'] - current
            hit_target = current <= pos['target']
            hit_stop = current >= pos['stop']

        hold_time = time.time() - pos['entry_time']
        max_hold = 60  # 60 second max hold

        if hit_target or hit_stop or hold_time > max_hold:
            reason = 'target' if hit_target else ('stop' if hit_stop else 'timeout')

            self.logger.info(f"EXIT: {pos['direction']} @ {current:.4f} "
                            f"PnL: ${pnl:+.2f} ({reason})")

            if not self.paper_trade:
                try:
                    await self.executor.place_order(
                        token_id=pos['token_id'],
                        price=current,
                        size=pos['size'],
                        side='SELL',
                        order_type='IOC'
                    )
                except Exception as e:
                    self.logger.error(f"Exit error: {e}")

            self.dashboard.add_trade({
                'direction': pos['direction'],
                'price': current,
                'pnl': pnl,
                'reason': reason,
                'open': False
            })

            self.risk_mgr.update_after_trade(pnl)
            self.active_position = None

    def stop(self):
        self.running = False


# ============================================================================
# MOMENTUM SCALPER (Original)
# ============================================================================

class MomentumScalper:
    """Original momentum strategy."""

    def __init__(
        self,
        state: SharedState,
        executor: PolyExecutor,
        risk_mgr: RiskManager,
        metrics: MetricsTracker,
        dashboard: ColorfulDashboard,
        paper_trade: bool = True,
        entry_window_start: float = 210,
        entry_window_end: float = 240,
        min_confidence: float = 0.70,
    ):
        self.state = state
        self.executor = executor
        self.risk_mgr = risk_mgr
        self.metrics = metrics
        self.dashboard = dashboard
        self.paper_trade = paper_trade
        self.entry_window_start = entry_window_start
        self.entry_window_end = entry_window_end
        self.min_confidence = min_confidence

        self.price_history = []
        self.active_position = None
        self.market_start_time = None
        self.running = False

        self.logger = logging.getLogger("strategy.momentum")

    def _mean(self, data: list) -> float:
        return sum(data) / len(data) if data else 0.0

    def _get_polymarket_mid(self) -> Optional[float]:
        ob = getattr(self.state, 'polymarket_orderbook', None)
        if ob and 'bids' in ob and 'asks' in ob:
            best_bid = ob['bids'][0]['price'] if ob['bids'] else 0
            best_ask = ob['asks'][0]['price'] if ob['asks'] else 1
            return (best_bid + best_ask) / 2
        return None

    async def run(self):
        self.logger.info("MomentumScalper starting...")
        self.running = True

        while self.running:
            try:
                mid = self._get_polymarket_mid()
                if mid:
                    self.price_history.append({'timestamp': time.time(), 'price': mid})
                    if len(self.price_history) > 60:
                        self.price_history = self.price_history[-60:]

                if self.market_start_time is None and self.state.market_end_ts:
                    self.market_start_time = self.state.market_end_ts - 300

                if self.market_start_time:
                    elapsed = time.time() - self.market_start_time

                    if self.active_position:
                        await self._manage_position()
                    else:
                        signal = self._generate_signal(elapsed)
                        if signal:
                            await self._enter_position(signal)

                await asyncio.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Error: {e}")
                await asyncio.sleep(1)

    def _generate_signal(self, elapsed: float) -> Optional[Dict]:
        if not (self.entry_window_start <= elapsed <= self.entry_window_end):
            return None

        # Simplified signal generation
        if len(self.price_history) < 5:
            return None

        recent = [p['price'] for p in self.price_history[-5:]]
        momentum = (recent[-1] - recent[0]) / recent[0] if recent[0] else 0

        if abs(momentum) < 0.001:
            return None

        direction = 'UP' if momentum > 0 else 'DOWN'
        confidence = min(1.0, abs(momentum) * 100)

        if confidence < self.min_confidence:
            return None

        current = recent[-1]
        return {
            'direction': direction,
            'confidence': confidence,
            'entry_price': current + (0.002 if direction == 'UP' else -0.002),
            'target': current + 0.03 if direction == 'UP' else current - 0.03,
            'stop': current - 0.05 if direction == 'UP' else current + 0.05,
        }

    async def _enter_position(self, signal: Dict):
        if not self.risk_mgr.can_trade():
            return

        size = self.risk_mgr.calculate_position_size(confidence=signal['confidence'])
        token_id = (self.state.active_token_id if signal['direction'] == 'UP'
                   else self.state.active_token_down)

        self.active_position = {
            'direction': signal['direction'],
            'entry_price': signal['entry_price'],
            'entry_time': time.time(),
            'size': size,
            'target': signal['target'],
            'stop': signal['stop'],
            'token_id': token_id,
        }

        self.dashboard.add_trade({
            'direction': signal['direction'],
            'price': signal['entry_price'],
            'pnl': 0,
            'reason': 'momentum',
            'open': True
        })

        if not self.paper_trade:
            try:
                await self.executor.place_order(
                    token_id=token_id,
                    price=signal['entry_price'],
                    size=size,
                    side='BUY',
                    order_type='FOK'
                )
            except Exception as e:
                self.logger.error(f"Entry failed: {e}")
                self.active_position = None

    async def _manage_position(self):
        if not self.active_position:
            return

        pos = self.active_position
        current = self._get_polymarket_mid()
        if not current:
            return

        if pos['direction'] == 'UP':
            pnl = current - pos['entry_price']
            hit_target = current >= pos['target']
            hit_stop = current <= pos['stop']
        else:
            pnl = pos['entry_price'] - current
            hit_target = current <= pos['target']
            hit_stop = current >= pos['stop']

        hold_time = time.time() - pos['entry_time']

        if hit_target or hit_stop or hold_time > 45:
            reason = 'target' if hit_target else ('stop' if hit_stop else 'timeout')

            self.dashboard.add_trade({
                'direction': pos['direction'],
                'price': current,
                'pnl': pnl,
                'reason': reason,
                'open': False
            })

            self.risk_mgr.update_after_trade(pnl)
            self.active_position = None

    def stop(self):
        self.running = False


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_file_logging(level: str, log_file: str):
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else "logs", exist_ok=True)

    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)

    # Remove console handlers
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)


# ============================================================================
# MAIN
# ============================================================================

def select_strategy():
    print(f"\n{Colors.CYAN}{'═' * 50}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.WHITE}  POLYMARKET SNIPER v2 - STRATEGY SELECTOR{Colors.RESET}")
    print(f"{Colors.CYAN}{'═' * 50}{Colors.RESET}")
    print(f"\n  {Colors.GREEN}1.{Colors.RESET} CLASSIC    - Original taker/maker dual mode")
    print(f"  {Colors.YELLOW}2.{Colors.RESET} MOMENTUM   - Late entry scalping (3:30-4:00)")
    print(f"  {Colors.MAGENTA}3.{Colors.RESET} SMART MONEY- Follow top holders (from polyclaudescraper)")
    print(f"  {Colors.CYAN}4.{Colors.RESET} ALL        - Run all strategies")
    print(f"{Colors.CYAN}{'═' * 50}{Colors.RESET}")

    while True:
        choice = input(f"\n  Select {Colors.BOLD}(1/2/3/4){Colors.RESET}: ").strip()
        if choice in ["1", "2", "3", "4"]:
            break
        print(f"  {Colors.RED}Invalid choice{Colors.RESET}")

    strategies = {"1": "classic", "2": "momentum", "3": "smart_money", "4": "all"}
    return strategies[choice]


async def main():
    selected = select_strategy()
    use_classic = selected in ("classic", "all")
    use_momentum = selected in ("momentum", "all")
    use_smart = selected in ("smart_money", "all")

    # Setup logging
    setup_file_logging(LOG_LEVEL, LOG_FILE)
    logger = logging.getLogger("main")

    # Create dashboard
    dashboard = ColorfulDashboard()

    logger.info("Starting with strategy: " + selected)

    # Initialize components
    state = SharedState()
    finder = MarketFinder(state)
    binance = BinanceListener(state)
    rtds = RTDSListener(state, symbol="btc")
    chainlink = ChainlinkListener(state)
    poly_book = PolyBookListener(state)
    executor = PolyExecutor()
    metrics = MetricsTracker(log_dir="logs", state=state)

    balance = executor.get_balance() if not PAPER_TRADE else 1000.0
    state.current_balance = balance
    risk_mgr = RiskManager(starting_balance=balance)

    # Initialize strategies
    strategies = []

    if use_momentum:
        momentum = MomentumScalper(
            state, executor, risk_mgr, metrics, dashboard, PAPER_TRADE
        )
        strategies.append(momentum)

    if use_smart:
        smart = SmartMoneyStrategy(
            state, executor, risk_mgr, metrics, dashboard, PAPER_TRADE
        )
        strategies.append(smart)

    # Classic mode
    taker = maker = None
    if use_classic:
        taker = TakerSniper(state, executor, risk_mgr, metrics, paper_trade=PAPER_TRADE)
        maker = MakerQuoter(state, executor, metrics, paper_trade=PAPER_TRADE)

    # Fetch initial market
    initial = await finder.fetch_once()
    if initial:
        state.active_token_id = initial["_token_up"]
        state.active_token_down = initial["_token_down"]
        state.active_condition_id = initial.get("conditionId", "")
        state.market_end_ts = initial["_end_ts"]
        state.seconds_remaining = max(0, initial["_end_ts"] - int(time.time()))

    # Tasks
    tasks = [
        asyncio.create_task(finder.start()),
        asyncio.create_task(binance.start()),
        asyncio.create_task(rtds.start()),
        asyncio.create_task(chainlink.start()),
    ]

    # Dashboard updater
    async def dashboard_updater():
        while True:
            signal = dashboard.last_signal if hasattr(dashboard, 'last_signal') else {}
            holders = dashboard.smart_holders if hasattr(dashboard, 'smart_holders') else []
            dashboard.update(state, metrics, signal, holders)
            await asyncio.sleep(0.5)

    tasks.append(asyncio.create_task(dashboard_updater()))

    # Strategy tasks
    for strat in strategies:
        tasks.append(asyncio.create_task(strat.run()))

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        pass
    finally:
        dashboard.close()
        for strat in strategies:
            strat.stop()


if __name__ == "__main__":
    asyncio.run(main())