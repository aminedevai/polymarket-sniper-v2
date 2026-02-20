"""
main.py
=======
Polymarket Sniper v2 - FULL COPY TRADING
Copies ALL positions from target wallet across ALL markets
Target: 0x63ce342161250d705dc0b16df89036c8e5f9ba9a
Profile: https://polymarket.com/@0x8dxd
"""

import asyncio
import logging
import os
import sys
import time
import requests
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from collections import defaultdict

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
PAPER_TRADE = cfg.get("paper_trade", True)
TRADING_MODE = cfg.get("strategy", {}).get("mode", "dual")
LOG_LEVEL = cfg.get("log_level", "INFO")
LOG_FILE = cfg.get("log_file", "logs/bot.log")
LOOP_MS = 0.05

# ============================================================================
# TARGET WALLET CONFIGURATION
# ============================================================================

TARGET_WALLET = "0x63ce342161250d705dc0b16df89036c8e5f9ba9a"
TARGET_PROFILE = "https://polymarket.com/@0x8dxd"
DATA_API = "https://data-api.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"


# ============================================================================
# COLOR SCHEME
# ============================================================================

class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"
    ORANGE = "\033[38;5;208m"


# ============================================================================
# COPY TRADING DATA FETCHER
# ============================================================================

@dataclass
class TargetPosition:
    """Position held by target wallet"""
    condition_id: str
    market_title: str
    outcome: str
    size: float
    avg_price: float
    current_price: float
    pnl: float
    token_id: str
    slug: str
    market_url: str
    last_updated: float

    @property
    def position_value(self) -> float:
        return self.size * self.current_price


@dataclass
class TargetTrade:
    """Recent trade by target wallet"""
    timestamp: str
    market_title: str
    side: str
    outcome: str
    size: float
    price: float
    value: float
    condition_id: str


class CopyTradingFetcher:
    """
    Fetches ALL positions and trades from target wallet using Polymarket Data API.
    Tracks positions across ALL markets, not just current BTC market.
    """

    def __init__(self, wallet_address: str):
        self.wallet = wallet_address
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        })
        self.all_positions: Dict[str, TargetPosition] = {}  # key: condition_id_outcome
        self.position_history: List[Dict] = []
        self.last_fetch = 0
        self.total_value = 0.0
        self.total_pnl = 0.0

    def fetch_all_positions(self) -> Dict[str, TargetPosition]:
        """Fetch ALL current positions of target wallet across ALL markets."""
        try:
            url = f"{DATA_API}/positions"
            params = {
                "user": self.wallet,
                "sizeThreshold": 0.1,  # Include smaller positions
                "limit": 500  # Max limit to get all
            }

            resp = self.session.get(url, params=params, timeout=15)
            data = resp.json()

            new_positions = {}
            self.total_value = 0.0
            self.total_pnl = 0.0

            for pos in data:
                condition_id = pos.get('conditionId', '')
                outcome = pos.get('outcome', 'Unknown')
                key = f"{condition_id}_{outcome}"

                # Get market info
                market_info = self._get_market_info(condition_id)

                position = TargetPosition(
                    condition_id=condition_id,
                    market_title=pos.get('title', 'Unknown Market'),
                    outcome=outcome,
                    size=float(pos.get('size', 0)),
                    avg_price=float(pos.get('avgPrice', 0)),
                    current_price=float(pos.get('price', 0)),
                    pnl=float(pos.get('cashPnl', 0)),
                    token_id=pos.get('assetId', ''),
                    slug=market_info.get('slug', ''),
                    market_url=market_info.get('url', ''),
                    last_updated=time.time()
                )

                new_positions[key] = position
                self.total_value += position.position_value
                self.total_pnl += position.pnl

            # Detect changes
            self._detect_changes(new_positions)

            self.all_positions = new_positions
            self.last_fetch = time.time()

            return self.all_positions

        except Exception as e:
            logging.getLogger("copy_trader").error(f"Fetch all positions error: {e}")
            return self.all_positions

    def _detect_changes(self, new_positions: Dict[str, TargetPosition]):
        """Detect position changes for copy trading signals."""
        # Find new positions
        for key, pos in new_positions.items():
            if key not in self.all_positions:
                # New position opened
                self.position_history.append({
                    'time': time.time(),
                    'action': 'OPEN',
                    'position': pos,
                    'message': f"NEW: {pos.market_title[:40]} | {pos.outcome} | {pos.size:.2f} shares"
                })

        # Find closed positions
        for key, pos in self.all_positions.items():
            if key not in new_positions:
                # Position closed
                self.position_history.append({
                    'time': time.time(),
                    'action': 'CLOSE',
                    'position': pos,
                    'message': f"CLOSED: {pos.market_title[:40]} | {pos.outcome} | PnL: ${pos.pnl:+.2f}"
                })

    def fetch_recent_trades(self, limit: int = 20) -> List[TargetTrade]:
        """Fetch recent trades of target wallet."""
        try:
            url = f"{DATA_API}/trades"
            params = {
                "user": self.wallet,
                "limit": limit,
                "takerOnly": "false"
            }

            resp = self.session.get(url, params=params, timeout=10)
            data = resp.json()

            trades = []
            for trade in data:
                trades.append(TargetTrade(
                    timestamp=trade.get('matchTime', ''),
                    market_title=trade.get('title', 'Unknown'),
                    side=trade.get('side', 'UNKNOWN'),
                    outcome=trade.get('outcome', 'Unknown'),
                    size=float(trade.get('size', 0)),
                    price=float(trade.get('price', 0)),
                    value=float(trade.get('size', 0)) * float(trade.get('price', 0)),
                    condition_id=trade.get('conditionId', '')
                ))

            return trades

        except Exception as e:
            logging.getLogger("copy_trader").error(f"Fetch trades error: {e}")
            return []

    def _get_market_info(self, condition_id: str) -> Dict:
        """Get market slug and URL."""
        try:
            url = f"{GAMMA_API}/markets"
            params = {"condition_id": condition_id}
            resp = self.session.get(url, params=params, timeout=5)
            markets = resp.json()
            if markets:
                slug = markets[0].get('slug', '')
                return {
                    'slug': slug,
                    'url': f"https://polymarket.com/event/{slug}" if slug else ''
                }
        except:
            pass
        return {'slug': '', 'url': ''}

    def get_btc_5min_position(self) -> Optional[TargetPosition]:
        """Get position in current BTC 5-min market if exists."""
        for pos in self.all_positions.values():
            if "btc" in pos.market_title.lower() and "5m" in pos.slug.lower():
                return pos
        return None

    def get_all_market_urls(self) -> List[str]:
        """Get list of all market URLs with positions."""
        urls = []
        for pos in self.all_positions.values():
            if pos.market_url:
                urls.append(pos.market_url)
        return urls


# ============================================================================
# FULL COPY TRADING DASHBOARD
# ============================================================================

class FullCopyDashboard:
    """
    Dashboard showing ALL positions from target wallet across ALL markets.
    """

    def __init__(self):
        self.width = 95
        self.start_time = time.time()
        self.my_trades = []
        self.scroll_offset = 0
        self.selected_tab = 0  # 0: All, 1: BTC Only, 2: My Trades

        self._clear()
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()

    def _clear(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def _print(self, row: int, text: str, col: int = 0):
        sys.stdout.write(f"\033[{row};{col}H{text}")

    def _format_time(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _truncate(self, text: str, length: int) -> str:
        return text[:length - 3] + "..." if len(text) > length else text

    def update(self, state: SharedState, fetcher: CopyTradingFetcher, copy_trader: 'FullCopyTrader'):
        """Render full dashboard with all positions."""
        now = time.time()
        uptime = now - self.start_time

        positions = list(fetcher.all_positions.values())
        history = fetcher.position_history[-10:]  # Last 10 changes

        # Get BTC 5-min specific
        btc_position = fetcher.get_btc_5min_position()
        btc_price = getattr(state, 'binance_btc_price', 0.0)

        lines = []

        # Header
        lines.append(f"{Colors.CYAN}{'═' * self.width}{Colors.RESET}")
        lines.append(
            f"{Colors.BOLD}{Colors.WHITE}  POLYMARKET COPY TRADER - ALL MARKETS{Colors.RESET}  {Colors.GREEN}● LIVE{Colors.RESET}")
        lines.append(f"{Colors.CYAN}{'═' * self.width}{Colors.RESET}")

        # Target Info
        lines.append(f"{Colors.MAGENTA}  TARGET WALLET:{Colors.RESET} {Colors.YELLOW}{TARGET_WALLET}{Colors.RESET}")
        lines.append(f"{Colors.MAGENTA}  PROFILE:{Colors.RESET}       {Colors.BLUE}{TARGET_PROFILE}{Colors.RESET}")
        lines.append(
            f"{Colors.MAGENTA}  MODE:{Colors.RESET}          {Colors.GREEN}PAPER TRADING (Copying ALL positions){Colors.RESET}")
        lines.append("")

        # Portfolio Summary
        lines.append(f"{Colors.GREEN}  ▼ PORTFOLIO SUMMARY{Colors.RESET}")
        lines.append(f"{Colors.GRAY}  {'─' * 70}{Colors.RESET}")
        lines.append(f"  Total Positions: {Colors.WHITE}{len(positions)}{Colors.RESET}")
        lines.append(f"  Total Value:     {Colors.CYAN}${fetcher.total_value:,.2f}{Colors.RESET}")
        lines.append(
            f"  Total P&L:       {Colors.GREEN if fetcher.total_pnl >= 0 else Colors.RED}${fetcher.total_pnl:+.2f}{Colors.RESET}")
        lines.append("")

        # BTC 5-Min Market (Current Focus)
        lines.append(f"{Colors.YELLOW}  ▼ CURRENT BTC 5-MIN MARKET{Colors.RESET}")
        lines.append(f"{Colors.GRAY}  {'─' * 70}{Colors.RESET}")

        if btc_position:
            url = btc_position.market_url
            lines.append(f"  {Colors.CYAN}URL:{Colors.RESET} {url}")
            lines.append(f"  Market:  {Colors.WHITE}{self._truncate(btc_position.market_title, 50)}{Colors.RESET}")
            out_color = Colors.GREEN if btc_position.outcome == "Yes" else Colors.RED
            lines.append(
                f"  Target:  {out_color}{btc_position.outcome}{Colors.RESET} | {btc_position.size:.2f} shares @ {btc_position.current_price:.4f}")
            lines.append(
                f"  BTC Price: ${btc_price:,.2f} | Position PnL: {Colors.GREEN if btc_position.pnl >= 0 else Colors.RED}${btc_position.pnl:+.2f}{Colors.RESET}")
        else:
            lines.append(f"  {Colors.GRAY}No BTC 5-min position found{Colors.RESET}")
            # Show current market URL anyway
            if hasattr(state, 'market_end_ts') and state.market_end_ts:
                timestamp = int(state.market_end_ts)
                url = f"https://polymarket.com/event/btc-updown-5m-{timestamp}"
                lines.append(f"  {Colors.CYAN}Current Market:{Colors.RESET} {url}")

        lines.append("")

        # ALL Positions Table
        lines.append(f"{Colors.YELLOW}  ▼ ALL TARGET POSITIONS (Across All Markets){Colors.RESET}")
        lines.append(f"{Colors.GRAY}  {'─' * 90}{Colors.RESET}")

        if positions:
            # Header
            lines.append(
                f"  {Colors.BOLD}{'Market':<30} {'Outcome':<8} {'Size':>10} {'Price':>8} {'Value':>12} {'PnL':>12} {'URL':>20}{Colors.RESET}")
            lines.append(f"  {Colors.GRAY}{'─' * 90}{Colors.RESET}")

            # Sort by value (largest first)
            sorted_positions = sorted(positions, key=lambda p: p.position_value, reverse=True)

            for pos in sorted_positions[:8]:  # Show top 8
                pnl_color = Colors.GREEN if pos.pnl >= 0 else Colors.RED
                out_color = Colors.GREEN if pos.outcome == "Yes" else Colors.RED
                url_short = pos.slug[:17] + "..." if len(pos.slug) > 20 else pos.slug

                lines.append(f"  {self._truncate(pos.market_title, 29):<30} {out_color}{pos.outcome:<8}{Colors.RESET} "
                             f"{pos.size:>10.2f} {pos.current_price:>8.4f} "
                             f"{Colors.CYAN}${pos.position_value:>11.2f}{Colors.RESET} "
                             f"{pnl_color}${pos.pnl:>+11.2f}{Colors.RESET} "
                             f"{Colors.BLUE}{url_short:>20}{Colors.RESET}")
        else:
            lines.append(f"  {Colors.GRAY}Loading positions...{Colors.RESET}")

        lines.append("")

        # Recent Activity
        lines.append(f"{Colors.ORANGE}  ▼ RECENT ACTIVITY (Position Changes){Colors.RESET}")
        lines.append(f"{Colors.GRAY}  {'─' * 70}{Colors.RESET}")

        if history:
            for item in history[-5:]:
                action_color = Colors.GREEN if item['action'] == 'OPEN' else Colors.RED
                time_str = datetime.fromtimestamp(item['time']).strftime('%H:%M:%S')
                lines.append(f"  [{time_str}] {action_color}{item['action']}{Colors.RESET}: {item['message']}")
        else:
            lines.append(f"  {Colors.GRAY}Waiting for changes...{Colors.RESET}")

        lines.append("")

        # My Copy Trades
        total_my_pnl = sum(t.get('pnl', 0) for t in self.my_trades)
        lines.append(f"{Colors.CYAN}  ▼ MY COPY TRADES{Colors.RESET}")
        lines.append(f"{Colors.GRAY}  {'─' * 70}{Colors.RESET}")
        lines.append(
            f"  Total: {len(self.my_trades)} trades | P&L: {Colors.GREEN if total_my_pnl >= 0 else Colors.RED}${total_my_pnl:+.2f}{Colors.RESET}")

        if self.my_trades:
            for t in self.my_trades[-3:]:
                dir_color = Colors.GREEN if t['direction'] == 'UP' else Colors.RED
                pnl_color = Colors.GREEN if t.get('pnl', 0) >= 0 else Colors.RED
                market = self._truncate(t.get('market', 'Unknown'), 25)
                lines.append(f"  {dir_color}{t['direction']:<6}{Colors.RESET} {market:<25} @ {t.get('price', 0):.4f} "
                             f"PnL: {pnl_color}${t.get('pnl', 0):+.2f}{Colors.RESET} [{t.get('reason', '')}]")
        else:
            lines.append(f"  {Colors.GRAY}No copy trades executed yet...{Colors.RESET}")

        lines.append("")
        lines.append(f"{Colors.CYAN}{'═' * self.width}{Colors.RESET}")
        lines.append(
            f"  {Colors.GRAY}Uptime: {self._format_time(uptime)} | Positions update every 10s | Ctrl+C to stop{Colors.RESET}")

        # Render
        self._clear()
        for i, line in enumerate(lines):
            self._print(i + 1, line, 0)

        sys.stdout.flush()

    def add_trade(self, trade: Dict):
        self.my_trades.append(trade)
        if len(self.my_trades) > 50:
            self.my_trades = self.my_trades[-50:]

    def close(self):
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()


# ============================================================================
# FULL COPY TRADER - ALL MARKETS
# ============================================================================

class FullCopyTrader:
    """
    Copies ALL positions from target wallet across ALL markets.
    Not limited to BTC 5-min markets.
    """

    def __init__(
            self,
            state: SharedState,
            executor: PolyExecutor,
            risk_mgr: RiskManager,
            metrics: MetricsTracker,
            dashboard: FullCopyDashboard,
            paper_trade: bool = True,
            copy_scale: float = 0.1,  # Copy 10% of target size
    ):
        self.state = state
        self.executor = executor
        self.risk_mgr = risk_mgr
        self.metrics = metrics
        self.dashboard = dashboard
        self.paper_trade = paper_trade
        self.copy_scale = copy_scale

        self.fetcher = CopyTradingFetcher(TARGET_WALLET)
        self.status = "INITIALIZING"
        self.running = False

        # Track our copied positions
        self.my_positions: Dict[str, Dict] = {}  # key: condition_id_outcome

        self.logger = logging.getLogger("full_copy_trader")
        self.logger.info(f"FullCopyTrader initialized for {TARGET_WALLET[:20]}...")
        self.logger.info(f"Copy scale: {copy_scale * 100}% of target position size")

    async def run(self):
        """Main loop - copy ALL positions."""
        self.logger.info("FullCopyTrader starting...")
        self.running = True
        self.status = "MONITORING"

        while self.running:
            try:
                # Fetch all target positions
                target_positions = self.fetcher.fetch_all_positions()

                # Copy each position
                for key, target_pos in target_positions.items():
                    if key not in self.my_positions:
                        # New position - OPEN it
                        await self._open_position(target_pos)
                    else:
                        # Check for size changes
                        my_pos = self.my_positions[key]
                        size_diff = abs(target_pos.size - my_pos['target_size'])
                        if size_diff / max(target_pos.size, 1) > 0.2:  # 20% change
                            await self._adjust_position(target_pos, my_pos)

                # Check for closed positions
                for key in list(self.my_positions.keys()):
                    if key not in target_positions:
                        await self._close_position(key)

                self.status = f"TRACKING {len(target_positions)} positions"

                await asyncio.sleep(10)  # Update every 10 seconds

            except Exception as e:
                self.logger.error(f"FullCopyTrader error: {e}")
                self.status = "ERROR"
                await asyncio.sleep(15)

    async def _open_position(self, target_pos: TargetPosition):
        """Open a new copied position."""
        self.status = "OPENING POSITION"

        copy_size = target_pos.size * self.copy_scale

        direction = 'UP' if target_pos.outcome == "Yes" else 'DOWN'

        self.logger.info(f"COPYING: {target_pos.market_title[:40]} | "
                         f"{direction} | {copy_size:.2f} shares @ {target_pos.current_price:.4f}")

        trade_data = {
            'direction': direction,
            'market': target_pos.market_title,
            'price': target_pos.current_price,
            'size': copy_size,
            'target_size': target_pos.size,  # Track original
            'pnl': 0,
            'open': True,
            'reason': f"copy_{TARGET_WALLET[:6]}",
            'condition_id': target_pos.condition_id,
            'outcome': target_pos.outcome,
            'token_id': target_pos.token_id,
            'market_url': target_pos.market_url
        }

        key = f"{target_pos.condition_id}_{target_pos.outcome}"
        self.my_positions[key] = trade_data
        self.dashboard.add_trade(trade_data)

        if not self.paper_trade:
            try:
                # Determine token ID based on outcome
                token_id = target_pos.token_id

                await self.executor.place_order(
                    token_id=token_id,
                    price=target_pos.current_price,
                    size=copy_size,
                    side='BUY',
                    order_type='FOK'
                )
                self.logger.info(f"Order executed: {token_id[:20]}...")
            except Exception as e:
                self.logger.error(f"Order failed: {e}")

        self.status = "MONITORING"

    async def _adjust_position(self, target_pos: TargetPosition, my_pos: Dict):
        """Adjust position size to match target."""
        new_size = target_pos.size * self.copy_scale
        old_size = my_pos['size']

        self.logger.info(f"ADJUSTING: {target_pos.market_title[:40]} | "
                         f"{old_size:.2f} -> {new_size:.2f}")

        my_pos['target_size'] = target_pos.size
        my_pos['size'] = new_size
        my_pos['price'] = target_pos.current_price

    async def _close_position(self, key: str):
        """Close a position when target closes."""
        if key not in self.my_positions:
            return

        my_pos = self.my_positions[key]

        # Calculate PnL
        current_price = my_pos['price']  # Simplified
        if my_pos['direction'] == 'UP':
            pnl = (current_price - my_pos['price']) * my_pos['size']
        else:
            pnl = (my_pos['price'] - current_price) * my_pos['size']

        self.logger.info(f"CLOSING: {my_pos['market'][:40]} | PnL: ${pnl:.2f}")

        my_pos['pnl'] = pnl
        my_pos['open'] = False
        my_pos['reason'] = 'target_closed'

        self.dashboard.add_trade(my_pos)
        self.risk_mgr.update_after_trade(pnl)

        if not self.paper_trade:
            try:
                await self.executor.place_order(
                    token_id=my_pos['token_id'],
                    price=current_price,
                    size=my_pos['size'],
                    side='SELL',
                    order_type='IOC'
                )
            except Exception as e:
                self.logger.error(f"Close order failed: {e}")

        del self.my_positions[key]

    def stop(self):
        self.running = False

    def get_all_market_urls(self) -> List[str]:
        """Get all market URLs being tracked."""
        return self.fetcher.get_all_market_urls()


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

    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)


# ============================================================================
# MAIN
# ============================================================================

def select_mode():
    print(f"\n{Colors.CYAN}{'═' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.WHITE}  POLYMARKET FULL COPY TRADER{Colors.RESET}")
    print(f"{Colors.CYAN}{'═' * 60}{Colors.RESET}")
    print(f"\n  {Colors.GREEN}1.{Colors.RESET} FULL COPY   - Copy ALL positions from wallet")
    print(f"  {Colors.YELLOW}2.{Colors.RESET} BTC FOCUS   - Copy only BTC 5-min markets")
    print(f"  {Colors.BLUE}3.{Colors.RESET} DASHBOARD   - View only (no copying)")
    print(f"{Colors.CYAN}{'═' * 60}{Colors.RESET}")

    while True:
        choice = input(f"\n  Select mode {Colors.BOLD}(1/2/3){Colors.RESET}: ").strip()
        if choice in ["1", "2", "3"]:
            break
        print(f"  {Colors.RED}Invalid choice{Colors.RESET}")

    modes = {"1": "full", "2": "btc", "3": "view"}
    return modes[choice]


async def main():
    selected_mode = select_mode()

    # Setup logging
    setup_file_logging(LOG_LEVEL, LOG_FILE)
    logger = logging.getLogger("main")

    # Create dashboard
    dashboard = FullCopyDashboard()

    logger.info(f"Starting in {selected_mode} mode")
    print(f"\n{Colors.GREEN}Starting copy trader...{Colors.RESET}")
    print(f"{Colors.GRAY}Target: {TARGET_WALLET}{Colors.RESET}")
    print(f"{Colors.GRAY}Profile: {TARGET_PROFILE}{Colors.RESET}\n")

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

    # Initialize copy trader
    copy_trader = FullCopyTrader(
        state, executor, risk_mgr, metrics, dashboard,
        paper_trade=PAPER_TRADE,
        copy_scale=0.1  # 10% of target size
    )

    # Fetch initial market for BTC price context
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
            dashboard.update(state, copy_trader.fetcher, copy_trader)
            await asyncio.sleep(0.5)

    tasks.append(asyncio.create_task(dashboard_updater()))

    # Copy trader task
    if selected_mode in ["full", "btc"]:
        tasks.append(asyncio.create_task(copy_trader.run()))

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        pass
    finally:
        dashboard.close()
        copy_trader.stop()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())