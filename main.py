"""
main.py
=======
Polymarket Copy Trader with Bet URLs and Countdown Timers
"""

import asyncio
import logging
import os
import sys
import time
import requests
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
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
# CONFIGURATION
# ============================================================================

TARGET_WALLET = "0x63ce342161250d705dc0b16df89036c8e5f9ba9a"
TARGET_PROFILE = "https://polymarket.com/@0x8dxd"
DATA_API = "https://data-api.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"

PAPER_BUDGET = 100.0
COPY_SCALE = 0.5


# ============================================================================
# COLORS
# ============================================================================

class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
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
# HELPER FUNCTIONS
# ============================================================================

def format_time_remaining(end_date_str: str) -> str:
    """
    Format time remaining until market ends.
    Returns: "2h 15m" or "45m" or "5m 30s" or "ENDED"
    """
    if not end_date_str:
        return "Unknown"

    try:
        # Parse end date (ISO format)
        end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)

        if end_date <= now:
            return f"{Colors.RED}ENDED{Colors.RESET}"

        diff = end_date - now
        total_seconds = int(diff.total_seconds())

        days = total_seconds // 86400
        hours = (total_seconds % 86400) // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        # Format based on time left
        if days > 0:
            return f"{Colors.YELLOW}{days}d {hours}h{Colors.RESET}"
        elif hours > 0:
            return f"{Colors.YELLOW}{hours}h {minutes}m{Colors.RESET}"
        elif minutes > 0:
            return f"{Colors.ORANGE}{minutes}m {seconds}s{Colors.RESET}"
        else:
            return f"{Colors.RED}{seconds}s{Colors.RESET}"

    except Exception:
        return "Unknown"


def get_market_url(slug: str) -> str:
    """Generate Polymarket URL from slug."""
    if not slug:
        return ""
    return f"https://polymarket.com/event/{slug}"


# ============================================================================
# TRADE TRACKING WITH URL AND TIMER
# ============================================================================

@dataclass
class Trade:
    """Trade with URL and end date tracking."""
    trade_id: str
    market_title: str
    outcome: str
    slug: str

    entry_price: float
    entry_shares: float
    entry_amount: float

    exit_price: float = 0.0
    exit_amount: float = 0.0
    realized_pnl: float = 0.0

    is_open: bool = True
    open_time: float = 0.0
    close_time: Optional[float] = None
    end_date: str = ""  # ISO format end date

    @property
    def market_url(self) -> str:
        return get_market_url(self.slug)

    @property
    def time_remaining(self) -> str:
        return format_time_remaining(self.end_date)

    def calculate_pnl(self) -> float:
        if self.is_open:
            return (self.exit_price - self.entry_price) * self.entry_shares
        else:
            return self.realized_pnl


class BudgetCopyTrader:
    """Copy trader with URL and timer support."""

    def __init__(self, wallet_address: str, budget: float = PAPER_BUDGET, copy_scale: float = COPY_SCALE):
        self.wallet = wallet_address
        self.budget = budget
        self.copy_scale = copy_scale

        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        })

        # Budget tracking
        self.invested = 0.0
        self.available = budget
        self.returned = 0.0
        self.realized_profit = 0.0

        # Positions and trades
        self.target_positions: Dict[str, Dict] = {}
        self.my_positions: Dict[str, Dict] = {}
        self.all_trades: List[Trade] = []
        self.open_trades: Dict[str, Trade] = {}

        self.logger = logging.getLogger("budget_trader")

    def fetch_target_positions(self) -> Dict[str, Dict]:
        """Fetch positions with slug and end date."""
        try:
            url = f"{DATA_API}/positions"
            params = {
                "user": self.wallet,
                "sizeThreshold": 0.1,
                "limit": 500,
            }

            resp = self.session.get(url, params=params, timeout=15)
            data = resp.json()

            positions = {}
            for pos in data:
                condition_id = pos.get('conditionId', '')
                outcome = pos.get('outcome', 'Unknown')
                key = f"{condition_id}_{outcome}"

                avg_price = float(pos.get('avgPrice', 0) or 0)
                shares = float(pos.get('size', 0) or 0)
                slug = pos.get('slug', '')
                end_date = pos.get('endDate', '')

                positions[key] = {
                    'condition_id': condition_id,
                    'market_title': pos.get('title', 'Unknown Market'),
                    'outcome': outcome,
                    'avg_price': avg_price,
                    'cur_price': float(pos.get('curPrice', 0) or 0),
                    'shares': shares,
                    'entry_amount': avg_price * shares,
                    'current_value': float(pos.get('currentValue', 0) or 0),
                    'slug': slug,
                    'market_url': get_market_url(slug),
                    'end_date': end_date,
                }

            return positions

        except Exception as e:
            self.logger.error(f"Fetch error: {e}")
            return self.target_positions

    def sync_positions(self):
        """Sync with budget constraints."""
        actions = []

        new_target = self.fetch_target_positions()

        # OPEN new positions
        for key, target_pos in new_target.items():
            if key not in self.my_positions and key not in self.target_positions:
                needed = target_pos['entry_amount'] * self.copy_scale

                if needed <= self.available:
                    action = self._open_position(target_pos, needed)
                    actions.append(action)
                else:
                    actions.append(f"SKIP: {target_pos['market_title'][:25]} | "
                                   f"Need ${needed:.2f}, have ${self.available:.2f}")

        # UPDATE existing
        for key, target_pos in new_target.items():
            if key in self.my_positions:
                self.my_positions[key]['cur_price'] = target_pos['cur_price']
                self.my_positions[key]['current_value'] = target_pos['current_value'] * self.copy_scale

                if key in self.open_trades:
                    self.open_trades[key].exit_price = target_pos['cur_price']

        # CLOSE positions
        for key in list(self.my_positions.keys()):
            if key not in new_target and key in self.target_positions:
                action = self._close_position(key)
                actions.append(action)

        self.target_positions = new_target
        return actions

    def _open_position(self, target_pos: Dict, amount: float) -> str:
        """Open position with URL and timer."""
        our_shares = target_pos['shares'] * self.copy_scale

        # Deduct from budget
        self.available -= amount
        self.invested += amount

        position = {
            'condition_id': target_pos['condition_id'],
            'market_title': target_pos['market_title'],
            'outcome': target_pos['outcome'],
            'avg_price': target_pos['avg_price'],
            'cur_price': target_pos['cur_price'],
            'shares': our_shares,
            'entry_amount': amount,
            'current_value': target_pos['current_value'] * self.copy_scale,
            'slug': target_pos['slug'],
            'market_url': target_pos['market_url'],
            'end_date': target_pos['end_date'],
        }

        key = f"{target_pos['condition_id']}_{target_pos['outcome']}"
        self.my_positions[key] = position

        # Track trade with URL and end date
        trade = Trade(
            trade_id=key,
            market_title=target_pos['market_title'],
            outcome=target_pos['outcome'],
            slug=target_pos['slug'],
            entry_price=target_pos['avg_price'],
            entry_shares=our_shares,
            entry_amount=amount,
            exit_price=target_pos['cur_price'],
            is_open=True,
            open_time=time.time(),
            end_date=target_pos['end_date']
        )
        self.open_trades[key] = trade
        self.all_trades.append(trade)

        # Format time remaining for display
        time_left = format_time_remaining(target_pos['end_date'])

        self.logger.info(f"OPENED: ${amount:.2f} | {position['market_title'][:40]} | "
                         f"Time left: {time_left}")

        return (f"OPEN: ${amount:.2f} | {position['market_title'][:25]} | "
                f"Time: {time_left}")

    def _close_position(self, key: str) -> str:
        """Close position."""
        if key not in self.my_positions:
            return ""

        pos = self.my_positions[key]

        # Calculate return
        exit_value = pos['cur_price'] * pos['shares']
        profit = exit_value - pos['entry_amount']

        # Return to budget
        self.invested -= pos['entry_amount']
        self.available += exit_value
        self.returned += exit_value
        self.realized_profit += profit

        # Update trade
        if key in self.open_trades:
            trade = self.open_trades[key]
            trade.is_open = False
            trade.close_time = time.time()
            trade.exit_amount = exit_value
            trade.realized_pnl = profit

        del self.open_trades[key]
        del self.my_positions[key]

        return (f"CLOSE: ${pos['entry_amount']:.2f} → ${exit_value:.2f} | "
                f"Profit: ${profit:+.2f}")

    def get_budget_summary(self) -> Dict:
        """Get budget summary."""
        unrealized = sum(
            (p['cur_price'] - p['avg_price']) * p['shares']
            for p in self.my_positions.values()
        )

        return {
            'budget': self.budget,
            'invested': self.invested,
            'available': self.available,
            'returned': self.returned,
            'realized_profit': self.realized_profit,
            'unrealized_pnl': unrealized,
            'total_value': self.available + sum(p['current_value'] for p in self.my_positions.values()),
            'positions': len(self.my_positions),
        }


# ============================================================================
# DASHBOARD WITH URLS AND TIMERS
# ============================================================================

class BudgetDashboard:
    """Dashboard with bet URLs and countdown timers."""

    def __init__(self):
        self.width = 110  # Wider for URLs
        self.start_time = time.time()
        self.last_actions = []

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

    def _bar(self, current: float, max_val: float, width: int = 30) -> str:
        if max_val == 0:
            return "░" * width
        filled = int((current / max_val) * width)
        return "█" * filled + "░" * (width - filled)

    def _truncate_url(self, url: str, max_len: int = 40) -> str:
        """Truncate URL for display."""
        if len(url) <= max_len:
            return url
        return url[:max_len - 3] + "..."

    def update(self, trader: BudgetCopyTrader):
        """Render dashboard with URLs and timers."""
        now = time.time()
        uptime = now - self.start_time

        summary = trader.get_budget_summary()
        my_positions = list(trader.my_positions.values())
        closed_trades = [t for t in trader.all_trades if not t.is_open][-5:]

        lines = []

        # Header
        lines.append(f"{Colors.CYAN}{'═' * self.width}{Colors.RESET}")
        lines.append(
            f"{Colors.BOLD}{Colors.WHITE}  POLYMARKET COPY TRADER - $100 BUDGET + URLs & TIMERS{Colors.RESET}  {Colors.GREEN}● LIVE{Colors.RESET}")
        lines.append(f"{Colors.CYAN}{'═' * self.width}{Colors.RESET}")
        lines.append("")

        # BUDGET BREAKDOWN
        lines.append(f"{Colors.GREEN}{Colors.BOLD}  ▼ YOUR $100 BUDGET{Colors.RESET}")
        lines.append(f"{Colors.GRAY}  {'─' * 80}{Colors.RESET}")

        invested_pct = (summary['invested'] / summary['budget']) * 100
        bar = self._bar(summary['invested'], summary['budget'])

        lines.append(f"  Total Budget:      {Colors.BOLD}{Colors.WHITE}${summary['budget']:.2f}{Colors.RESET}")
        lines.append(f"")
        lines.append(
            f"  {Colors.YELLOW}INVESTED:{Colors.RESET}          {Colors.YELLOW}${summary['invested']:.2f}{Colors.RESET} ({invested_pct:.1f}%)  {bar}")
        lines.append(f"  {Colors.GREEN}AVAILABLE:         ${summary['available']:.2f}{Colors.RESET}")
        lines.append(f"  {Colors.BLUE}CASH RETURNED:     ${summary['returned']:.2f}{Colors.RESET}")
        lines.append(f"")

        realized_color = Colors.GREEN if summary['realized_profit'] >= 0 else Colors.RED
        unreal_color = Colors.GREEN if summary['unrealized_pnl'] >= 0 else Colors.RED

        lines.append(f"  Realized Profit:   {realized_color}${summary['realized_profit']:+.2f}{Colors.RESET}")
        lines.append(f"  Unrealized P&L:    {unreal_color}${summary['unrealized_pnl']:+.2f}{Colors.RESET}")
        lines.append(f"  Portfolio Value:   {Colors.CYAN}${summary['total_value']:.2f}{Colors.RESET}")
        lines.append("")

        # Target Info
        lines.append(f"{Colors.MAGENTA}  ▼ TARGET WALLET{Colors.RESET}")
        lines.append(f"{Colors.GRAY}  {'─' * 80}{Colors.RESET}")
        lines.append(f"  Wallet:  {Colors.YELLOW}{TARGET_WALLET[:25]}...{Colors.RESET}")
        lines.append(f"  Profile: {Colors.BLUE}{TARGET_PROFILE}{Colors.RESET}")
        lines.append(f"  Copy Scale: {Colors.YELLOW}50%{Colors.RESET}")
        lines.append("")

        # OPEN POSITIONS WITH URLS AND TIMERS
        lines.append(f"{Colors.YELLOW}  ▼ YOUR POSITIONS (With Bet URLs & Time Remaining){Colors.RESET}")
        lines.append(f"{Colors.GRAY}  {'─' * 105}{Colors.RESET}")

        if my_positions:
            # Header
            lines.append(
                f"  {Colors.BOLD}{'Market':<25} {'Inv $':>8} {'P&L $':>9} {'Time Left':>12} {'Bet URL':<45}{Colors.RESET}")
            lines.append(f"  {Colors.GRAY}{'─' * 105}{Colors.RESET}")

            for pos in sorted(my_positions, key=lambda p: p['entry_amount'], reverse=True)[:6]:
                pnl = (pos['cur_price'] - pos['avg_price']) * pos['shares']
                pnl_color = Colors.GREEN if pnl >= 0 else Colors.RED

                market_name = pos['market_title'][:24] if len(pos['market_title']) <= 25 else pos['market_title'][
                                                                                                  :22] + ".."
                time_left = format_time_remaining(pos['end_date'])
                url = self._truncate_url(pos['market_url'], 45)

                lines.append(f"  {market_name:<25} "
                             f"{Colors.YELLOW}${pos['entry_amount']:>7.2f}{Colors.RESET} "
                             f"{pnl_color}${pnl:>+8.2f}{Colors.RESET} "
                             f"{time_left:>12} "
                             f"{Colors.BLUE}{url}{Colors.RESET}")
        else:
            lines.append(f"  {Colors.GRAY}No positions - $100 available{Colors.RESET}")

        lines.append("")

        # CLOSED TRADES
        lines.append(f"{Colors.GREEN}  ▼ CLOSED TRADES (Completed Bets){Colors.RESET}")
        lines.append(f"{Colors.GRAY}  {'─' * 105}{Colors.RESET}")

        if closed_trades:
            lines.append(
                f"  {Colors.BOLD}{'Market':<30} {'Inv $':>8} {'Ret $':>8} {'Profit':>9} {'Bet URL':<50}{Colors.RESET}")
            lines.append(f"  {Colors.GRAY}{'─' * 105}{Colors.RESET}")

            for trade in reversed(closed_trades):
                pnl_color = Colors.GREEN if trade.realized_pnl >= 0 else Colors.RED
                # FIXED LINE HERE - removed extra quote
                market_name = trade.market_title[:29] if len(trade.market_title) <= 30 else trade.market_title[
                                                                                                :27] + ".."
                url = self._truncate_url(trade.market_url, 50)

                lines.append(f"  {market_name:<30} "
                             f"{Colors.YELLOW}${trade.entry_amount:>7.2f}{Colors.RESET} "
                             f"{Colors.CYAN}${trade.exit_amount:>7.2f}{Colors.RESET} "
                             f"{pnl_color}${trade.realized_pnl:>+8.2f}{Colors.RESET} "
                             f"{Colors.BLUE}{url}{Colors.RESET}")
        else:
            lines.append(f"  {Colors.GRAY}No closed trades yet{Colors.RESET}")

                lines.append("")
                # Recent Activity
                lines.append(f"{Colors.ORANGE}  ▼ RECENT ACTIVITY{Colors.RESET}")
                lines.append(f"{Colors.GRAY}  {'─' * 80}{Colors.RESET}")

                if self.last_actions:
                    for
                action in self.last_actions[-5:]:
                if "OPEN" in action:
                    color = Colors.GREEN
                elif "CLOSE" in action:
                    color = Colors.RED
                elif "SKIP" in action:
                    color = Colors.YELLOW
                else:
                    color = Colors.WHITE
                lines.append(f"  {color}{action}{Colors.RESET}")
                else:
                lines.append(f"  {Colors.GRAY}No activity{Colors.RESET}")

                lines.append("")
                lines.append(f"{Colors.CYAN}{'═' * self.width}{Colors.RESET}")
                lines.append(
                    f"  {Colors.GRAY}Uptime: {self._format_time(uptime)} | Updates every 10s | Ctrl+C to stop{Colors.RESET}")

                # Render
                self._clear()
                for i, line in enumerate(lines):
                    self._print(i + 1, line, 0)

                sys.stdout.flush()

    def add_action(self, action: str):
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.last_actions.append(f"[{timestamp}] {action}")
        if len(self.last_actions) > 10:
            self.last_actions = self.last_actions[-10:]

    def close(self):
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(level: str, log_file: str):
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

async def main():
    setup_logging(LOG_LEVEL, LOG_FILE)
    logger = logging.getLogger("main")

    print(f"\n{Colors.CYAN}{'═' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.WHITE}  POLYMARKET COPY TRADER - URLs & TIMERS{Colors.RESET}")
    print(f"{Colors.CYAN}{'═' * 70}{Colors.RESET}")
    print(f"\n  {Colors.GREEN}PAPER TRADING MODE{Colors.RESET} | Budget: {Colors.BOLD}$100.00{Colors.RESET}")
    print(f"  Copy Scale: {Colors.YELLOW}50%{Colors.RESET}")
    print(f"")
    print(f"  {Colors.GRAY}Features:{Colors.RESET}")
    print(f"  • Shows Polymarket bet URL for each position")
    print(f"  • Countdown timer showing time until market ends")
    print(f"  • Tracks your $100 budget in real-time")
    print(f"\n{Colors.CYAN}{'═' * 70}{Colors.RESET}\n")

    dashboard = BudgetDashboard()
    trader = BudgetCopyTrader(TARGET_WALLET, budget=PAPER_BUDGET, copy_scale=COPY_SCALE)

    print(f"{Colors.YELLOW}Fetching target positions...{Colors.RESET}")
    trader.fetch_target_positions()
    print(f"  Target has {Colors.WHITE}{len(trader.target_positions)}{Colors.RESET} positions")

    print(f"\n{Colors.YELLOW}Copying positions with URLs and timers...{Colors.RESET}")
    actions = trader.sync_positions()
    for action in actions:
        dashboard.add_action(action)
        print(f"  {action}")

    print(f"\n{Colors.GREEN}Budget: ${trader.available:.2f} available / $100.00 total{Colors.RESET}")
    print(f"\n{Colors.GREEN}Starting monitoring...{Colors.RESET}\n")

    try:
        while True:
            actions = trader.sync_positions()
            for action in actions:
                dashboard.add_action(action)

            dashboard.update(trader)

            await asyncio.sleep(10)

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Shutting down...{Colors.RESET}")
    finally:
        dashboard.close()

        summary = trader.get_budget_summary()
        print(f"\n{Colors.CYAN}{'═' * 70}{Colors.RESET}")
        print(f"{Colors.BOLD}FINAL BUDGET REPORT{Colors.RESET}")
        print(f"{Colors.CYAN}{'═' * 70}{Colors.RESET}")
        print(f"Started with:      ${summary['budget']:.2f}")
        print(f"Currently Invested: ${summary['invested']:.2f}")
        print(f"Available Cash:    ${summary['available']:.2f}")
        print(
            f"Realized Profit:   {Colors.GREEN if summary['realized_profit'] >= 0 else Colors.RED}${summary['realized_profit']:+.2f}{Colors.RESET}")
        print(
            f"Unrealized P&L:    {Colors.GREEN if summary['unrealized_pnl'] >= 0 else Colors.RED}${summary['unrealized_pnl']:+.2f}{Colors.RESET}")
        print(f"Portfolio Value:   {Colors.CYAN}${summary['total_value']:.2f}{Colors.RESET}")
        print(f"{Colors.CYAN}{'═' * 70}{Colors.RESET}")


if __name__ == "__main__":
    asyncio.run(main())