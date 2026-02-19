"""
risk/position_manager.py
========================
Kelly Criterion position sizing + drawdown-based kill switch.
Uses quarter-Kelly for conservatism.
Hard cap per trade at MAX_POSITION_USDC.
"""

import logging

logger = logging.getLogger(__name__)


class RiskManager:
    MAX_POSITION_USDC = 50.0       # Hard cap per trade
    MAX_DRAWDOWN_PCT = 0.20        # Stop bot if down 20% from peak
    KELLY_FRACTION = 0.25          # Quarter-Kelly
    MIN_TRADE_USDC = 1.0           # Don't bother below $1

    def __init__(self, starting_balance: float):
        self.starting_balance = starting_balance
        self.peak_balance = starting_balance
        self.current_balance = starting_balance
        self._halted = False

    def kelly_size(
        self,
        win_probability: float,
        price: float,
        available_usdc: float,
    ) -> float:
        """
        Kelly formula for binary outcome:
          f* = (p × b - q) / b
          where b = net odds = (1/price - 1), q = 1 - p

        Returns recommended USDC trade size (after fractional Kelly + cap).
        Returns 0 if bet is -EV or too small.
        """
        if price <= 0.001 or price >= 0.999:
            return 0.0
        if win_probability <= 0 or win_probability >= 1:
            return 0.0

        b = (1.0 / price) - 1.0   # net odds (e.g. price=0.80 → b=0.25)
        q = 1.0 - win_probability
        kelly = (win_probability * b - q) / b
        kelly = max(0.0, kelly)

        if kelly == 0:
            return 0.0

        bet_fraction = self.KELLY_FRACTION * kelly
        bet_usdc = bet_fraction * available_usdc
        capped = min(bet_usdc, self.MAX_POSITION_USDC)

        if capped < self.MIN_TRADE_USDC:
            return 0.0

        return round(capped, 2)

    def update_balance(self, new_balance: float):
        self.current_balance = new_balance
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance

        # Check drawdown
        if self.peak_balance > 0:
            drawdown = (self.peak_balance - new_balance) / self.peak_balance
            if drawdown >= self.MAX_DRAWDOWN_PCT and not self._halted:
                logger.critical(
                    f"DRAWDOWN LIMIT BREACHED: {drawdown:.1%} from peak "
                    f"${self.peak_balance:.2f} → ${new_balance:.2f}. BOT HALTED."
                )
                self._halted = True

    def is_active(self) -> bool:
        return not self._halted

    def reset_halt(self):
        """Manual override — use with caution."""
        logger.warning("Risk halt manually reset.")
        self._halted = False
        self.peak_balance = self.current_balance

    @property
    def current_drawdown(self) -> float:
        if self.peak_balance == 0:
            return 0.0
        return (self.peak_balance - self.current_balance) / self.peak_balance

    def __repr__(self):
        return (
            f"RiskManager(balance=${self.current_balance:.2f} "
            f"peak=${self.peak_balance:.2f} "
            f"dd={self.current_drawdown:.1%} "
            f"halted={self._halted})"
        )
