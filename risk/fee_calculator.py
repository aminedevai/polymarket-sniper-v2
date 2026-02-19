"""
risk/fee_calculator.py
======================
Implements the official Polymarket fee curve (confirmed Feb 2026):
  fee = C × 0.25 × (p × (1 - p))²

Key insight: fees peak near p=0.50 (~3.1%) and fall near zero at extremes.
Strategy should target p < 0.25 or p > 0.75 to minimize fee drag.
"""


class FeeCalculator:
    FEE_RATE = 0.25
    EXPONENT = 2

    @classmethod
    def taker_fee(cls, price: float, shares: float) -> float:
        """Total USDC fee for a taker order of `shares` at `price`."""
        p = max(0.001, min(0.999, price))
        return shares * cls.FEE_RATE * (p * (1 - p)) ** cls.EXPONENT

    @classmethod
    def effective_fee_rate(cls, price: float) -> float:
        """Fee as fraction of capital deployed (C × p)."""
        p = max(0.001, min(0.999, price))
        fee_per_share = cls.FEE_RATE * (p * (1 - p)) ** cls.EXPONENT
        return fee_per_share / p

    @classmethod
    def required_win_probability(cls, price: float) -> float:
        """
        Minimum true win probability to break even as a taker.
        Assumes held to expiry (no exit fee).
        break_even: P_win × 1.0 - fee_per_share = price
        ⟹ P_win = price + fee_per_share
        """
        p = max(0.001, min(0.999, price))
        fee = cls.taker_fee(p, 1.0)
        return p + fee

    @classmethod
    def is_tradeable_zone(cls, price: float, max_fee_rate: float = 0.015) -> bool:
        """True if effective fee rate is below threshold."""
        return cls.effective_fee_rate(price) <= max_fee_rate

    @classmethod
    def expected_value(
        cls,
        price: float,
        true_win_prob: float,
        shares: float = 1.0,
    ) -> float:
        """
        Expected USDC profit for one taker buy of `shares` at `price`.
        EV = true_win_prob × (1 - price) × shares
             - (1 - true_win_prob) × price × shares
             - taker_fee
        """
        p = max(0.001, min(0.999, price))
        fee = cls.taker_fee(p, shares)
        win_pnl = true_win_prob * (1.0 - p) * shares
        loss_pnl = (1.0 - true_win_prob) * p * shares
        return win_pnl - loss_pnl - fee

    @classmethod
    def print_fee_table(cls):
        """Debug helper — prints the fee table from the v2.0 doc."""
        print(f"{'Price':>8} {'Fee/share':>12} {'Eff Rate':>10} {'Min Win%':>10} {'Zone':>8}")
        print("-" * 55)
        for p_cent in [10, 15, 20, 25, 30, 40, 50, 65, 75, 80, 85, 90, 95]:
            p = p_cent / 100
            fee = cls.taker_fee(p, 1.0)
            eff = cls.effective_fee_rate(p)
            min_win = cls.required_win_probability(p)
            zone = "LOW" if eff < 0.012 else ("MID" if eff < 0.025 else "HIGH")
            tradeable = "✓" if cls.is_tradeable_zone(p) else "✗"
            print(
                f"  ${p:.2f}   ${fee:.6f}   {eff:>8.2%}   {min_win:>8.2%}"
                f"   {zone} {tradeable}"
            )
