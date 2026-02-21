"""
strategy/signal_time.py
=======================
Time-based signal for 5-min BTC copy trader.
Based on backtest findings:
  - Bearish bias: 10:00 UTC, 14:00 UTC
  - Bullish bias: 13:00, 17:00, 18:00 UTC
  - Extreme price fade: UP >95% → bet DOWN
"""

from datetime import datetime, timezone

# ── Signal config (adjust after more backtesting) ─────────────────────────────

# Hours UTC where DOWN wins more often than expected
BEARISH_HOURS = {10, 14}

# Hours UTC where UP wins more often than expected
BULLISH_HOURS = {13, 17, 18}

# If UP is priced above this, consider fading (betting DOWN)
EXTREME_BULLISH_FADE_THRESHOLD = 0.95

# If UP is priced below this, consider fading (betting UP)
EXTREME_BEARISH_FADE_THRESHOLD = 0.05

# Minimum confidence to generate a signal (0.0 - 1.0)
MIN_CONFIDENCE = 0.60


def get_signal(up_price: float) -> dict:
    """
    Returns a trading signal based on time + price.

    Returns:
        {
          'action':     'UP' | 'DOWN' | 'SKIP',
          'confidence': float,
          'reasons':    list[str],
        }
    """
    now      = datetime.now(timezone.utc)
    hour     = now.hour
    minute   = now.minute
    reasons  = []
    score    = 0.0   # positive = lean UP, negative = lean DOWN

    # ── Time-based signals ────────────────────────────────────────────────────
    if hour in BEARISH_HOURS:
        score   -= 0.30
        reasons.append(f"bearish hour {hour:02d}:00 UTC (historical DOWN bias)")
    elif hour in BULLISH_HOURS:
        score   += 0.25
        reasons.append(f"bullish hour {hour:02d}:00 UTC (historical UP bias)")

    # US market open: first 30 min of 14:30 UTC is especially volatile/bearish
    if hour == 14 and minute < 30:
        score   -= 0.20
        reasons.append("US market open window (14:00-14:30 UTC) extra bearish")

    # ── Extreme price fade signals ────────────────────────────────────────────
    if up_price >= EXTREME_BULLISH_FADE_THRESHOLD:
        score   -= 0.40
        reasons.append(f"UP overpriced at {up_price:.0%} (fade DOWN, historical -34% edge)")

    elif up_price <= EXTREME_BEARISH_FADE_THRESHOLD:
        score   += 0.40
        reasons.append(f"DOWN overpriced at {1-up_price:.0%} (fade UP, historical edge)")

    # ── Strong price + time agreement ────────────────────────────────────────
    if up_price >= 0.85 and hour in BULLISH_HOURS:
        score   += 0.15
        reasons.append("strong UP price + bullish hour = ride momentum")

    if up_price <= 0.15 and hour in BEARISH_HOURS:
        score   -= 0.15
        reasons.append("strong DOWN price + bearish hour = ride momentum")

    # ── Decision ─────────────────────────────────────────────────────────────
    confidence = min(abs(score), 1.0)

    if score < 0 and confidence >= MIN_CONFIDENCE:
        action = 'DOWN'
    elif score > 0 and confidence >= MIN_CONFIDENCE:
        action = 'UP'
    else:
        action = 'SKIP'
        reasons.append(f"confidence {confidence:.0%} below threshold {MIN_CONFIDENCE:.0%}")

    return {
        'action':     action,
        'confidence': confidence,
        'score':      score,
        'reasons':    reasons,
        'hour_utc':   hour,
        'up_price':   up_price,
    }


def format_signal(sig: dict) -> str:
    """Pretty-print a signal for display."""
    a  = sig['action']
    c  = sig['confidence']
    col = "\033[92m" if a == 'UP' else "\033[91m" if a == 'DOWN' else "\033[90m"
    R  = "\033[0m"
    lines = [
        f"  {col}SIGNAL: {a}  (confidence {c:.0%}){R}",
        f"  UP price: {sig['up_price']:.1%}  Hour: {sig['hour_utc']:02d}:00 UTC",
    ]
    for r in sig['reasons']:
        lines.append(f"    → {r}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick test
    print("=== Signal Engine Test ===\n")
    test_cases = [
        (0.97, "Extreme bullish price"),
        (0.87, "Strong bullish price"),
        (0.50, "50/50 price"),
        (0.03, "Extreme bearish price"),
    ]
    for price, label in test_cases:
        sig = get_signal(price)
        print(f"  {label}  (UP={price:.0%})")
        print(format_signal(sig))
        print()