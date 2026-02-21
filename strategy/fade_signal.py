"""
strategy/fade_signal.py
=======================
Signal: BET DOWN when:
  1. UP is priced >= 90% (market overconfident)
  2. Previous candle was DOWN (momentum confirmation)

Based on backtest:
  - UP priced 90-100% → only wins 66.7% (edge -28.8%, n=93)
  - After DOWN candle → DOWN again 65.3% (n=120 pairs)
  - Combined: both conditions = strongest signal
"""

from datetime import datetime, timezone

# ── Config ────────────────────────────────────────────────────────────────────
FADE_THRESHOLD      = 0.90   # bet DOWN when UP >= this
MOMENTUM_CONFIRMED  = True   # require prev candle = DOWN too
MIN_TIME_LEFT_SECS  = 60     # don't enter if < 60s left in candle
MAX_TIME_LEFT_SECS  = 240    # don't enter too early (> 4min left)


def get_signal(
    up_price: float,
    prev_candle_was_up: bool | None,
    seconds_left: int,
) -> dict:
    """
    Returns signal dict:
      action:     'DOWN' | 'SKIP'
      confidence: float 0-1
      reasons:    list[str]
    """
    reasons    = []
    conditions = []

    # ── Condition 1: Extreme bullish overpricing ──────────────────────────────
    if up_price >= FADE_THRESHOLD:
        edge = -(up_price - 0.667)   # expected: only 66.7% win at this level
        conditions.append(('extreme_price', abs(edge)))
        reasons.append(
            f"UP overpriced at {up_price:.0%} "
            f"(historically only wins 66.7%, edge={edge:+.1%})"
        )
    else:
        return _skip(f"UP={up_price:.0%} below fade threshold {FADE_THRESHOLD:.0%}")

    # ── Condition 2: Momentum (prev candle DOWN) ──────────────────────────────
    if prev_candle_was_up is None:
        reasons.append("no prev candle data — partial signal only")
        confidence = 0.50
    elif not prev_candle_was_up:
        conditions.append(('momentum', 0.653))
        reasons.append("prev candle was DOWN → momentum DOWN (65.3% historical)")
        confidence = 0.75
    else:
        reasons.append("prev candle was UP → momentum weakens signal")
        confidence = 0.40   # still some edge from price alone, but weaker
        if confidence < 0.60:
            return _skip("prev candle UP reduces confidence below threshold")

    # ── Condition 3: Timing ───────────────────────────────────────────────────
    if seconds_left < MIN_TIME_LEFT_SECS:
        return _skip(f"only {seconds_left}s left — too late to enter")
    if seconds_left > MAX_TIME_LEFT_SECS:
        reasons.append(f"entering at {seconds_left}s left (within window)")

    # ── Final confidence ──────────────────────────────────────────────────────
    # Boost if both conditions met
    if len(conditions) >= 2:
        confidence = min(confidence + 0.15, 1.0)
        reasons.append("both conditions confirmed → boosted confidence")

    return {
        'action':     'DOWN',
        'confidence': confidence,
        'reasons':    reasons,
        'up_price':   up_price,
        'seconds_left': seconds_left,
    }


def _skip(reason: str) -> dict:
    return {'action': 'SKIP', 'confidence': 0.0, 'reasons': [reason],
            'up_price': 0, 'seconds_left': 0}


def format_signal(sig: dict) -> str:
    R  = "\033[0m"
    GR = "\033[92m"
    RE = "\033[91m"
    GY = "\033[90m"
    B  = "\033[1m"
    a  = sig['action']
    c  = sig.get('confidence', 0)
    col = RE if a == 'DOWN' else GY
    lines = [f"  {col}{B}SIGNAL: {a}  confidence={c:.0%}{R}"]
    for r in sig['reasons']:
        lines.append(f"    {GY}→ {r}{R}")
    return "\n".join(lines)