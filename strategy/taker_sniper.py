"""
strategy/taker_sniper.py
========================
Taker Mode strategy: executes FOK market orders when SignalEngine
fires a high-confidence signal.

Responsibilities:
  - Receive TradeSignal from SignalEngine
  - Apply cooldown enforcement
  - Calculate trade size via RiskManager.kelly_size
  - Execute via PolyExecutor.execute_taker_fok
  - Log outcome to MetricsTracker
  - Update SharedState position fields
"""

import logging
import time

from core.state import SharedState
from core.executor import PolyExecutor
from risk.fee_calculator import FeeCalculator
from risk.position_manager import RiskManager
from strategy.signal_engine import TradeSignal
from utils.metrics import MetricsTracker

logger = logging.getLogger(__name__)


class TakerSniper:
    COOLDOWN_S = 5.0                  # Min seconds between taker trades
    MIN_CONFIDENCE = 0.40             # Only fire above this confidence
    # Confidence required per signal type
    MIN_CONF_BY_TYPE = {
        "COMBINED": 0.40,
        "VELOCITY": 0.50,
        "DIVERGENCE": 0.55,
    }

    def __init__(
        self,
        state: SharedState,
        executor: PolyExecutor,
        risk_mgr: RiskManager,
        metrics: MetricsTracker,
        paper_trade: bool = True,
    ):
        self.state = state
        self.executor = executor
        self.risk_mgr = risk_mgr
        self.metrics = metrics
        self.paper_trade = paper_trade
        self._last_trade_time = 0.0
        self._pending_outcome: dict | None = None  # track last trade for outcome logging

    def _is_cooled_down(self) -> bool:
        return (time.monotonic() - self._last_trade_time) >= self.COOLDOWN_S

    def should_execute(self, signal: TradeSignal) -> tuple[bool, str]:
        if not signal.should_trade:
            return False, "No signal"
        if not self._is_cooled_down():
            remaining = self.COOLDOWN_S - (time.monotonic() - self._last_trade_time)
            return False, f"Cooldown: {remaining:.1f}s"

        min_conf = self.MIN_CONF_BY_TYPE.get(signal.signal_type, self.MIN_CONFIDENCE)
        if signal.confidence < min_conf:
            return False, f"Confidence {signal.confidence:.3f} < min {min_conf:.3f}"

        if self.state.open_position_side:
            return False, f"Already in position: {self.state.open_position_side}"

        return True, "OK"

    def execute(self, signal: TradeSignal, available_usdc: float) -> dict | None:
        ok, reason = self.should_execute(signal)
        if not ok:
            return None

        # Estimate win probability: required break-even + confidence bonus
        est_win_prob = (
            FeeCalculator.required_win_probability(signal.target_price)
            + 0.04 * signal.confidence
        )
        size_usdc = self.risk_mgr.kelly_size(
            est_win_prob, signal.target_price, available_usdc
        )

        if size_usdc < 1.0:
            logger.debug(f"Kelly size too small: ${size_usdc:.2f}")
            return None

        fee = FeeCalculator.taker_fee(signal.target_price, size_usdc / signal.target_price)

        if self.paper_trade:
            logger.info(
                f"[PAPER TAKER] {signal.side} ${size_usdc:.2f} "
                f"@ {signal.target_price:.4f} | fee=${fee:.4f} | "
                f"conf={signal.confidence:.3f} [{signal.signal_type}]"
            )
            self._last_trade_time = time.monotonic()
            self.metrics.record_paper_trade(signal, size_usdc, fee)
            # Update state so other modules see open position
            self.state.open_position_side = signal.side
            self.state.open_position_size = size_usdc
            self.state.open_position_price = signal.target_price
            return {"status": "paper", "side": signal.side, "size": size_usdc}

        # Live execution
        try:
            clob_side = "BUY"  # always buy (YES or NO token)
            resp = self.executor.execute_taker_fok(
                token_id=self.state.active_token_id,
                amount_usdc=size_usdc,
                side=clob_side,
            )
            status = resp.get("status", "unknown")

            if status == "matched":
                logger.info(
                    f"FILLED {signal.side} ${size_usdc:.2f} "
                    f"@ {signal.target_price:.4f} | {signal.reason}"
                )
                self._last_trade_time = time.monotonic()
                self.state.open_position_side = signal.side
                self.state.open_position_size = size_usdc
                self.state.open_position_price = signal.target_price
                self.metrics.record_live_trade(signal, size_usdc, fee, resp)
            else:
                logger.info(f"FOK rejected (no liquidity): {status}")
                self.metrics.record_fok_rejection(signal)

            return resp

        except Exception as e:
            logger.error(f"Taker execution error: {e}")
            return None
