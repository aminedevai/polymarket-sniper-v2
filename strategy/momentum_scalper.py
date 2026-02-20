# strategy/momentum_scalper.py
"""
MomentumScalper Strategy for 5-Minute BTC Markets

Optimized for high-frequency 5-minute windows with:
- Micro-momentum detection (10-30 second windows)
- Order book flow analysis
- Cross-exchange lag arbitrage (Binance -> Polymarket)
- Time-decay entry (enter at T+3:30 for better risk/reward)
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class SignalStrength(Enum):
    NONE = 0
    WEAK = 1
    MEDIUM = 2
    STRONG = 3
    EXTREME = 4


@dataclass
class MarketMicrostructure:
    """Real-time order book analysis"""
    bid_ask_spread: float
    bid_depth_5pct: float  # Depth within 5% of best bid
    ask_depth_5pct: float
    imbalance: float  # (bid_depth - ask_depth) / total_depth
    recent_trade_flow: float  # Buy vs sell pressure
    volatility_10s: float
    timestamp: float


@dataclass
class Signal:
    direction: str  # 'UP' or 'DOWN'
    strength: SignalStrength
    confidence: float  # 0.0 to 1.0
    entry_price: float
    target_price: float
    stop_price: float
    time_to_close: float  # seconds remaining
    reason: str
    metadata: Dict[str, Any]


class MomentumScalper:
    """
    High-frequency scalping strategy for 5-minute BTC markets.

    Key concepts:
    1. LATE ENTRY: Enter at 3:30-4:00 mark when direction is clearer
    2. MOMENTUM IGNITION: Detect micro-breakouts in 10-20s windows
    3. FLOW ANALYSIS: Follow smart money via order book changes
    4. CROSS-LAG: Exploit Binance price leading Polymarket by 1-3s
    """

    def __init__(
            self,
            state: Any,  # Your SharedState
            executor: Any,  # PolyExecutor
            risk_manager: Any,  # RiskManager
            metrics: Any,  # MetricsTracker
            paper_trade: bool = True,
            # Strategy params
            entry_window_start: float = 210,  # 3:30 in seconds
            entry_window_end: float = 240,  # 4:00 in seconds
            min_confidence: float = 0.65,
            momentum_lookback: int = 5,  # 5 ticks
            profit_target_ticks: int = 3,  # 0.03 (3 cents)
            stop_loss_ticks: int = 5,  # 0.05 (5 cents)
            max_position_time: float = 45,  # Max 45s in trade
            cross_lag_threshold: float = 0.002,  # 0.2% Binance lead
    ):
        self.state = state
        self.executor = executor
        self.risk_mgr = risk_manager
        self.metrics = metrics
        self.paper_trade = paper_trade

        # Config
        self.entry_window_start = entry_window_start
        self.entry_window_end = entry_window_end
        self.min_confidence = min_confidence
        self.momentum_lookback = momentum_lookback
        self.profit_target = profit_target_ticks / 100  # Convert to probability
        self.stop_loss = stop_loss_ticks / 100
        self.max_position_time = max_position_time
        self.cross_lag_threshold = cross_lag_threshold

        # State
        self.price_history: list = []
        self.ob_history: list = []  # Order book history
        self.last_binance_price: Optional[float] = None
        self.last_polymarket_price: Optional[float] = None
        self.active_position: Optional[Dict] = None
        self.market_start_time: Optional[float] = None

        logger.info(f"MomentumScalper initialized (paper={paper_trade})")
        logger.info(f"Entry window: {entry_window_start}-{entry_window_end}s")

    async def start(self):
        """Main strategy loop"""
        logger.info("MomentumScalper starting...")

        while True:
            try:
                await self._strategy_loop()
                await asyncio.sleep(0.1)  # 100ms tick
            except Exception as e:
                logger.error(f"Strategy error: {e}", exc_info=True)
                await asyncio.sleep(1)

    async def _strategy_loop(self):
        """Core strategy logic per tick"""

        # Check if we have active market
        if not hasattr(self.state, 'current_market') or not self.state.current_market:
            logger.debug("No active market")
            return

        market = self.state.current_market
        now = time.time()

        # Calculate time elapsed in 5-min window
        if self.market_start_time is None:
            self.market_start_time = market.get('startTime', 0) / 1000  # ms to s

        elapsed = now - self.market_start_time
        time_to_close = 300 - elapsed  # 5 minutes = 300s

        if time_to_close <= 0:
            logger.info("Market expired, resetting...")
            self._reset_market()
            return

        # Update price histories
        self._update_data()

        # Check for exit if in position
        if self.active_position:
            await self._manage_position(time_to_close)
            return

        # Entry logic - only in specific time window
        if self.entry_window_start <= elapsed <= self.entry_window_end:
            signal = self._generate_signal(time_to_close)

            if signal and signal.confidence >= self.min_confidence:
                await self._enter_position(signal)
        else:
            logger.debug(
                f"Outside entry window: {elapsed:.0f}s (window: {self.entry_window_start}-{self.entry_window_end})")

    def _update_data(self):
        """Update price and order book histories"""

        # Get latest prices from state (populated by your listeners)
        binance_price = getattr(self.state, 'binance_btc_price', None)
        poly_mid = self._get_polymarket_mid()

        if binance_price:
            self.last_binance_price = binance_price

        if poly_mid:
            self.last_polymarket_price = poly_mid
            self.price_history.append({
                'timestamp': time.time(),
                'price': poly_mid,
                'binance': self.last_binance_price
            })

            # Keep last 60 ticks (approx 6 seconds at 100ms)
            if len(self.price_history) > 60:
                self.price_history = self.price_history[-60:]

        # Update order book microstructure if available
        ob = getattr(self.state, 'polymarket_orderbook', None)
        if ob:
            micro = self._analyze_orderbook(ob)
            self.ob_history.append(micro)
            if len(self.ob_history) > 30:
                self.ob_history = self.ob_history[-30:]

    def _get_polymarket_mid(self) -> Optional[float]:
        """Get Polymarket mid price from state"""
        ob = getattr(self.state, 'polymarket_orderbook', None)
        if ob and 'bids' in ob and 'asks' in ob:
            best_bid = ob['bids'][0]['price'] if ob['bids'] else 0
            best_ask = ob['asks'][0]['price'] if ob['asks'] else 1
            return (best_bid + best_ask) / 2
        return None

    def _analyze_orderbook(self, ob: Dict) -> MarketMicrostructure:
        """Analyze order book for microstructure signals"""

        bids = ob.get('bids', [])
        asks = ob.get('asks', [])

        if not bids or not asks:
            return MarketMicrostructure(0, 0, 0, 0, 0, 0, time.time())

        best_bid = bids[0]['price']
        best_ask = asks[0]['price']
        spread = best_ask - best_bid

        # Calculate depth within 5% of mid
        mid = (best_bid + best_ask) / 2
        bid_depth = sum(b['size'] for b in bids if b['price'] >= mid * 0.95)
        ask_depth = sum(a['size'] for a in asks if a['price'] <= mid * 1.05)
        total_depth = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0

        # Recent trade flow (if available in state)
        recent_trades = getattr(self.state, 'recent_trades', [])
        buy_pressure = sum(t['size'] for t in recent_trades[-10:] if t['side'] == 'buy')
        sell_pressure = sum(t['size'] for t in recent_trades[-10:] if t['side'] == 'sell')
        total_pressure = buy_pressure + sell_pressure
        trade_flow = (buy_pressure - sell_pressure) / total_pressure if total_pressure > 0 else 0

        # Volatility
        vol = self._calc_volatility()

        return MarketMicrostructure(
            bid_ask_spread=spread,
            bid_depth_5pct=bid_depth,
            ask_depth_5pct=ask_depth,
            imbalance=imbalance,
            recent_trade_flow=trade_flow,
            volatility_10s=vol,
            timestamp=time.time()
        )

    def _calc_volatility(self) -> float:
        """Calculate 10-second realized volatility"""
        if len(self.price_history) < 10:
            return 0.0

        recent = self.price_history[-10:]
        prices = [p['price'] for p in recent]
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns) * np.sqrt(10)  # Annualized-ish

    def _generate_signal(self, time_to_close: float) -> Optional[Signal]:
        """
        Generate trading signal based on:
        1. Cross-exchange lag (Binance leads Polymarket)
        2. Order book momentum
        3. Time pressure (late in window = more certainty)
        """

        if len(self.price_history) < self.momentum_lookback:
            return None

        # 1. Cross-exchange lag analysis
        lag_signal = self._analyze_cross_lag()

        # 2. Order book momentum
        ob_signal = self._analyze_ob_momentum()

        # 3. Price momentum
        price_signal = self._analyze_price_momentum()

        # 4. Time pressure factor (later = higher confidence needed but clearer direction)
        time_factor = min(1.0, (300 - time_to_close) / 240)  # 0 to 1

        # Combine signals
        combined = self._combine_signals(lag_signal, ob_signal, price_signal, time_factor)

        if combined['confidence'] < self.min_confidence:
            return None

        # Determine direction
        direction = 'UP' if combined['score'] > 0 else 'DOWN'

        # Calculate entry/target/stop
        current_price = self.last_polymarket_price or 0.5

        # Dynamic sizing based on confidence
        if combined['confidence'] > 0.85:
            strength = SignalStrength.EXTREME
        elif combined['confidence'] > 0.75:
            strength = SignalStrength.STRONG
        elif combined['confidence'] > 0.65:
            strength = SignalStrength.MEDIUM
        else:
            strength = SignalStrength.WEAK

        # Entry price with slippage estimate
        entry = current_price + (0.002 if direction == 'UP' else -0.002)

        # Targets based on strength
        if strength in [SignalStrength.EXTREME, SignalStrength.STRONG]:
            target = entry + self.profit_target if direction == 'UP' else entry - self.profit_target
            stop = entry - self.stop_loss if direction == 'UP' else entry + self.stop_loss
        else:
            # Tighter targets for weaker signals
            target = entry + (self.profit_target * 0.6) if direction == 'UP' else entry - (self.profit_target * 0.6)
            stop = entry - (self.stop_loss * 0.8) if direction == 'UP' else entry + (self.stop_loss * 0.8)

        return Signal(
            direction=direction,
            strength=strength,
            confidence=combined['confidence'],
            entry_price=entry,
            target_price=max(0.01, min(0.99, target)),
            stop_price=max(0.01, min(0.99, stop)),
            time_to_close=time_to_close,
            reason=combined['reason'],
            metadata=combined['metadata']
        )

    def _analyze_cross_lag(self) -> Dict:
        """
        Detect when Binance price leads Polymarket.
        Returns signal dict with score and confidence.
        """
        if self.last_binance_price is None or self.last_polymarket_price is None:
            return {'score': 0, 'confidence': 0, 'reason': 'no_data'}

        # Calculate lag (positive = Binance higher)
        lag = self.last_binance_price - self.last_polymarket_price
        lag_pct = lag / self.last_polymarket_price

        # If Binance moved significantly and Polymarket hasn't caught up
        if abs(lag_pct) > self.cross_lag_threshold:
            direction = 1 if lag > 0 else -1  # 1 = UP, -1 = DOWN
            confidence = min(1.0, abs(lag_pct) / 0.01)  # Scale up to 1% lag

            return {
                'score': direction * confidence,
                'confidence': confidence * 0.9,  # Slight discount
                'reason': f'cross_lag_{direction}',
                'metadata': {'lag_pct': lag_pct, 'binance': self.last_binance_price, 'poly': self.last_polymarket_price}
            }

        return {'score': 0, 'confidence': 0, 'reason': 'no_lag', 'metadata': {}}

    def _analyze_ob_momentum(self) -> Dict:
        """Analyze order book for momentum signals"""
        if len(self.ob_history) < 3:
            return {'score': 0, 'confidence': 0, 'reason': 'insufficient_data'}

        recent = self.ob_history[-3:]

        # Imbalance trend
        imb_trend = recent[-1].imbalance - recent[0].imbalance

        # Trade flow
        flow = recent[-1].recent_trade_flow

        # Spread tightening/widening
        spread_trend = recent[-1].bid_ask_spread - recent[0].bid_ask_spread

        score = 0
        confidence = 0

        # Strong bid imbalance + buy flow = UP
        if recent[-1].imbalance > 0.3 and flow > 0.2:
            score = 1
            confidence = 0.6 + (recent[-1].imbalance * 0.3)
        # Strong ask imbalance + sell flow = DOWN
        elif recent[-1].imbalance < -0.3 and flow < -0.2:
            score = -1
            confidence = 0.6 + (abs(recent[-1].imbalance) * 0.3)

        # Spread tightening increases confidence
        if spread_trend < -0.001:
            confidence += 0.1

        return {
            'score': score * confidence,
            'confidence': min(1.0, confidence),
            'reason': 'ob_momentum',
            'metadata': {
                'imbalance': recent[-1].imbalance,
                'flow': flow,
                'spread': recent[-1].bid_ask_spread
            }
        }

    def _analyze_price_momentum(self) -> Dict:
        """Analyze recent price momentum"""
        if len(self.price_history) < self.momentum_lookback:
            return {'score': 0, 'confidence': 0}

        prices = [p['price'] for p in self.price_history[-self.momentum_lookback:]]

        # Simple momentum: slope of recent prices
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]

        # Normalize slope to approximate probability change per tick
        momentum = slope * len(prices)

        if abs(momentum) < 0.001:
            return {'score': 0, 'confidence': 0, 'reason': 'no_momentum'}

        direction = 1 if momentum > 0 else -1
        confidence = min(1.0, abs(momentum) / 0.02)  # 2% move = high confidence

        return {
            'score': direction * confidence,
            'confidence': confidence,
            'reason': 'price_momentum',
            'metadata': {'momentum': momentum}
        }

    def _combine_signals(self, lag_sig, ob_sig, price_sig, time_factor) -> Dict:
        """Combine multiple signals with weighting"""

        # Weights (adjust based on backtesting)
        weights = {
            'lag': 0.4,  # Cross-exchange lag is strong for 5-min
            'ob': 0.35,  # Order book flow
            'price': 0.25  # Price momentum
        }

        total_score = (
                lag_sig.get('score', 0) * weights['lag'] +
                ob_sig.get('score', 0) * weights['ob'] +
                price_sig.get('score', 0) * weights['price']
        )

        # Average confidence weighted by signal strength
        total_confidence = (
                lag_sig.get('confidence', 0) * weights['lag'] +
                ob_sig.get('confidence', 0) * weights['ob'] +
                price_sig.get('confidence', 0) * weights['price']
        )

        # Boost confidence slightly as time passes (more certainty)
        total_confidence = min(1.0, total_confidence * (0.9 + 0.2 * time_factor))

        reasons = []
        if lag_sig.get('confidence', 0) > 0.5:
            reasons.append(f"lag({lag_sig['reason']})")
        if ob_sig.get('confidence', 0) > 0.5:
            reasons.append(f"ob({ob_sig['metadata'].get('imbalance', 0):.2f})")
        if price_sig.get('confidence', 0) > 0.5:
            reasons.append(f"mom({price_sig['metadata'].get('momentum', 0):.4f})")

        return {
            'score': total_score,
            'confidence': total_confidence,
            'reason': '+'.join(reasons) if reasons else 'weak_signals',
            'metadata': {
                'lag': lag_sig.get('metadata', {}),
                'ob': ob_sig.get('metadata', {}),
                'price': price_sig.get('metadata', {})
            }
        }

    async def _enter_position(self, signal: Signal):
        """Execute entry order"""

        # Risk check
        if not self.risk_mgr.can_trade():
            logger.warning("Risk manager blocked trade")
            return

        size = self.risk_mgr.calculate_position_size(
            confidence=signal.confidence,
            strength=signal.strength.value
        )

        token_id = self._get_token_id(signal.direction)

        logger.info(f"ENTER {signal.direction} | Price: {signal.entry_price:.4f} | "
                    f"Size: {size} | Conf: {signal.confidence:.2f} | Reason: {signal.reason}")

        if self.paper_trade:
            self.active_position = {
                'direction': signal.direction,
                'entry_price': signal.entry_price,
                'entry_time': time.time(),
                'size': size,
                'target': signal.target_price,
                'stop': signal.stop_price,
                'token_id': token_id,
                'paper': True
            }
            self.metrics.record_entry(signal, size, paper=True)
        else:
            # Live execution
            try:
                order = await self.executor.place_order(
                    token_id=token_id,
                    price=signal.entry_price,
                    size=size,
                    side='BUY',
                    order_type='FOK'  # Fill or kill for quick entry
                )

                if order and order.get('success'):
                    self.active_position = {
                        'direction': signal.direction,
                        'entry_price': signal.entry_price,
                        'entry_time': time.time(),
                        'size': size,
                        'target': signal.target_price,
                        'stop': signal.stop_price,
                        'token_id': token_id,
                        'order_id': order.get('order_id'),
                        'paper': False
                    }
                    self.metrics.record_entry(signal, size, paper=False)
                else:
                    logger.error(f"Entry failed: {order}")

            except Exception as e:
                logger.error(f"Entry error: {e}")

    async def _manage_position(self, time_to_close: float):
        """Manage open position - check exits"""

        pos = self.active_position
        current_price = self.last_polymarket_price

        if not current_price:
            return

        # Calculate P&L
        if pos['direction'] == 'UP':
            pnl = current_price - pos['entry_price']
            hit_target = current_price >= pos['target']
            hit_stop = current_price <= pos['stop']
        else:
            pnl = pos['entry_price'] - current_price
            hit_target = current_price <= pos['target']
            hit_stop = current_price >= pos['stop']

        # Time-based exit (close to expiration)
        time_exit = time_to_close < 10  # Last 10 seconds

        # Max holding time
        hold_time = time.time() - pos['entry_time']
        max_time_exit = hold_time > self.max_position_time

        exit_reason = None

        if hit_target:
            exit_reason = 'target'
        elif hit_stop:
            exit_reason = 'stop'
        elif time_exit:
            exit_reason = 'time'
        elif max_time_exit:
            exit_reason = 'max_time'

        if exit_reason:
            await self._exit_position(current_price, exit_reason, pnl)

    async def _exit_position(self, price: float, reason: str, pnl: float):
        """Close position"""

        pos = self.active_position

        logger.info(f"EXIT {pos['direction']} | Price: {price:.4f} | "
                    f"PnL: {pnl:.4f} | Reason: {reason} | "
                    f"Time: {time.time() - pos['entry_time']:.1f}s")

        if not self.paper_trade and not pos.get('paper'):
            try:
                # Place sell order
                await self.executor.place_order(
                    token_id=pos['token_id'],
                    price=price,
                    size=pos['size'],
                    side='SELL',
                    order_type='IOC'  # Immediate or cancel
                )
            except Exception as e:
                logger.error(f"Exit error: {e}")

        self.metrics.record_exit(pos, price, pnl, reason)
        self.risk_mgr.update_after_trade(pnl)
        self.active_position = None

    def _get_token_id(self, direction: str) -> str:
        """Get token ID for direction from current market"""
        market = self.state.current_market
        if not market:
            return ""

        # Assuming your market data has token IDs stored
        # Adjust based on your actual data structure
        tokens = market.get('tokens', {})
        return tokens.get('UP' if direction == 'UP' else 'DOWN', '')

    def _reset_market(self):
        """Reset for new market"""
        self.price_history = []
        self.ob_history = []
        self.active_position = None
        self.market_start_time = None
        self.last_binance_price = None
        self.last_polymarket_price = None