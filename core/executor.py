"""
core/executor.py
================
Wraps py-clob-client v0.34.5 for order execution.
Supports:
  - Taker FOK market orders (execute_taker_fok)
  - Maker post-only limit orders (execute_maker_limit)
  - Cancel all open orders (cancel_all)
  - Balance query (get_balance)

Wallet types:
  - EOA (MetaMask): SIGNATURE_TYPE=0, no funder
  - Email/Magic:    SIGNATURE_TYPE=1, funder=profile address
  - Gnosis Safe:    SIGNATURE_TYPE=2, funder=safe address
"""

import logging
import os

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import MarketOrderArgs, OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL

logger = logging.getLogger(__name__)


class PolyExecutor:
    HOST = "https://clob.polymarket.com"
    CHAIN_ID = 137  # Polygon mainnet

    def __init__(self):
        private_key = os.environ["POLYGON_PRIVATE_KEY"]
        funder = os.environ.get("POLYGON_FUNDER_ADDRESS")  # email wallets only
        sig_type = int(os.environ.get("SIGNATURE_TYPE", "0"))

        self.client = ClobClient(
            self.HOST,
            key=private_key,
            chain_id=self.CHAIN_ID,
            signature_type=sig_type,
            funder=funder if funder else None,
        )
        # Derive or reuse API credentials
        self.client.set_api_creds(self.client.create_or_derive_api_creds())
        logger.info(
            f"PolyExecutor initialized | sig_type={sig_type} | "
            f"funder={'yes' if funder else 'no'}"
        )

    def get_balance(self) -> float:
        """Returns available USDC balance."""
        try:
            resp = self.client.get_balance()
            return float(resp.get("balance", 0))
        except Exception as e:
            logger.error(f"get_balance error: {e}")
            return 0.0

    def execute_taker_fok(
        self,
        token_id: str,
        amount_usdc: float,
        side: str = "BUY",
    ) -> dict:
        """
        Place a Fill-Or-Kill market order.
        amount_usdc: USDC to spend (not shares).
        Returns the CLOB response dict.
        """
        clob_side = BUY if side == "BUY" else SELL
        order_args = MarketOrderArgs(
            token_id=token_id,
            amount=amount_usdc,
            side=clob_side,
        )
        signed_order = self.client.create_market_order(order_args)
        response = self.client.post_order(signed_order, OrderType.FOK)
        logger.info(f"FOK {side} ${amount_usdc:.2f} → status={response.get('status')}")
        return response

    def execute_maker_limit(
        self,
        token_id: str,
        price: float,
        size: float,
        side: str = "BUY",
    ) -> dict:
        """
        Place a post-only limit order (maker, zero taker fee, earns rebates).
        price: per-share price (0.0 – 1.0).
        size: number of shares.
        Uses GTC order type — ensure price is inside spread to avoid crossing.
        """
        clob_side = BUY if side == "BUY" else SELL
        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=size,
            side=clob_side,
        )
        signed_order = self.client.create_limit_order(order_args)
        # GTC with post-only intent — will be rejected if it crosses
        response = self.client.post_order(signed_order, OrderType.GTC)
        logger.info(
            f"Maker limit {side} {size:.0f} shares @ {price:.4f} "
            f"→ status={response.get('status')}"
        )
        return response

    def cancel_order(self, order_id: str) -> dict:
        """Cancel a specific open order by ID."""
        resp = self.client.cancel(order_id)
        logger.info(f"Cancelled order {order_id}: {resp}")
        return resp

    def cancel_all(self) -> dict:
        """Emergency: cancel all open orders."""
        resp = self.client.cancel_all()
        logger.info(f"Cancel all: {resp}")
        return resp

    def get_open_orders(self) -> list:
        """Returns list of open orders."""
        try:
            return self.client.get_orders() or []
        except Exception as e:
            logger.error(f"get_open_orders error: {e}")
            return []