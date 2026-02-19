"""
approve.py
==========
ONE-TIME SETUP: Run this once before your first live trade (EOA/MetaMask wallets only).
Approves the Polymarket exchange contracts to spend USDC and CTF tokens from your wallet.

Usage:
  pip install web3 python-dotenv
  python approve.py
"""

import os
from dotenv import load_dotenv
from web3 import Web3

load_dotenv()

RPC_URL     = "https://polygon-rpc.com"
PRIVATE_KEY = os.environ["POLYGON_PRIVATE_KEY"]

# Official Polygon mainnet contract addresses
USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
CTF_ADDRESS  = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
EXCHANGE     = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"

w3      = Web3(Web3.HTTPProvider(RPC_URL))
account = w3.eth.account.from_key(PRIVATE_KEY)

print(f"Wallet: {account.address}")
print(f"RPC connected: {w3.is_connected()}")
print(f"Approving exchange: {EXCHANGE}")
print()

MAX_INT = 2**256 - 1
ERC20_ABI = [{
    "inputs": [{"type": "address"}, {"type": "uint256"}],
    "name": "approve",
    "outputs": [{"type": "bool"}],
    "type": "function",
}]

for token_addr, label in [(USDC_ADDRESS, "USDC"), (CTF_ADDRESS, "CTF")]:
    contract = w3.eth.contract(address=token_addr, abi=ERC20_ABI)
    nonce = w3.eth.get_transaction_count(account.address)
    tx = contract.functions.approve(EXCHANGE, MAX_INT).build_transaction({
        "from":               account.address,
        "nonce":              nonce,
        "gas":                60_000,
        "maxFeePerGas":       w3.to_wei("50", "gwei"),
        "maxPriorityFeePerGas": w3.to_wei("30", "gwei"),
    })
    signed = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
    status = "✓ SUCCESS" if receipt.status == 1 else "✗ FAILED"
    print(f"{label} approval: {status} | tx: {tx_hash.hex()}")

print()
print("Done. You can now run main.py with PAPER_TRADE=false.")
