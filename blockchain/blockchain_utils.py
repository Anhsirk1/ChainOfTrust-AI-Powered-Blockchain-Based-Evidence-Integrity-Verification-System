from web3 import Web3
import json
import os

# -----------------------------
# 🔹 Step 1: Connect to Polygon Amoy Testnet
# -----------------------------
ALCHEMY_URL = "https://polygon-amoy.g.alchemy.com/v2/OArXXY0W6AGFsRkoymdcJ"
PRIVATE_KEY = "c545192ea9d882f8f6571a42ef7a0f07e7ad8f5fa4e72d3ea4880b2d1a167549"
CONTRACT_ADDRESS = Web3.to_checksum_address("0x130dc8fd5d5c0bf5efcc602f6b1c6bee2611f9f7")

web3 = Web3(Web3.HTTPProvider(ALCHEMY_URL))

if web3.is_connected():
    print("✅ Connected to Polygon Amoy Testnet")
else:
    print("❌ Blockchain connection failed")

# -----------------------------
# 🔹 Step 2: Load Contract ABI
# -----------------------------
with open("contract_abi.json") as f:
    contract_abi = json.load(f)

contract = web3.eth.contract(address=CONTRACT_ADDRESS, abi=contract_abi)
ACCOUNT_ADDRESS = web3.eth.account.from_key(PRIVATE_KEY).address

# -----------------------------
# 🔹 Step 3: Record Events on Chain
# -----------------------------
def record_on_chain(role, action):
    try:
        data_hash = web3.keccak(text=f"{role}-{action}")
        nonce = web3.eth.get_transaction_count(ACCOUNT_ADDRESS)

        txn = contract.functions.recordEvent(data_hash, role, action).build_transaction({
            'from': ACCOUNT_ADDRESS,
            'gas': 250000,
            'gasPrice': web3.to_wei('5', 'gwei'),
            'nonce': nonce,
            'chainId': 80002
        })

        signed_txn = web3.eth.account.sign_transaction(txn, private_key=PRIVATE_KEY)
        tx_hash = web3.eth.send_raw_transaction(signed_txn.rawTransaction)

        print(f"✅ Event recorded on blockchain | TxHash: {web3.to_hex(tx_hash)}")
        return web3.to_hex(tx_hash)
    except Exception as e:
        print("❌ Blockchain error:", e)
        return None
