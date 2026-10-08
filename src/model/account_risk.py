"""Step 5 — Account Risk Module."""
import pandas as pd


def calculate_account_risk(suspicious_ids, account_transactions):
    """Aggregate suspicious transactions to account-level risk."""
    risk = {}
    for acct, txs in account_transactions.items():
        total = len(txs)
        susp = sum(1 for t in txs if t in suspicious_ids)
        risk[acct] = {
            "total": total,
            "suspicious": susp,
            "score": round(susp / max(total, 1) * 100, 1),
            "level": "HIGH" if susp / max(total, 1) > 0.5 else "MEDIUM" if susp / max(total, 1) > 0.2 else "LOW"
        }
    return risk
