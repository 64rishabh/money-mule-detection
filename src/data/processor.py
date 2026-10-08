"""Step 2 — Data Processing Module."""
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/ibm-transactions-for-anti-money-laundering-aml")


def load_transactions(dataset="HI-Small"):
    """Load transaction CSV. Returns DataFrame with cleaned types."""
    path = DATA_DIR / f"{dataset}_Trans.csv"
    df = pd.read_csv(path)
    df["Amount_Received"] = pd.to_numeric(df["Amount Received"], errors="coerce")
    df["Amount_Paid"] = pd.to_numeric(df["Amount Paid"], errors="coerce")
    df["Is_Laundering"] = df["Is Laundering"].astype(int)
    return df


def load_accounts(dataset="HI-Small"):
    path = DATA_DIR / f"{dataset}_accounts.csv"
    return pd.read_csv(path)
