"""Step 2 — Data Processing Module (A/B/C/D)."""
import pandas as pd
import numpy as np
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


# A) FEATURE ENGINEERING
def engineer_features(df):
    df = df.copy()
    df["hour"] = pd.to_datetime(df["Timestamp"], errors="coerce").dt.hour.fillna(0).astype(int)
    df["amount_log"] = np.log1p(df["Amount Received"].fillna(0))
    df["direction_flag"] = (df["From Bank"] != df["To Bank"]).astype(int)
    return df[["hour", "amount_log", "direction_flag", "Is_Laundering"]]


# B) TRAIN/TEST SPLIT + BALANCING
def split_and_balance(df, test_size=0.2):
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx], df.iloc[split_idx:]


# C) INTEGRATION HOOK (processor → graph → model)
def build_graph_features(dataset="HI-Small"):
    df = load_transactions(dataset)
    feats = engineer_features(df)
    from src.graph.network import build_transaction_graph
    G = build_transaction_graph(dataset)
    return G, feats


# D) FULL PIPELINE VERIFICATION
def verify_pipeline(dataset="HI-Small", n_samples=1000):
    df = load_transactions(dataset).head(n_samples)
    feats = engineer_features(df)
    assert "hour" in feats.columns and "amount_log" in feats.columns
    print("Pipeline OK — features:", feats.shape, "labels:", feats["Is_Laundering"].value_counts().to_dict())
    return feats
