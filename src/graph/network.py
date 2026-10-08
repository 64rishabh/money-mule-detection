"""Step 3 — Transaction Graph Module."""
import networkx as nx
from src.data.processor import load_transactions


def build_transaction_graph(dataset="HI-Small"):
    df = load_transactions(dataset)
    G = nx.DiGraph()
    for _, row in df.iterrows():
        bank_from = str(row['From Bank'])
        acct_from = str(row['Account'])
        bank_to = str(row['To Bank'])
        acct_to = str(row['Account.1'])
        sender = "B" + bank_from + "_A" + acct_from
        receiver = "B" + bank_to + "_A" + acct_to
        G.add_edge(sender, receiver,
                   amount=float(row.get("Amount Received", 0) or 0),
                   timestamp=str(row.get("Timestamp", "")),
                   is_laundering=int(row.get("Is Laundering", 0)))
    return G
