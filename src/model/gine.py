"""Step 4 — ML Detection Module (GINE-style GNN)."""
import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool


class FraudSentinelGINE(nn.Module):
    def __init__(self, node_dim=32, edge_dim=8, hidden=64, out=1):
        super().__init__()
        self.node_enc = nn.Linear(node_dim, hidden)
        self.edge_enc = nn.Linear(edge_dim, hidden)
        self.conv1 = GINEConv(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU()))
        self.conv2 = GINEConv(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU()))
        self.classifier = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out))

    def forward(self, x, edge_index, edge_attr):
        x = self.node_enc(x)
        e = self.edge_enc(edge_attr)
        x = self.conv1(x, edge_index, e).relu()
        x = self.conv2(x, edge_index, e).relu()
        return self.classifier(x).squeeze(-1)
