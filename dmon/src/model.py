import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class DMoN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_clusters, num_layers,temperature=1.0):
        super().__init__()
        self.num_layers = num_layers
        self.temperature = temperature

        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(input_dim, hidden_dim, add_self_loops=False))
        for _ in range(num_layers - 1):
            self.gcn_layers.apend(GCNConv(hidden_dim, hidden_dim, add_self_loops=False))

        self.linear = nn.Linear(hidden_dim, num_clusters)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, edge_index, edge_weight=None):
        h = x
        for gcn in self.gcn_layers:
            h = F.selu(gcn(h, edge_index, edge_weight))
        C = F.softmax(self.linear(h) / self.temperature, dim=1)
        return C