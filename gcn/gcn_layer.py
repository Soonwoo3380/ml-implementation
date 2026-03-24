import torch
import torch.nn as nn
from torch_geometric.utils import add_self_loops, degree


class GCNConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.W = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, edge_index, num_nodes):
        edge_index = add_self_loops(edge_index, num_nodes=num_nodes)

        row, col = edge_index
        deg = degree(col, num_nodes=num_nodes, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        x = self.W(x)

        out = torch.zeros(num_nodes, x.size(1), device=x.device)
        out.scatter_add_()