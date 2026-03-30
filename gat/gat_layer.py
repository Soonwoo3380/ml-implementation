import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, softmax

class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.6, alpha=0.2):
        super().__init__()

        self.W = nn.Linear(in_channels, out_channels, bias=False)
        self.a = nn.Parameter(torch.empty(2 * out_channels, 1))

        self.dropout = dropout
        self.alpha = alpha

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x, edge_index, num_nodes, return_attention=False):
        x = F.dropout(x, p=self.dropout, training=self.training)

        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index

        x = self.W(x)

        x_i = x[col]
        x_j = x[row]

        e = torch.cat([x_i, x_j], dim=1) @ self.a
        e = F.leaky_relu(e.squeeze(-1), negative_slope=self.alpha)

        alpha = softmax(e, col)

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.zeros(num_nodes, x.size(1), device=x.device)
        out.index_add_(0, col, alpha.unsqueeze(-1) * x_j)

        if return_attention:
            return out, alpha, edge_index
        return out
    

class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads, concat=True, dropout=0.6, alpha=0.2):
        super().__init__()

        self.concat = concat
        self.heads = nn.ModuleList([
            GATLayer(in_channels, out_channels, dropout=dropout, alpha=alpha)
            for _ in range(heads)
        ])
    
    def forward(self, x, edge_index, num_nodes):
        head_outs = [head(x, edge_index, num_nodes) for head in self.heads]
    
        if self.concat:
            return torch.cat(head_outs, dim=1)
        else:
            return torch.mean(torch.stack(head_outs, dim=0), dim=0)