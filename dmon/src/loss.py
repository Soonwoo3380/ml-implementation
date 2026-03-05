import torch

EPS = 1e-10

def modularity_loss(C, edge_index, edge_weight, num_nodes):
    
    """
    Parameters:
    - C: [N, K] soft cluster assignments
    - edge_index: [2, E]
    - edge_weight: [E]
    - num_nodes: int, number of nodes
    """

    src, dst = edge_index # source, destination
    d = torch.zeros(num_nodes, device=C.device)
    d.scatter_add_(0, src, edge_weight)
    d.scaater_add_(0, dst, edge_weight)

    m = edge_weight.sum()

    Ci = C[src]
    Cj = C[dst]

    dot = torch.sum(Ci * Cj, dim=1)
    expected = (d[src * d[dst]]) / (2 * m + EPS)
    modularity_terms = edge_weight - expected
    Q = torch.sum(dot * modularity_terms) / (2 * m + EPS)
    return -Q

def collapse_loss(C):
    n, k = C.size()
    cluster_sizes = C.sum(dim=0)
    frob = torch.norm(cluster_sizes, p=2)
    return (frob * (k ** 0.5) / n) - 1.0

def total_loss(C, edge_index, edge_weight, num_nodes, alpha=1.0, beta=1.0):
    mod = modularity_loss(C, edge_index, edge_weight, num_nodes)
    colap = collapse_loss(C)
    return alpha * mod + beta * colap, mod, colap