# Graph Attention Networks

## Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio

**Paper**  
https://arxiv.org/abs/1710.10903

## Understanding GAT Message Passing Step by Step

This repository demonstrates the internal workings of a GAT layer by implementing it from scratch and examining each intermediate step.

The goal is to clearly understand how node features are updated through:
- input dropout
- self-loop addition
- linear transformation
- pairwise attention score computation
- attention normalization
- weighted message passsing
- aggregation
- multi-head combination
  
## Overview
A GAT layer updates node features by aggregating information from neighboring nodes, while assigning different importance to different neighbors.
The core operation can be summarized as:

$$
h_i' = \sum_{j \in \mathcal{N}{i}} \alpha_{ij} W h_j
$$

where the attention coefficient is computed as:

$$
e_{ij} = \text{LeakyReLU}\left(a^\top [W h_i \, \| \, W h_j]\right)
$$

 and normalized by softmax:

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}{i}} \exp(e_{ik})}
$$

Where:

- $W$: learnable linear transformation
- $a$: learnable attention vector
- $e_{ij}$: unnormalized attention score between node $i$ and node $j$
- $\alpha_{ij}$: normalized attention coefficient
- $\mathcal{N}(i)$: neighbors of nodes $i$

Unlike GCN, GAT does not use a fixed normalization term based on node degree.
Instead, it learns how important each neighboring node is.

### Example Graph
- Node 0 → feature: 1.0
- Node 1 → feature: 2.0
- Node 2 → feature: 3.0
  
### Edges (undirected)
- 0--1
- 1--2

### PyG-style 'edge_index'
```python
edge_index = torch.tensor([
    [0, 1, 1, 2],
    [1, 0, 2, 1]
])
```
### Overall Pipeline
[01] [Input node features `x`](#1-input-node-features-x)  
[02] [Input `edge_index`](#2-input-edge_index)  
[03] [Apply input dropout](#3-apply-input-dropout)  
[04] [Add self-loops](#4-add-self-loops)  
[05] [Split into `row` / `col`](#5-split-into-row--col)  
[06] [Linear transformation](#6-linear-transformation)  
[07] [Gather target node features `x_i`](#7-gather-target-node-features-x_i)  
[08] [Gather source node features `x_j`](#8-gather-source-node-features-x_j)  
[09] [Concatenate `[x_i || x_j]`](#9-concatenate-x_i--x_j)  
[10] [Compute raw attention score](#10-compute-raw-attention-score)  
[11] [Apply LeakyReLU](#11-apply-leakyrelu)  
[12] [Normalize with softmax](#12-normalize-with-softmax)  
[13] [Apply attention dropout](#13-apply-attention-dropout)  
[14] [Initialize output tensor](#14-initialize-output-tensor)  
[15] [Aggregate weighted neighbor messages](#15-aggregate-weighted-neighbor-messages)  
[16] [Multi-head concatenation or averaging](#16-multi-head-concatenation-or-averaging)

## Step-by-Step

#### [01] Input node features `x`
- Shape: `[num_nodes, in_channels]`
```Python
x = torch.tensor([
  [1.0],
  [2.0],
  [3.0]
])
```

#### [02] Input `edge_index`
- Shape: `[2, num_edges]`
- Meaning:
  - first row → source nodes
  - second row → target nodes
```Python
edge_index = torch.tensor([
    [0, 1, 1, 2],
    [1, 0, 2, 1]
])
```

#### [03] Apply input dropout
```Python
x = F.dropout(x, p=self.dropout, training=self.training)
```
- Randomly drops some feature values during training
- Helps regularization
- Prevents over-reliance on specific input features

#### [04] Add self-loops
```Python
edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
```
- Each node is connected to itself
- Ensures that a node can also attend to its own feature
Example:
```Python
edge_index = torch.tensor([
    [0, 1, 1, 2, 0, 1, 2],
    [1, 0, 2, 1, 0, 1, 2]
])
```

#### [05] Split into `row`/`col`
```Python
row, col = edge_index
```
- `row`: source nodes
- `col`: target nodes
This means each edge is interpreted as:
- messages goes from row[k]
- messages arrives at col[k]

#### [06] Linear transformation
```Python
x = self.W(x)
```
- Projects node features into a new learnable space
- Makes feature comparison and message passing task-relevant
If the original feature is not directly suitable for attention, the model learns a transformed representation.
##### Example
```Python
W = [[2.0]]

x = [
  [1.0],
  [2.0],
  [3.0]
]
```
```Python
x' = W(x)

node 0 → 2 × 1 = 2
node 1 → 2 × 2 = 4
node 2 → 2 × 3 = 6
```
```Python
x = [
  [2.0],
  [4.0],
  [6.0]
]
```

#### [07] Gather target node feature `x_i`
```Python
x_i = x[col]
```
- For each edge, collect the feature of the target node
##### Example
```Python
col = [1, 0, 2, 1, 0, 1, 2]
```
```Python
x_i =
[
  x[1],  → 4
  x[0],  → 2
  x[2],  → 6
  x[1],  → 4
  x[0],  → 2
  x[1],  → 4
  x[2]   → 6
]
```
```Python
x_i = [4, 2, 6, 4, 2, 4, 6]
```

#### [08] Gather target node feature `x_j`
```Python
x_j = x[row]
```
- For each edge, collect the feature of the source node
- This is the node that sends the message
##### Example
```Python
row = [0, 1, 1, 2, 0, 1, 2]
```
```Python
x_j =
[
  x[0], → 2
  x[1], → 4
  x[1], → 4
  x[2], → 6
  x[0], → 2
  x[1], → 4
  x[2]  → 6
]
```
```Python
x_j = [2, 4, 4, 6, 2, 4, 6]
```

#### [09] Concatenate `[x_i || x_j]`
```Python
torch.cat([x_i, x_j], dim=1)
```
- Combines target and source features for each edge
- Lets the model evaluate how important source node `j` is to target node `i`
##### Example
```Python
edge 0 → [4, 2]
edge 1 → [2, 4]
edge 2 → [6, 4]
edge 3 → [4, 6]
...
```

#### [10] Compute raw attention score
```Python
torch.cat([x_i, x_j], dim=1) @ self.a
```
- `self.a` is a learnable attention vector
- Produces one scalar score per edge
- This score represents the raw importance of that neighbor before normalization
##### Example
```Python
a = [[1.0], [1.0]]
```
```Python
edge 0 → 4 + 2 = 6
edge 1 → 2 + 4 = 6
edge 2 → 6 + 4 = 10
edge 3 → 4 + 6 = 10
...
```
```Python
e = [6, 6, 10, 10, ...]
```

#### [11] Apply LeakyReLU
```Python
e = F.leaky_relu(e.squeeze(-1), negative_slope=self.alpha)
```
- Introduces nonlinearity into the attention score
- Prevents the attention mechanism from becoming purely linear

#### [12] Normalize with softmax
```Python
alpha = softmax(e, col)
```
- Normalizes attention scores over incoming edges of each target node
- For the same target node, all neighbor attention weights sum to 1

This means:
- larger `alpha` → more important neighbor
- smaller `alpha` → less important neighbor
##### Example
We compute the softmax normalization over all edges incoming to node 1.
Incoming edges to node 1:
- edge 0 → 𝑒 = 6
- edge 3 → 𝑒 = 10
- edge 5 → 𝑒 = 8
##### Step 1: Exponentiation
```Python
exp(6)  = 403
exp(10) = 22026
exp(8)  = 2981
```
##### Step 2: Compute normalization constant
```Python
sum = 403 + 22026 + 2981 = 25410
```
##### Step 3: Compute attention coefficients
```Python
α₀ = 403   / 25410 ≈ 0.016
α₃ = 22026 / 25410 ≈ 0.867
α₅ = 2981  / 25410 ≈ 0.117
```
##### What do they mean?
- Among the neighbors of node 1:
  - edge 3 (from node 2) has the highest importance
  - edge 0 (from node 0)has almost no influence

#### [13] Apply attention dropout
```Python
alpha = F.dropout(alpha, p=self.dropout, training=self.training)
```
- Drops some attention weights during training
- Helps prevent overfitting
- Encourages more robust neighbor selection

#### [14] Initialize output tensor
```Python
out = torch.zeros(num_nodes, x.size(1), device=x.device)
```
- Prepares a tensor to store updated node features

#### [15] Aggregate weighted neighbor messages
```Python
out.index_add_(0, col, alpha.unsqueeze(-1) * x_j)
```
- Each source features `x_j` is multiplied by its attention weight
- Then it is adde to the corresponding target node output
So the target node receives:
$$
\sum_{j \in \mathcal{N}{i}} \alpha_{ij} W h_j
$$
This is the core message-passing step of GAT.

#### [16] Multi-head concatenation or averaging
```Python
head_outs = [head(x, edge_index, num_nodes) for head in self.heads]
```
If `concat=True`:
```Python
return torch.cat(heads_outs, dim=1)
```
- Concatenates outputs from all heads
- Commonly used in hidden layers
If `concat=False`:
```Python
return torch.mean(torch.stack(head_outs, dim=0), dim=0)
```
- Averages outputs from all heads
- Commonly used in the final layer

#### Result
| Edge | Source | Target | Attention (α) | Source Feature | Message |
|:----:|:------:|:------:|:-------------:|:--------------:|:-------:|
| 0    | 0      | 1      | 0.25          | 2.0            | 0.50    |
| 1    | 1      | 0      | 0.60          | 4.0            | 2.40    |
| 2    | 1      | 2      | 0.70          | 4.0            | 2.80    |
| 3    | 2      | 1      | 0.50          | 6.0            | 3.00    |
| 4    | 0      | 0      | 0.40          | 2.0            | 0.80    |
| 5    | 1      | 1      | 0.30          | 4.0            | 1.20    |
| 6    | 2      | 2      | 0.30          | 6.0            | 1.80    |

#### Keypoints
- `W(x)`: decides **what to send**
- `α (attention)`: decides **how important each neighbor is**
- `softmax`: decides **relative importance among neighbors (per target node)**
- `out.index_add_`: decides **where to send**

---

# One-line summary
Ultimately, GCN updates node representations by aggregating and averaging information from neighboring nodes on a graph.

---