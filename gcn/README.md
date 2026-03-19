# Semi-Supervised Classificiation with Graph Convolutional Networks
Thomas N. Kipf, Max Welling
--

**Paper**  
https://arxiv.org/abs/1609.02907

## Understanding GCN Message Passing Step by Step
This repository demonstrates the internal workings of a GCN layer by implementing it from scratch and printing intermidiate results.

The goal is to clearly understand how node features are updated through:
- self-loop addition
- degree normalization
- linear transformation
- message passing
- aggregation

## Overview
A GCN layer updates node features by aggregating information from neighboring nodes.
The core operation can be summarized as:

$$
h_j' = \sum_{i \in \mathcal{N}(j)} \frac{1}{\sqrt{d_i d_j}} \cdot W x_i
$$

Where:
- $W$: learnable weight matrix
- $d_i$: degree of node $i$
- $\mathcal{N}(j)$: neighbors of node $j$

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
[03] [Add self-loops](#3-add-self-loops)  
[04] [row / col separation](#4-row--col-separation)  
[05] [Degree computation](#5-degree-computation)  
[06] [Inverse sqrt degree](#6-inverse-sqrt-degree)  
[07] [Edge normalization](#7-edge-normalization)  
[08] [Linear transformation](#8-linear-transformation)  
[09] [Gather source node features](#9-gather-source-node-features)  
[10] [Reshape norm](#10-reshape-norm)  
[11] [Compute messages](#11-compute-messages)  
[12] [Initialize output](#12-initialize-output)  
[13] [Aggregate messages](#13-aggregate-messages)

## Step-by-Step

#### [01] Input node features `x`
- Shape: `[num_nodes, in_channels]`
- Meaning: feature vector per node = node feature
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

#### [03] Add self-loops
- Each node connects to itself
- Ensures a node keeps its own information
```python
edge_index = torch.tensor([
    [0, 1, 1, 2, 0, 1, 2],
    [1, 0, 2, 1, 0, 1, 2]
])
```

#### [04] row / col separation
```python
row, col = edge_index
```

- `row`: source nodes (where edges depart from)
- `col`: target nodes (where edges arrive at)

#### [05] Degree computation
```python
deg = degree(col)
```
- Counts how many edges point into each node
- Example:
  - deg(0) = 2
  - deg(1) = 3
  - deg(2) = 2
  
#### [06] Inverse sqrt degree
```python
deg = degree(col)
```
- Used for normalization
- Prevents high-degree nodes from dominating

#### [07] Edge normalization
```python
norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
```
- The effect of degree on the influence of connected nodes

#### [08] Linear transformation
```python
x = W(x)
```
- Transforms node features
- Learns which features are important
  - To transform node features into a learnable representation to propagate only task-relevant information to neighbors
  - `It is incapable of perfomring downstream tasks, such as classification and clustering, on its own.`

#### [09] Gather source node features
```python
x[row]
```
- Each edge gets its source node feature

#### [10] Reshape norm
```python
norm = norm.unsqueeze(-1)
```
- Shape change:
  - before: `[E]`
  - after : `[E, 1]`
- Required for broadcasting

#### [11] Compute messages
```python
src = norm.unsqueeze(-1) * x[row]
```
- Departing node feature $×$ edge weight
  - 'message' that will be delivered

#### [12] Initialize output
```python
out = torch.zeros(num_nodes, x.size(1), device=x.device)
```
- Stores updated node features
- Reset every forward pass

#### [13] Aggregate messages
```python
out.scatter_add_(
  dim=0,
  index=col.unsqueeze(-1).expand_as(x[row]), 
  src=norm.unsqueeze(-1) * x[row]
)
```
Equivalent to:
```python
for each edge (i → j):
    out[j] += message(i → j)
```

#### Edge-wise Message Passing
| Edge | Source | Target | Norm   | Source Feature | Message |
|:----:|:------:|:------:|:------:|:--------------:|:-------:|
| 0    | 0      | 1      | 0.4082 | 2.0            | 0.8164  |
| 1    | 1      | 0      | 0.4082 | 4.0            | 1.6328  |
| 2    | 1      | 2      | 0.4082 | 4.0            | 1.6328  |
| 3    | 2      | 1      | 0.4082 | 6.0            | 2.4492  |
| 4    | 0      | 0      | 0.5000 | 2.0            | 1.0000  |
| 5    | 1      | 1      | 0.3333 | 4.0            | 1.3333  |
| 6    | 2      | 2      | 0.5000 | 6.0            | 3.0000  |

#### Final Output
```python
# sum the messages for each target node
out = [
  [1.0000+1.6328],
  [0.8164+2.4492+1.3333],
  [1.6328+3.0000]
]
```

#### Keypoints
- `W(x)`: decides **what to send**
- `norm`: decides **how much to send**
- `scatter_add_`: decides **where to send**

---

# One-line summary
Nodes with similar properties are embedded into similar vectors, while nodes with different properties are mapped to distinguishable representations.