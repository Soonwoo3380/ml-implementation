# Graph Convolutional Networks

**Paper**  
https://arxiv.org/abs/1609.02907

### Understanding GCN Message Passing Step by Step
This repository demonstrates the internal workings of a GCN layer by implementing it from scratch and printing intermidiate results.

The goal is to clearly understand how node features are updated through:
- self-loop addition
- degree normalization
- linear transformation
- message passing
- aggregation

### Overview
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
  
#### Edges (undirected)
- 0--1
- 1--2

#### PyG-style 'edge_index'
```python
edge_index = torch.tensor([
    [0, 1, 1, 2],
    [1, 0, 2, 1]
])
