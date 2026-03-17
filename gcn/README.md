## GCN (Graph Convolutional Networks)

**Paper**  
https://arxiv.org/abs/1609.02907

### One-line Summary
Applying on graphs that contain labeled and **non-labeled nodes**.

### Key Ideas
- $Z = \tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}X\Theta$
- $Z = \text{softmax}\left(\hat{A}\ \text{ReLU}\ \left(\hat{A}XW^{(0)}\right)W^{(1)}\right)$
- $L = -\sum_{l \in Y_L}\sum_{f} Y_{lf}\ln Z_{lf}$
  - evaluate the error over only on labeled nodes.
  - the gradient propagates to non-labeled nodes via $\hat{A}$.


### Result
The method for graph classification tasks outperforms related methods.