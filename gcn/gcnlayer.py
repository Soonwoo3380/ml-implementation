{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3dcb3bae",
   "metadata": {},
   "outputs": [],
   "source": [
    "class GCNConv(nn.Module):\n",
    "    def __init__(self, in_channels, out_channels):\n",
    "        self.W = nn.Linear(in_channels, out_channels, bias=False)\n",
    "\n",
    "    def forward(self, x, edge_index, num_nodes):\n",
    "        edge_index = add_self_loops(edge_index, num_nodes=num_nodes)\n",
    "\n",
    "        row, col = edge_index\n",
    "        deg = degree(col, num_nodes=num_nodes, dtype=x.dtype)\n",
    "        deg_inv_sqrt = deg.pow(-0.5)\n",
    "\n",
    "        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]\n",
    "\n",
    "        x = self.W(x)\n",
    "\n",
    "        out = torch.zeros(num_nodes, x.size(1), device=x.device)\n",
    "        out.scatter_add_()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "topicgraph_py310",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.19"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
