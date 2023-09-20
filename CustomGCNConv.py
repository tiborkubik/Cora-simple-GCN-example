import torch
from torch_geometric.typing import Adj
from torch_geometric.nn.inits import zeros
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class CustomGCNConv(MessagePassing):
    def __init__(self, in_features: int, out_features: int, self_loops: bool = True) -> None:
        super().__init__()  # Default aggregation is 'Add', which reflects the nature of GCN.

        self.W = torch.nn.Linear(in_features=in_features, out_features=out_features)
        self.bias = torch.nn.Parameter(torch.empty(out_features))

        self.self_loops = self_loops  # Node self-loops are used in original implementation, to resemble classic conv.

        super().reset_parameters()
        self.W.reset_parameters()
        zeros(self.bias)

    def forward(self, x_k_1: torch.Tensor, edge_index: Adj) -> torch.Tensor:
        # 1) Add self-loops to the adjacency matrix.
        if self.self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x_k_1.size(0))

        # 2) Feature transformation, learned params here.
        x_k_1_t = self.W(x_k_1)

        # 3) Normalization.
        row, col = edge_index
        deg = degree(col, x_k_1_t.size(0), dtype=x_k_1_t.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        deg = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 4) Propagate internally calls message(), aggregate() and update().
        x_k = self.propagate(edge_index=edge_index, x=x_k_1_t, deg=deg)

        # 5) Last step of equation: add bias.
        x_k += self.bias

        return x_k

    def message(self, x_j: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
        # This is called under the hood for each adjacent node.
        return deg.view(-1, 1) * x_j

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        # Nothing special here in GCN, just added for completeness.
        return aggr_out
