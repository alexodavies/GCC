import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv  # Updated to use the new GATConv from dgl.nn


class UnsupervisedGAT(nn.Module):
    def __init__(
        self, node_input_dim, node_hidden_dim, edge_input_dim, num_layers, num_heads
    ):
        super(UnsupervisedGAT, self).__init__()
        assert node_hidden_dim % num_heads == 0
        self.layers = nn.ModuleList(
            [
                GATConv(
                    in_feats=node_input_dim if i == 0 else node_hidden_dim,
                    out_feats=node_hidden_dim // num_heads,
                    num_heads=num_heads,
                    feat_drop=0.0,
                    attn_drop=0.0,
                    negative_slope=0.2,
                    residual=False,
                    activation=F.leaky_relu if i + 1 < num_layers else None,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, g, n_feat):
        for i, layer in enumerate(self.layers):
            n_feat = layer(g, n_feat)
            if i + 1 < len(self.layers):
                n_feat = n_feat.flatten(1)  # Flatten the output as per GATConv usage
        return n_feat


if __name__ == "__main__":
    # Model definition
    model = UnsupervisedGAT(node_input_dim=64, node_hidden_dim=128, edge_input_dim=0, num_layers=2, num_heads=4)
    print(model)

    # Create a DGLGraph (replaced with dgl.graph)
    edges = ([0, 0, 1], [1, 2, 2])  # Edge pairs (source, destination)
    g = dgl.graph(edges)  # Replace deprecated DGLGraph with the new graph function

    # Random node features
    feat = torch.rand(3, 64)  # 3 nodes, 64-dimensional input feature

    # Model forward pass
    print(model(g, feat).shape)
