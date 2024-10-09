#!/usr/bin/env python
# encoding: utf-8

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import Set2Set

from gcc.models.gat import UnsupervisedGAT
from gcc.models.gin import UnsupervisedGIN
from gcc.models.mpnn import UnsupervisedMPNN


class OGBGraphEncoder(nn.Module):
    """
    MPNN from
    `Neural Message Passing for Quantum Chemistry <https://arxiv.org/abs/1704.01212>`__

    Parameters
    ----------
    node_input_dim : int
        Dimension of input node feature, default to be determined from the dataset.
    edge_input_dim : int
        Dimension of input edge feature, default to be determined from the dataset.
    output_dim : int
        Dimension of prediction, default to be 32.
    node_hidden_dim : int
        Dimension of node feature in hidden layers, default to be 64.
    edge_hidden_dim : int
        Dimension of edge feature in hidden layers, default to be 128.
    num_step_message_passing : int
        Number of message passing steps, default to be 6.
    num_step_set2set : int
        Number of set2set steps
    num_layer_set2set : int
        Number of set2set layers
    """

    def __init__(
        self,
        node_input_dim,
        edge_input_dim,
        output_dim=32,
        node_hidden_dim=32,
        edge_hidden_dim=32,
        num_layers=6,
        num_heads=4,
        num_step_set2set=6,
        num_layer_set2set=3,
        norm=False,
        gnn_model="mpnn",
        lstm_as_gate=False,
    ):
        super(OGBGraphEncoder, self).__init__()

        # Initialize the GNN model (GIN, MPNN, or GAT)
        if gnn_model == "mpnn":
            self.gnn = UnsupervisedMPNN(
                output_dim=output_dim,
                node_input_dim=node_input_dim,
                node_hidden_dim=node_hidden_dim,
                edge_input_dim=edge_input_dim,
                edge_hidden_dim=edge_hidden_dim,
                num_step_message_passing=num_layers,
                lstm_as_gate=lstm_as_gate,
            )
        elif gnn_model == "gat":
            self.gnn = UnsupervisedGAT(
                node_input_dim=node_input_dim,
                node_hidden_dim=node_hidden_dim,
                edge_input_dim=edge_input_dim,
                num_layers=num_layers,
                num_heads=num_heads,
            )
        elif gnn_model == "gin":
            self.gnn = UnsupervisedGIN(
                num_layers=num_layers,
                num_mlp_layers=2,
                input_dim=node_input_dim,
                hidden_dim=node_hidden_dim,
                output_dim=output_dim,
                final_dropout=0.5,
                learn_eps=False,
                graph_pooling_type="sum",
                neighbor_pooling_type="sum",
                use_selayer=False,
            )
        self.gnn_model = gnn_model

        # Set2Set layer for pooling node features into a graph-level representation
        self.set2set = Set2Set(node_hidden_dim, num_step_set2set, num_layer_set2set)

        # Linear layers for readout after Set2Set
        self.lin_readout = nn.Sequential(
            nn.Linear(2 * node_hidden_dim, node_hidden_dim),
            nn.ReLU(),
            nn.Linear(node_hidden_dim, output_dim),
        )

        self.norm = norm

    def forward(self, g, return_all_outputs=False):
        """Predict graph labels

        Parameters
        ----------
        g : DGLGraph
            Input DGLGraph.
        n_feat : tensor of shape (num_nodes, node_feature_dim)
            Node features.
        e_feat : tensor of shape (num_edges, edge_feature_dim)
            Edge features.

        Returns
        -------
        res : Predicted graph-level representations.
        """

        n_feat = g.ndata['feat'].float()  # Assuming OGB node features are stored in 'feat'
        
        # Edge features may be optional, but cast to float if available
        e_feat = g.edata['feat'].float() if 'feat' in g.edata else None


        # Pass the features through the GNN model
        if self.gnn_model == "gin":
            x, all_outputs = self.gnn(g, n_feat, e_feat)
        else:
            x, all_outputs = self.gnn(g, n_feat, e_feat), None

            # Pool node features into a graph-level representation using Set2Set
            x = self.set2set(g, x)
            x = self.lin_readout(x)

        # Normalize the output if specified
        if self.norm:
            x = F.normalize(x, p=2, dim=-1, eps=1e-5)

        if return_all_outputs:
            return x, all_outputs
        else:
            return x


if __name__ == "__main__":
    # Example of using GraphEncoder
    model = GraphEncoder(gnn_model="gin", node_input_dim=16, edge_input_dim=10)
    print(model)

    # Example DGLGraph with node and edge features
    g = dgl.DGLGraph()
    g.add_nodes(3)
    g.add_edges([0, 0, 1, 2], [1, 2, 2, 1])

    g.ndata["feat"] = torch.rand(3, 16)  # Node features
    g.edata["feat"] = torch.rand(4, 10)  # Edge features

    # Batch graphs and run through the model
    g = dgl.batch([g, g, g])
    y = model(g)
    print(y.shape)
    print(y)
