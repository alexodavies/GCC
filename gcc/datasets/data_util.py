#!/usr/bin/env python
# encoding: utf-8
# File Name: data_util.py
# Author: Jiezhong Qiu
# Create Time: 2019/12/30 14:20
# TODO:

import io
import itertools
import os
import os.path as osp
from collections import defaultdict, namedtuple

import dgl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.sparse as sparse
import sklearn.preprocessing as preprocessing
import torch
import torch.nn.functional as F
from dgl.data.tu import TUDataset
from scipy.sparse import linalg
from ogb.graphproppred import DglGraphPropPredDataset


# Moved batcher_dev outside of batcher
def batcher_dev(graph_data):
    graph_q, graph_k = zip(*graph_data)
    batched_graph_q = dgl.batch(graph_q)
    batched_graph_k = dgl.batch(graph_k)
    return batched_graph_q, batched_graph_k

def batcher():
    return batcher_dev


def labeled_batcher():
    def batcher_dev(batch):
        graph_q, label = zip(*batch)
        graph_q = dgl.batch(graph_q)
        return graph_q, torch.LongTensor(label)

    return batcher_dev


Data = namedtuple("Data", ["x", "edge_index", "y"])


def create_graph_classification_dataset(dataset_name):
    name = {
        "imdb-binary": "IMDB-BINARY",
        "imdb-multi": "IMDB-MULTI",
        "rdt-b": "REDDIT-BINARY",
        "rdt-5k": "REDDIT-MULTI-5K",
        "collab": "COLLAB",
    }[dataset_name]
    dataset = TUDataset(name)
    dataset.num_labels = dataset.num_labels[0]
    dataset.graph_labels = dataset.graph_labels.squeeze()
    return dataset



class Edgelist(object):
    """
    A class to represent an edge list and associated node labels for a graph.
    Attributes:
    -----------
    name : str
        The name of the dataset.
    data : Data
        The processed data containing edge indices and node labels.
    transform : None
        Placeholder for potential data transformations.
    Methods:
    --------
    __init__(self, root, name):
        Initializes the Edgelist object with the given root directory and dataset name.
    get(self, idx):
        Returns the processed data. Only supports idx == 0.
    _preprocess(self, edge_list_path, node_label_path):
        Processes the edge list and node label files to create edge indices and node labels.
    """

    def __init__(self, root, name):
        self.name = name
        edge_list_path = os.path.join(root, name + ".edgelist")
        node_label_path = os.path.join(root, name + ".nodelabel")
        edge_index, y, self.node2id = self._preprocess(edge_list_path, node_label_path)
        self.data = Data(x=None, edge_index=edge_index, y=y)
        self.transform = None

    def get(self, idx):
        assert idx == 0
        return self.data

    def _preprocess(self, edge_list_path, node_label_path):
        with open(edge_list_path) as f:
            edge_list = []
            node2id = defaultdict(int)
            for line in f:
                x, y = list(map(int, line.split()))
                # Reindex
                if x not in node2id:
                    node2id[x] = len(node2id)
                if y not in node2id:
                    node2id[y] = len(node2id)
                edge_list.append([node2id[x], node2id[y]])
                edge_list.append([node2id[y], node2id[x]])

        num_nodes = len(node2id)
        with open(node_label_path) as f:
            nodes = []
            labels = []
            label2id = defaultdict(int)
            for line in f:
                x, label = list(map(int, line.split()))
                if label not in label2id:
                    label2id[label] = len(label2id)
                nodes.append(node2id[x])
                if "hindex" in self.name:
                    labels.append(label)
                else:
                    labels.append(label2id[label])
            if "hindex" in self.name:
                median = np.median(labels)
                labels = [int(label > median) for label in labels]
        assert num_nodes == len(set(nodes))
        y = torch.zeros(num_nodes, len(label2id))
        y[nodes, labels] = 1
        return torch.LongTensor(edge_list).t(), y, node2id


class SSSingleDataset(object):
    def __init__(self, root, name):
        edge_index = self._preprocess(root, name)
        self.data = Data(x=None, edge_index=edge_index, y=None)
        self.transform = None

    def get(self, idx):
        assert idx == 0
        return self.data

    def _preprocess(self, root, name):
        graph_path = os.path.join(root, name + ".graph")

        with open(graph_path) as f:
            edge_list = []
            node2id = defaultdict(int)
            f.readline()
            for line in f:
                x, y, t = list(map(int, line.split()))
                # Reindex
                if x not in node2id:
                    node2id[x] = len(node2id)
                if y not in node2id:
                    node2id[y] = len(node2id)
                # repeat t times
                for _ in range(t):
                    # to undirected
                    edge_list.append([node2id[x], node2id[y]])
                    edge_list.append([node2id[y], node2id[x]])

        num_nodes = len(node2id)

        return torch.LongTensor(edge_list).t()

class SSDataset(object):
    def __init__(self, root, name1, name2):
        edge_index_1, dict_1, self.node2id_1 = self._preprocess(root, name1)
        edge_index_2, dict_2, self.node2id_2 = self._preprocess(root, name2)
        self.data = [
            Data(x=None, edge_index=edge_index_1, y=dict_1),
            Data(x=None, edge_index=edge_index_2, y=dict_2),
        ]
        self.transform = None

    def get(self, idx):
        assert idx == 0
        return self.data

    def _preprocess(self, root, name):
        dict_path = os.path.join(root, name + ".dict")
        graph_path = os.path.join(root, name + ".graph")

        with open(graph_path) as f:
            edge_list = []
            node2id = defaultdict(int)
            f.readline()
            for line in f:
                x, y, t = list(map(int, line.split()))
                # Reindex
                if x not in node2id:
                    node2id[x] = len(node2id)
                if y not in node2id:
                    node2id[y] = len(node2id)
                # repeat t times
                for _ in range(t):
                    # to undirected
                    edge_list.append([node2id[x], node2id[y]])
                    edge_list.append([node2id[y], node2id[x]])

        name_dict = dict()
        with open(dict_path) as f:
            for line in f:
                name, str_x = line.split("\t")
                x = int(str_x)
                if x not in node2id:
                    node2id[x] = len(node2id)
                name_dict[name] = node2id[x]

        num_nodes = len(node2id)

        return torch.LongTensor(edge_list).t(), name_dict, node2id

def create_node_classification_dataset(dataset_name):
    if "airport" in dataset_name:
        return Edgelist(
            "data/struc2vec/",
            {
                "usa_airport": "usa-airports",
                "brazil_airport": "brazil-airports",
                "europe_airport": "europe-airports",
            }[dataset_name],
        )
    elif "h-index" in dataset_name:
        return Edgelist(
            "data/hindex/",
            {
                "h-index-rand-1": "aminer_hindex_rand1_5000",
                "h-index-top-1": "aminer_hindex_top1_5000",
                "h-index": "aminer_hindex_rand20intop200_5000",
            }[dataset_name],
        )
    elif dataset_name in ["kdd", "icdm", "sigir", "cikm", "sigmod", "icde"]:
        return SSSingleDataset("data/panther/", dataset_name)
    else:
        print(f"{dataset_name} not available")
        raise NotImplementedError

def _rwr_trace_to_dgl_graph(g, seed, trace, positional_embedding_size, entire_graph=False):
    """
    Convert a random walk trace to a DGL subgraph.

    Parameters:
    - g (dgl.DGLGraph): The original graph.
    - seed (int): The seed node from which the walk started.
    - trace (list of list): A list of random walk traces.
    - positional_embedding_size (int): The size of positional embeddings.
    - entire_graph (bool): Whether to return the entire graph as the subgraph.

    Returns:
    - A DGL subgraph based on the nodes in the trace.
    """
    # Ensure each element in trace is converted to a tensor if necessary
    trace_tensors = [torch.tensor(t) if not isinstance(t, torch.Tensor) else t for t in trace]
    trace_tensors = torch.tensor(trace_tensors)

    # Concatenate the traces to get the subgraph nodes
    
    subv = torch.unique(trace_tensors).tolist()

    # Ensure the seed node is the first node
    try:
        subv.remove(seed)
    except ValueError:
        pass
    subv = [seed] + subv

    # Generate subgraph: either the entire graph or a subgraph with the selected nodes
    if entire_graph:
        subg = g.subgraph(g.nodes())
    else:
        subg = g.subgraph(subv)

    # Add positional embeddings or other features to the subgraph
    subg = _add_undirected_graph_positional_embedding(subg, positional_embedding_size)

    # Set the seed node indicator in the graph's node data
    subg.ndata["seed"] = torch.zeros(subg.number_of_nodes(), dtype=torch.long)
    if entire_graph:
        subg.ndata["seed"][seed] = 1
    else:
        subg.ndata["seed"][0] = 1

    return subg


def eigen_decomposision(n, k, laplacian, hidden_size, retry):
    if k <= 0:
        return torch.zeros(n, hidden_size)
    laplacian = laplacian.astype("float64")
    ncv = min(n, max(2 * k + 1, 20))
    # follows https://stackoverflow.com/questions/52386942/scipy-sparse-linalg-eigsh-with-fixed-seed
    v0 = np.random.rand(n).astype("float64")
    for i in range(retry):
        try:
            s, u = linalg.eigsh(laplacian, k=k, which="LA", ncv=ncv, v0=v0)
        except sparse.linalg.ArpackError:
            # print("arpack error, retry=", i)
            ncv = min(ncv * 2, n)
            if i + 1 == retry:
                sparse.save_npz("arpack_error_sparse_matrix.npz", laplacian)
                u = torch.zeros(n, k)
        else:
            break
    x = preprocessing.normalize(u, norm="l2")
    x = torch.from_numpy(x.astype("float32"))
    x = F.pad(x, (0, hidden_size - k), "constant", 0)
    return x


def _add_undirected_graph_positional_embedding(g, hidden_size, retry=10):
    """
    Adds positional embeddings to the graph using eigenvectors of the normalized graph Laplacian.

    Parameters:
    - g (dgl.DGLGraph): The input graph.
    - hidden_size (int): The size of the positional embeddings.
    - retry (int): The number of retries for eigenvalue decomposition in case of failure.

    Returns:
    - g (dgl.DGLGraph): The graph with positional embeddings added to the node features.
    """
    n = g.number_of_nodes()

    # Get the CSR adjacency matrix as tensors
    indptr, indices, _ = g.adj_tensors(fmt="csr")

    # Convert CSR tensors to a SciPy sparse matrix
    adj = sparse.csr_matrix((torch.ones_like(indices).numpy(), indices.numpy(), indptr.numpy()), shape=(n, n))

    # Normalize for the Laplacian
    norm = sparse.diags(g.in_degrees().numpy().clip(1) ** -0.5, dtype=float)
    
    # Normalized Laplacian calculation
    laplacian = norm @ adj @ norm

    # Perform eigen decomposition
    k = min(n - 2, hidden_size)
    x = eigen_decomposision(n, k, laplacian, hidden_size, retry)

    # Add positional embeddings to the graph's node data
    g.ndata["pos_undirected"] = x.float()

    return g
