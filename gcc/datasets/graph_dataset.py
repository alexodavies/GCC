import math
import operator
import random
import dgl
import dgl.data
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from dgl.data import AmazonCoBuy, Coauthor
from gcc.datasets import data_util


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.graphs, _ = dgl.data.utils.load_graphs(
        dataset.dgl_graphs_file, dataset.jobs[worker_id]
    )
    dataset.length = sum([g.number_of_nodes() for g in dataset.graphs])
    np.random.seed(worker_info.seed % (2 ** 32))

def random_walk_with_restart(graph, seeds, restart_prob=0.8, max_nodes_per_seed=10):
    """
    Perform random walks with restart for a given list of seed nodes.

    Parameters:
    - graph (dgl.DGLGraph): The input DGL graph.
    - seeds (list of int): A list of seed nodes to start the walks from.
    - restart_prob (float): The probability to restart the walk from the seed node at each step.
    - max_nodes_per_seed (int): The maximum number of nodes visited for each seed during the walk.

    Returns:
    - traces (list of list of int): A list of walks, one per seed node, containing the sequence of nodes visited.
    """
    traces = []

    for seed_node in seeds:
        walk = [seed_node]  # Start the walk at the seed node
        current_node = seed_node
        
        for _ in range(max_nodes_per_seed - 1):  # Walk up to max_nodes_per_seed nodes
            if random.random() < restart_prob:
                # Restart the walk from the seed node
                current_node = seed_node
            else:
                # Get the neighbors of the current node
                neighbors = graph.successors(current_node).tolist()

                if len(neighbors) == 0:
                    # If the node has no neighbors, restart the walk
                    current_node = seed_node
                else:
                    # Choose a random neighbor to continue the walk
                    current_node = random.choice(neighbors)

            walk.append(current_node)

        traces.append(walk)

    return traces



class LoadBalanceGraphDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        rw_hops=64,
        restart_prob=0.8,
        positional_embedding_size=32,
        step_dist=[1.0, 0.0, 0.0],
        num_workers=1,
        dgl_graphs_file="./data/small.bin",
        num_samples=10000,
        num_copies=1,
        graph_transform=None,
        aug="rwr",
        num_neighbors=5,
    ):
        super(LoadBalanceGraphDataset, self).__init__()
        self.rw_hops = rw_hops
        self.num_neighbors = num_neighbors
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        self.num_samples = num_samples
        assert sum(step_dist) == 1.0
        assert positional_embedding_size > 1
        self.dgl_graphs_file = dgl_graphs_file
        graph_sizes = dgl.data.utils.load_labels(dgl_graphs_file)["graph_sizes"].tolist()
        print("load graph done")

        # Load balance algorithm to assign graphs to workers
        assert num_workers % num_copies == 0
        jobs = [list() for _ in range(num_workers // num_copies)]
        workloads = [0] * (num_workers // num_copies)
        graph_sizes = sorted(enumerate(graph_sizes), key=operator.itemgetter(1), reverse=True)
        for idx, size in graph_sizes:
            argmin = workloads.index(min(workloads))
            workloads[argmin] += size
            jobs[argmin].append(idx)
        self.jobs = jobs * num_copies
        self.total = self.num_samples * num_workers
        self.graph_transform = graph_transform
        assert aug in ("rwr", "ns")
        self.aug = aug

    def __len__(self):
        return self.num_samples * len(self.jobs)

    def __iter__(self):
        degrees = torch.cat([g.in_degrees().double() ** 0.75 for g in self.graphs])
        prob = degrees / torch.sum(degrees)
        samples = np.random.choice(
            self.length, size=self.num_samples, replace=True, p=prob.numpy()
        )
        for idx in samples:
            yield self.__getitem__(idx)

    def __getitem__(self, idx):
        graph_idx = 0
        node_idx = idx
        for i in range(len(self.graphs)):
            if node_idx < self.graphs[i].number_of_nodes():
                graph_idx = i
                break
            else:
                node_idx -= self.graphs[i].number_of_nodes()

        step = np.random.choice(len(self.step_dist), 1, p=self.step_dist)[0]
        if step == 0:
            other_node_idx = node_idx
        else:
            walk_result = dgl.sampling.random_walk(
                g=self.graphs[graph_idx], seeds=[node_idx], num_traces=1, num_hops=step
            )
            other_node_idx = walk_result[0][0][-1].item()

        if self.aug == "rwr":
            max_nodes_per_seed = max(
                self.rw_hops,
                int(
                    (
                        (self.graphs[graph_idx].in_degree(node_idx) ** 0.75)
                        * math.e
                        / (math.e - 1)
                        / self.restart_prob
                    )
                    + 0.5
                ),
            )
            traces = random_walk_with_restart(
                self.graphs[graph_idx],
                seeds=[node_idx, other_node_idx],
                restart_prob=self.restart_prob,
                max_nodes_per_seed=max_nodes_per_seed,
            )
        elif self.aug == "ns":
            neighbors1 = dgl.sampling.sample_neighbors(
                self.graphs[graph_idx], [node_idx], self.num_neighbors, replace=True
            )
            trace1 = [neighbors1.edges()[0]]
            neighbors2 = dgl.sampling.sample_neighbors(
                self.graphs[graph_idx], [other_node_idx], self.num_neighbors, replace=True
            )
            trace2 = [neighbors2.edges()[0]]
            traces = [trace1, trace2]

        graph_q = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=node_idx,
            trace=traces[0],
            positional_embedding_size=self.positional_embedding_size,
        )
        graph_k = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=other_node_idx,
            trace=traces[1],
            positional_embedding_size=self.positional_embedding_size,
        )
        if self.graph_transform:
            graph_q = self.graph_transform(graph_q)
            graph_k = self.graph_transform(graph_k)
        return graph_q, graph_k


class GraphDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rw_hops=64,
        subgraph_size=64,
        restart_prob=0.8,
        positional_embedding_size=32,
        step_dist=[1.0, 0.0, 0.0],
    ):
        super(GraphDataset, self).__init__()
        self.rw_hops = rw_hops
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        assert sum(step_dist) == 1.0
        assert positional_embedding_size > 1
        graphs, _ = dgl.data.utils.load_graphs("data_bin/dgl/lscc_graphs.bin", [0, 1, 2])
        for name in ["cs", "physics"]:
            g = Coauthor(name)[0]
            g.remove_nodes((g.in_degrees() == 0).nonzero().squeeze())
            # g.readonly()
            graphs.append(g)
        for name in ["computers", "photo"]:
            g = AmazonCoBuy(name)[0]
            g.remove_nodes((g.in_degrees() == 0).nonzero().squeeze())
            # g.readonly()
            graphs.append(g)
        print("load graph done")
        self.graphs = graphs
        self.length = sum([g.number_of_nodes() for g in self.graphs])

    def __len__(self):
        return self.length

    def _convert_idx(self, idx):
        graph_idx = 0
        node_idx = idx
        for i in range(len(self.graphs)):
            if node_idx < self.graphs[i].number_of_nodes():
                graph_idx = i
                break
            else:
                node_idx -= self.graphs[i].number_of_nodes()
        return graph_idx, node_idx

    def __getitem__(self, idx):
        graph_idx, node_idx = self._convert_idx(idx)

        step = np.random.choice(len(self.step_dist), 1, p=self.step_dist)[0]
        if step == 0:
            other_node_idx = node_idx
        else:
            walk_result = dgl.sampling.random_walk(
                g=self.graphs[graph_idx], seeds=[node_idx], num_traces=1, num_hops=step
            )
            other_node_idx = walk_result[0][0][-1].item()

        max_nodes_per_seed = max(
            self.rw_hops,
            int(
                (
                    self.graphs[graph_idx].out_degrees(node_idx)
                    * math.e
                    / (math.e - 1)
                    / self.restart_prob
                )
                + 0.5
            ),
        )
        traces = random_walk_with_restart(
            self.graphs[graph_idx],
            seeds=[node_idx, other_node_idx],
            restart_prob=self.restart_prob,
            max_nodes_per_seed=max_nodes_per_seed,
        )

        graph_q = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=node_idx,
            trace=traces[0],
            positional_embedding_size=self.positional_embedding_size,
        )
        graph_k = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=other_node_idx,
            trace=traces[1],
            positional_embedding_size=self.positional_embedding_size,
        )
        return graph_q, graph_k


class NodeClassificationDataset(GraphDataset):
    def __init__(
        self,
        dataset,
        rw_hops=64,
        subgraph_size=64,
        restart_prob=0.8,
        positional_embedding_size=32,
        step_dist=[1.0, 0.0, 0.0],
    ):
        self.rw_hops = rw_hops
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        assert positional_embedding_size > 1

        self.data = data_util.create_node_classification_dataset(dataset).data
        self.graphs = [self._create_dgl_graph(self.data)]
        self.length = sum([g.number_of_nodes() for g in self.graphs])
        self.total = self.length

    def _create_dgl_graph(self, data):
        graph = dgl.graph((data.edge_index[0], data.edge_index[1]))
        # graph.readonly()
        return graph


class GraphClassificationDataset(NodeClassificationDataset):
    def __init__(
        self,
        dataset,
        rw_hops=64,
        subgraph_size=64,
        restart_prob=0.8,
        positional_embedding_size=32,
        step_dist=[1.0, 0.0, 0.0],
    ):
        super(GraphClassificationDataset, self).__init__(
            dataset, rw_hops, subgraph_size, restart_prob, positional_embedding_size, step_dist
        )
        self.entire_graph = True
        self.dataset = data_util.create_graph_classification_dataset(dataset)
        self.graphs = self.dataset.graph_lists
        self.length = len(self.graphs)
        self.total = self.length

    def _convert_idx(self, idx):
        graph_idx = idx
        node_idx = self.graphs[idx].out_degrees().argmax().item()
        return graph_idx, node_idx


class GraphClassificationDatasetLabeled(GraphClassificationDataset):
    def __init__(
        self,
        dataset,
        rw_hops=64,
        subgraph_size=64,
        restart_prob=0.8,
        positional_embedding_size=32,
        step_dist=[1.0, 0.0, 0.0],
    ):
        super(GraphClassificationDatasetLabeled, self).__init__(
            dataset, rw_hops, subgraph_size, restart_prob, positional_embedding_size, step_dist
        )
        self.num_classes = self.dataset.num_labels
        self.entire_graph = True
        self.dict = [self.getitem(idx) for idx in range(len(self))]

    def __getitem__(self, idx):
        return self.dict[idx]

    def getitem(self, idx):
        graph_idx = idx
        node_idx = self.graphs[idx].out_degrees().argmax().item()

        traces = random_walk_with_restart(
            self.graphs[graph_idx],
            seeds=[node_idx],
            restart_prob=self.restart_prob,
            max_nodes_per_seed=self.rw_hops,
        )

        graph_q = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=node_idx,
            trace=traces[0],
            positional_embedding_size=self.positional_embedding_size,
            entire_graph=True,
        )
        return graph_q, self.dataset.graph_labels[graph_idx].item()


class NodeClassificationDatasetLabeled(NodeClassificationDataset):
    def __init__(
        self,
        dataset,
        rw_hops=64,
        subgraph_size=64,
        restart_prob=0.8,
        positional_embedding_size=32,
        step_dist=[1.0, 0.0, 0.0],
        cat_prone=False,
    ):
        super(NodeClassificationDatasetLabeled, self).__init__(
            dataset, rw_hops, subgraph_size, restart_prob, positional_embedding_size, step_dist
        )
        assert len(self.graphs) == 1
        self.num_classes = self.data.y.shape[1]

    def __getitem__(self, idx):
        graph_idx = 0
        node_idx = idx
        for i in range(len(self.graphs)):
            if node_idx < self.graphs[i].number_of_nodes():
                graph_idx = i
                break
            else:
                node_idx -= self.graphs[i].number_of_nodes()

        traces = random_walk_with_restart(
            self.graphs[graph_idx],
            seeds=[node_idx],
            restart_prob=self.restart_prob,
            max_nodes_per_seed=self.rw_hops,
        )

        graph_q = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=node_idx,
            trace=traces[0],
            positional_embedding_size=self.positional_embedding_size,
        )
        return graph_q, self.data.y[idx].argmax().item()


if __name__ == "__main__":
    num_workers = 1
    import psutil

    mem = psutil.virtual_memory()
    print(mem.used / 1024 ** 3)
    graph_dataset = LoadBalanceGraphDataset(
        num_workers=num_workers, aug="ns", rw_hops=4, num_neighbors=5
    )
    mem = psutil.virtual_memory()
    print(mem.used / 1024 ** 3)
    graph_loader = torch.utils.data.DataLoader(
        graph_dataset,
        batch_size=1,
        collate_fn=data_util.batcher(),
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
    )
    mem = psutil.virtual_memory()
    print(mem.used / 1024 ** 3)
    for step, batch in enumerate(graph_loader):
        print("bs", batch[0].batch_size)
        print("n=", batch[0].number_of_nodes())
        print("m=", batch[0].number_of_edges())
        mem = psutil.virtual_memory()
        print(mem.used / 1024 ** 3)
        print(batch[0].ndata["pos_undirected"])
