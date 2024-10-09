#!/usr/bin/env python
# encoding: utf-8
# File Name: train_graph_moco.py
# Author: Jiezhong Qiu
# Create Time: 2019/12/13 16:44

import argparse
import os
import time

import dgl
import numpy as np
import torch

from gcc.contrastive.criterions import NCESoftmaxLoss, NCESoftmaxLossNS
from gcc.contrastive.memory_moco import MemoryMoCo
from gcc.datasets import (
    GRAPH_CLASSIFICATION_DSETS,
    GraphClassificationDataset,
    NodeClassificationDataset,
    worker_init_fn,
)
from gcc.datasets.data_util import batcher
from gcc.models import GraphEncoder
from gcc.utils.misc import AverageMeter, adjust_learning_rate, warmup_linear

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt


def test_moco(train_loader, model, opt):
    """
    One epoch training for MoCo.
    """
    model.eval()

    emb_list = []
    for batch in train_loader:
        # print(idx, batch)
        graph_q, graph_k = batch
        graph_q = graph_q.to(opt.device)
        graph_k = graph_k.to(opt.device)

        with torch.no_grad():
            feat_q = model(graph_q)
            feat_k = model(graph_k)

        assert feat_q.shape == (graph_q.batch_size, opt.hidden_size)
        emb_list.append(((feat_q + feat_k) / 2).detach().cpu())
    
    return torch.cat(emb_list)


def main(args_test):
    # Load checkpoint
    if os.path.isfile(args_test.load_path):
        print(f"=> loading checkpoint '{args_test.load_path}'")
        checkpoint = torch.load(args_test.load_path, map_location="cpu")
        print(f"=> loaded successfully '{args_test.load_path}' (epoch {checkpoint['epoch']})")
    else:
        print(f"=> no checkpoint found at '{args_test.load_path}'")
        return

    # Load arguments from the checkpoint
    args = checkpoint["opt"]

    print(f"Use GPU: {args_test.gpu} for generation")
    args.gpu = args_test.gpu
    args.device = torch.device("cpu") # if args.gpu is None else torch.device(f"cuda:{args.gpu}")

    # Select the appropriate dataset
    if args_test.dataset in GRAPH_CLASSIFICATION_DSETS:
        train_dataset = GraphClassificationDataset(
            dataset=args_test.dataset,
            rw_hops=args.rw_hops,
            subgraph_size=args.subgraph_size,
            restart_prob=args.restart_prob,
            positional_embedding_size=args.positional_embedding_size,
        )
    else:
        train_dataset = NodeClassificationDataset(
            dataset=args_test.dataset,
            rw_hops=args.rw_hops,
            subgraph_size=args.subgraph_size,
            restart_prob=args.restart_prob,
            positional_embedding_size=args.positional_embedding_size,
        )

    args.batch_size = len(train_dataset)

    # Data loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        collate_fn=batcher(),
        shuffle=False,
        num_workers= 8 # args.num_workers,
    )

    # Create model and move it to the correct device
    model = GraphEncoder(
        positional_embedding_size=args.positional_embedding_size,
        max_node_freq=args.max_node_freq,
        max_edge_freq=args.max_edge_freq,
        max_degree=args.max_degree,
        freq_embedding_size=args.freq_embedding_size,
        degree_embedding_size=args.degree_embedding_size,
        output_dim=args.hidden_size,
        node_hidden_dim=args.hidden_size,
        edge_hidden_dim=args.hidden_size,
        num_layers=args.num_layer,
        num_step_set2set=args.set2set_iter,
        num_layer_set2set=args.set2set_lstm_layer,
        gnn_model=args.model,
        norm=args.norm,
        degree_input=True,
    )

    model = model.to(args.device)
    model.load_state_dict(checkpoint["model"])
    del checkpoint

    # Test MoCo and save embeddings
    emb = test_moco(train_loader, model, args)
    np.save(os.path.join(args.model_folder, f"{args_test.dataset}_embeddings.npy"), emb.numpy())

    print(emb)
    print(emb.numpy().shape)

    tsne = TSNE().fit_transform(emb.numpy())

    plt.scatter(tsne[:,0], tsne[:,1])

    plt.show()




if __name__ == "__main__":
    parser = argparse.ArgumentParser("Argument parser for training/testing MoCo")
    
    # Add arguments for model loading and dataset selection
    parser.add_argument("--load-path", type=str, required=True, help="Path to load the model checkpoint")
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=["dgl", "wikipedia", "blogcatalog", "usa_airport", 
                                 "brazil_airport", "europe_airport", "cora", 
                                 "citeseer", "pubmed"] + GRAPH_CLASSIFICATION_DSETS, 
                        help="Dataset to be used for evaluation")
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use")
    
    # Parse arguments and run main
    args_test = parser.parse_args()
    main(args_test)
