import argparse
import os
import numpy as np
import dgl
import torch
import torch.nn as nn
from ogb.graphproppred import DglGraphPropPredDataset, Evaluator
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from gcc.models import GraphEncoder # , OGBGraphEncoder
from gcc.utils.misc import AverageMeter
from tqdm import tqdm
from gcc.datasets import OGBGraphClassificationDataset
from gcc.datasets.data_util import _add_undirected_graph_positional_embedding
from gcc.datasets.graph_dataset import random_walk_with_restart
import wandb

GRAPH_CLASSIFICATION_DSETS = ["ogbg-molclintox", "ogbg-molbbbp", "ogbg-molbace", "ogbg-molhiv"]

def setup_wandb(cfg, offline = False, name = None):
    """
    Uses a config dictionary to initialise wandb to track sampling.
    Requires a wandb account, https://wandb.ai/

    params: cfg: argparse Namespace

    returns:
    param: cfg: same config
    """

    # print(cfg.dataset, cfg.load_path)
    dataset = cfg.dataset
    model = cfg.load_path.split('/')[1]

    kwargs = {'name': "no-feats-" + dataset + "-" + model, 'project': f'gcl-validation-reiteration', 'config': cfg,
              'settings': wandb.Settings(_disable_stats=False), 'reinit': True, 'entity':'hierarchical-diffusion',
              'mode':'online' if offline else 'online'}
    wandb.init(**kwargs)
    wandb.save('*.txt')

    return cfg

def parse_option():

    # fmt: off
    parser = argparse.ArgumentParser("argument for training")

    parser.add_argument("--print-freq", type=int, default=10, help="print frequency")
    parser.add_argument("--tb-freq", type=int, default=250, help="tb frequency")
    parser.add_argument("--save-freq", type=int, default=1, help="save frequency")
    parser.add_argument("--batch-size", type=int, default=512, help="batch_size")
    parser.add_argument("--num-workers", type=int, default=12, help="num of workers to use")
    parser.add_argument("--num-copies", type=int, default=6, help="num of dataset copies that fit in memory")
    parser.add_argument("--num-samples", type=int, default=2000, help="num of samples per batch per worker")
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")

    # optimization
    parser.add_argument("--optimizer", type=str, default='adam', choices=['sgd', 'adam', 'adagrad'], help="optimizer")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--lr_decay_epochs", type=str, default="120,160,200", help="where to decay lr, can be a list")
    parser.add_argument("--lr_decay_rate", type=float, default=0.0, help="decay rate for learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 for adam")
    parser.add_argument("--beta2", type=float, default=0.999, help="beta2 for Adam")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--clip-norm", type=float, default=1.0, help="clip norm")

    # resume
    parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")

    # augmentation setting
    parser.add_argument("--aug", type=str, default="1st", choices=["1st", "2nd", "all"])

    parser.add_argument("--exp", type=str, default="")

    # dataset definition
    parser.add_argument("--dataset", type=str, default="dgl", choices=["dgl", "wikipedia", "blogcatalog", "usa_airport", "brazil_airport", "europe_airport", "cora", "citeseer", "pubmed", "kdd", "icdm", "sigir", "cikm", "sigmod", "icde", "h-index-rand-1", "h-index-top-1", "h-index"]  + GRAPH_CLASSIFICATION_DSETS)

    # model definition
    parser.add_argument("--model", type=str, default="gin", choices=["gat", "mpnn", "gin"])
    # other possible choices: ggnn, mpnn, graphsage ...
    parser.add_argument("--num-layer", type=int, default=5, help="gnn layers")
    parser.add_argument("--readout", type=str, default="avg", choices=["avg", "set2set"])
    parser.add_argument("--set2set-lstm-layer", type=int, default=3, help="lstm layers for s2s")
    parser.add_argument("--set2set-iter", type=int, default=6, help="s2s iteration")
    parser.add_argument("--norm", action="store_true", default=True, help="apply 2-norm on output feats")

    # loss function
    parser.add_argument("--nce-k", type=int, default=32)
    parser.add_argument("--nce-t", type=float, default=0.07)

    # random walk
    parser.add_argument("--rw-hops", type=int, default=256)
    parser.add_argument("--subgraph-size", type=int, default=128)
    parser.add_argument("--restart-prob", type=float, default=0.8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--positional-embedding-size", type=int, default=32)
    parser.add_argument("--max-node-freq", type=int, default=16)
    parser.add_argument("--max-edge-freq", type=int, default=16)
    parser.add_argument("--max-degree", type=int, default=512)
    parser.add_argument("--freq-embedding-size", type=int, default=16)
    parser.add_argument("--degree-embedding-size", type=int, default=16)

    # specify folder
    parser.add_argument("--model-path", type=str, default=None, help="path to save model")
    parser.add_argument("--tb-path", type=str, default=None, help="path to tensorboard")
    parser.add_argument("--load-path", type=str, default=None, help="loading checkpoint at test time")

    # memory setting
    parser.add_argument("--moco", action="store_true", help="using MoCo (otherwise Instance Discrimination)")

    # finetune setting
    parser.add_argument("--finetune", action="store_true")

    parser.add_argument("--alpha", type=float, default=0.999, help="exponential moving average weight")

    # GPU setting
    parser.add_argument("--gpu", default=None, type=int, nargs='+', help="GPU id to use.")

    # cross validation
    parser.add_argument("--seed", type=int, default=0, help="random seed.")
    parser.add_argument("--fold-idx", type=int, default=0, help="random seed.")
    parser.add_argument("--cv", action="store_true")
    # fmt: on

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(",")
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    return opt

def add_random_walk_positional_encodings(graphs, pos_enc_dim, rw_hops, restart_prob):
    """
    Adds random walk-based positional encodings to the nodes in each graph.

    Args:
    - graphs (list of DGLGraph): List of graphs to compute positional encodings for.
    - pos_enc_dim (int): The size of the positional encoding vectors.
    - rw_hops (int): Maximum number of hops for the random walk.
    - restart_prob (float): Probability to restart the random walk.

    Returns:
    - Updated list of graphs with positional encodings added to the node data.
    """
    for g in tqdm(graphs):
        seed_nodes = list(range(g.number_of_nodes()))  # Use all nodes as seeds for RWR
        traces = random_walk_with_restart(g, seed_nodes, restart_prob, max_nodes_per_seed=rw_hops)
        g = _add_undirected_graph_positional_embedding(g, pos_enc_dim)  # Add PEs via Laplacian eigenvectors
    return graphs

def load_ogb_dataset_with_positional_embeddings(dataset_name, pos_enc_dim, rw_hops, restart_prob):
    dataset = DglGraphPropPredDataset(name=dataset_name)
    dataset = OGBGraphClassificationDataset(dataset)
    # evaluator = Evaluator(name=dataset_name)

    # print("Adding random-walk positional embeddings to the dataset graphs...")
    # dataset.graphs = add_random_walk_positional_encodings(dataset.graphs, pos_enc_dim, rw_hops, restart_prob)

    return dataset

def train_finetune(epoch, train_loader, model, output_layer, criterion, optimizer, device):
    model.train()
    output_layer.train()
    loss_meter = AverageMeter()
    f1_meter = AverageMeter()
    label_n, label_sum = 0, 0
    for idx, (graphs, _, labels) in enumerate(train_loader):
        # graphs = (graph1.to(device), graph2.to(device))
        optimizer.zero_grad()
        graphs = graphs.to(device)
        labels = labels.to(device)#.view(-1)
        if labels.shape[1] > 1:
            labels = labels.argmax(dim=1).reshape(-1,1)

        feat_q = model(graphs)
        out = output_layer(feat_q)
        out = torch.sigmoid(out)
        # print(out)
        # if out.shape[1] > 1:
        #     out = torch.softmax(out, dim = 1)

        loss = criterion(out, labels.to(torch.float32))


        
        loss.backward()
        optimizer.step()

        # if out.shape[1] > 1:
        #     preds = out.argmax(dim=1)
        # else:
        #     preds = torch.round(out)
        # if labels.shape[1] > 1:
        #     labels = labels.argmax(dim=1)
        preds = torch.round(out)
        f1 = accuracy_score(labels.cpu().numpy(), preds.detach().cpu().numpy())

        loss_meter.update(loss.item(), graphs.batch_size)
        f1_meter.update(f1, graphs.batch_size)

        # if (idx + 1) % 10 == 0:
        #     print(f"Epoch [{epoch}] Iter [{idx + 1}/{len(train_loader)}] Loss: {loss_meter.avg:.4f} Accuracy: {f1_meter.avg:.4f}")


        
        label_n += labels.cpu().numpy().shape[0]
        label_sum += torch.sum(labels)

    wandb.log({"Train Loss":loss_meter.avg,
    "Accuracy Train":f1_meter.avg,
        "Label Weighting Train": label_sum/label_n})
    return loss_meter.avg, f1_meter.avg

def test_finetune(valid_loader, model, output_layer, criterion, device, evaluator):
    model.eval()
    output_layer.eval()
    loss_meter = AverageMeter()
    f1_meter = AverageMeter()
    roc_meter = AverageMeter()
    label_n, label_sum = 0, 0
    with torch.no_grad():
        for idx, (graphs, _, labels) in enumerate(valid_loader):
            # graphs = (graph1.to(device), graph2.to(device))
            graphs = graphs.to(device)
            labels = labels.to(device)
            if labels.shape[1] > 1:
                labels = labels.argmax(dim=1).reshape(-1,1)


            feat_q = model(graphs)
            out = output_layer(feat_q)
            out = torch.sigmoid(out)
            # print(out)
            # if out.shape[1] > 1:
            #     out = torch.softmax(out, dim = 1)

            loss = criterion(out, labels.to(torch.float32))

            # if out.shape[1] > 1:
            #     preds = out.argmax(dim=1)
            # else:
            #     preds = torch.round(out)
            # if labels.shape[1] > 1:
            #     labels = labels.argmax(dim=1)
            preds = torch.round(out)
            f1 = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())

            loss_meter.update(loss.item(), graphs.batch_size)
            f1_meter.update(f1, graphs.batch_size)
            # print(labels.cpu().numpy()[:10], preds.cpu().numpy()[:10])
            try:
                roc = roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy())
                roc_meter.update(roc, graphs.batch_size)
            except:
                # print(labels.cpu().numpy())
                print("ROC Error")
                pass
            label_n += labels.cpu().numpy().shape[0]
            label_sum += torch.sum(labels)
    wandb.log({"Val Loss":loss_meter.avg,
              "Accuracy":f1_meter.avg,
              "ROC":roc_meter.avg,
              "Label Weighting": label_sum/label_n})
    # print(f"Validation Loss: {loss_meter.avg:.4f} Accuracy: {f1_meter.avg:.4f} ROC: {roc_meter.avg:.4f}")
    return loss_meter.avg, f1_meter.avg, roc_meter.avg

def get_model(node_input_dim, edge_input_dim, num_tasks, device, args):
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
        norm=args.norm,
        gnn_model=args.model,
        degree_input=True,
    )

    if args.load_path:
        load_gnn_layers_only(model, args.load_path)

    output_layer = nn.Linear(in_features=64, out_features=num_tasks).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(output_layer.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    return model, criterion, optimizer, output_layer

def load_gnn_layers_only(model, pretrained_path):
    pretrained_dict = torch.load(pretrained_path, map_location="cpu")['model']
    model_dict = model.state_dict()
    pretrained_gnn_dict = {k: v for k, v in pretrained_dict.items() if 'gnn' in k}
    pretrained_gnn_dict = {k: v for k, v in pretrained_gnn_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_gnn_dict)
    model.load_state_dict(model_dict)

def main():
    args = parse_option()
    setup_wandb(args)
    device = "cuda" if torch.cuda.is_available() else "cpu" # torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print(f"Loading OGB dataset {args.dataset} with random-walk positional embeddings for fine-tuning")
    dataset = load_ogb_dataset_with_positional_embeddings(args.dataset, 0, 0, 0) #args.pos_enc_dim, args.rw_hops, args.restart_prob)
    print(dataset[0])

    split_idx = dataset.split_idx
    train_idx = split_idx['train']
    valid_idx = split_idx['valid']
    test_idx = split_idx['test']

    graph = dataset[0][0]  # First graph in the dataset
    node_input_dim = graph.ndata['pos_undirected'].shape[1]  # Use positional embeddings
    edge_input_dim = graph.edata['feat'].shape[1] if 'feat' in graph.edata else 0
    num_tasks = 1 # dataset.num_tasks

    train_loader = dgl.dataloading.GraphDataLoader(
        [dataset[i] for i in train_idx],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    valid_loader = dgl.dataloading.GraphDataLoader(
        [dataset[i] for i in valid_idx],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    test_loader = dgl.dataloading.GraphDataLoader(
        [dataset[i] for i in test_idx],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    # os.makedirs(args.model_path, exist_ok=True)

    best_val_roc = 0
    test_roc_at_best_val = 0

    model, criterion, optimizer, output_layer = get_model(node_input_dim, edge_input_dim, num_tasks, device, args)
    model = model.to(device)
    output_layer = output_layer.to(device)

    best_f1 = 0
    rocs = []
    for i in tqdm(range(10), colour='red'):
        model, criterion, optimizer, output_layer = get_model(node_input_dim, edge_input_dim, 1, device, args)
        model = model.to(device)
        output_layer = output_layer.to(device)
        pbar_epochs = tqdm(range(0, args.epochs), colour='green', leave = False)
        for epoch in pbar_epochs:
            train_finetune(epoch, train_loader, model, output_layer, criterion, optimizer, device)
            if epoch % 10 == 0 or epoch == 0:
                val_loss, val_f1, roc = test_finetune(valid_loader, model, output_layer, criterion, device, None)
                pbar_epochs.set_postfix({"Val Loss":val_loss, "Val F1":val_f1, "ROC":roc})
        rocs.append(roc)
        
        # # Save model if performance improves
        # if val_f1 > best_f1:
        #     best_f1 = val_f1
        #     save_path = os.path.join(args.model_path, f"best_model.pth")
        #     torch.save(model.state_dict(), save_path)
        #     print(f"Best model saved at epoch {epoch} with F1: {best_f1:.4f}")
    print(f"Mean ROC: {np.mean(rocs)} Dev ROC: {np.std(rocs)}")
    wandb.log({"Mean ROC":np.mean(rocs),
               "Dev ROC":np.std(rocs)})

if __name__ == "__main__":
    main()
