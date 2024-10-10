#!/usr/bin/env python
# encoding: utf-8

import argparse
import os
import time
import numpy as np
import dgl
import torch
import torch.nn as nn
from ogb.graphproppred import DglGraphPropPredDataset, Evaluator
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from gcc.models import GraphEncoder, OGBGraphEncoder
from gcc.utils.misc import AverageMeter
import random
import warnings
import wandb
from datetime import datetime
from tqdm import tqdm
warnings.filterwarnings("ignore")


def summarize_model(model):
    """
    Prints the layers of the model, total trainable parameters, and model size in MB.
    
    Args:
    model (torch.nn.Module): The PyTorch model to summarize.
    """
    
    # List model layers
    print("Model Layers:")
    for name, layer in model.named_children():
        print(f"{name}: {layer.__class__.__name__}")

    # Calculate the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Model size (in MB)
    model_size = total_params * 4 / (1024 ** 2)  # Assuming 32-bit (4 bytes per float)

    print(f"\nModel Size: {model_size:.2f} MB")
    print(f"Trainable Parameters: {trainable_params}")

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

    kwargs = {'name': dataset + "-" + model, 'project': f'gcl-validation-reiteration', 'config': cfg,
              'settings': wandb.Settings(_disable_stats=False), 'reinit': True, 'entity':'hierarchical-diffusion',
              'mode':'online' if offline else 'online'}
    wandb.init(**kwargs)
    wandb.save('*.txt')

    return cfg

def parse_option():
    parser = argparse.ArgumentParser("Supervised graph classification finetuning with OGB support")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--dataset", type=str, default="ogbg-molhiv", help="OGB dataset for finetuning")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--load-path", type=str, required=True, help="Path to pre-trained model")
    parser.add_argument("--model-path", type=str, default="./checkpoints", help="Path to save fine-tuned model")
    return parser.parse_args()

def add_seed_feature(graphs):
    """
    Adds a 'seed' feature to each graph. The seed node is randomly chosen, and its corresponding
    node feature is set to 1.0, while all other nodes' 'seed' features are set to 0.0.
    """
    for g in graphs:
        num_nodes = g.number_of_nodes()
        seed_node = random.randint(0, num_nodes - 1)  # Choose a random node as the seed
        seed_feature = torch.zeros(num_nodes, dtype=torch.float32)
        seed_feature[seed_node] = 1.0  # Mark the seed node
        g.ndata['seed'] = seed_feature
    return graphs

def load_ogb_dataset_with_seed(dataset_name):
    dataset = DglGraphPropPredDataset(name=dataset_name)
    evaluator = Evaluator(name=dataset_name)

    # Add 'seed' feature to the graphs
    dataset.graphs = add_seed_feature(dataset.graphs)
    
    return dataset, evaluator

def train_finetune(epoch, train_loader, model, output_layer, criterion, optimizer, device):
    model.train()
    output_layer.train()
    loss_meter = AverageMeter()
    f1_meter = AverageMeter()
    label_n, label_sum = 0, 0
    for idx, (graphs, labels) in enumerate(train_loader):
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
        for graphs, labels in valid_loader:
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
                print(labels.cpu().numpy())
                pass
            label_n += labels.cpu().numpy().shape[0]
            label_sum += torch.sum(labels)
    wandb.log({"Val Loss":loss_meter.avg,
              "Accuracy":f1_meter.avg,
              "ROC":roc_meter.avg,
              "Label Weighting": label_sum/label_n})
    # print(f"Validation Loss: {loss_meter.avg:.4f} Accuracy: {f1_meter.avg:.4f} ROC: {roc_meter.avg:.4f}")
    return loss_meter.avg, f1_meter.avg, roc_meter.avg

def load_gnn_layers_only(model, pretrained_path):
    # Load the pretrained state_dict from the checkpoint
    pretrained_dict = torch.load(pretrained_path, map_location="cpu")['model']

    # Get the current model's state_dict
    model_dict = model.state_dict()

    # Filter out only the GNN layer weights from the pre-trained model
    pretrained_gnn_dict = {k: v for k, v in pretrained_dict.items() if 'gnn' in k}

    # Ensure only the matching GNN layers are loaded
    pretrained_gnn_dict = {k: v for k, v in pretrained_gnn_dict.items() if k in model_dict and v.shape == model_dict[k].shape}

    # Update the model's state_dict with the pretrained GNN layers
    model_dict.update(pretrained_gnn_dict)

    # Load the updated state_dict into the model
    model.load_state_dict(model_dict)

    # print(f"Loaded GNN layers from {pretrained_path} into the current model.")

def get_model(node_input_dim, edge_input_dim, num_tasks, device, args):

    # Example usage:
    model = OGBGraphEncoder(
        node_input_dim=node_input_dim,  # Based on OGB node features
        edge_input_dim=edge_input_dim,  # Based on OGB edge features
        output_dim=64,
        node_hidden_dim=64,
        edge_hidden_dim=128,
        num_layers=5,       # Set based on your new architecture
        norm=True,
        gnn_model="gin"     # Match your architecture
    )

    # model.load_state_dict(torch.load(args.load_path, map_location=device)["model"])
    load_gnn_layers_only(model, args.load_path)

    output_layer = nn.Linear(in_features=64, out_features=num_tasks).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(output_layer.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    return model, criterion, optimizer, output_layer

def main():
    
    args = parse_option()
    # setup_wandb(args)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print(f"Loading OGB dataset {args.dataset} for fine-tuning")
    dataset, evaluator = load_ogb_dataset_with_seed(args.dataset)
    
    # Load the default train and test splits
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']
    test_idx = split_idx['test']

    graph = dataset[0][0]  # First graph in the dataset
    node_input_dim = graph.ndata['feat'].shape[1]  # Get the node feature dimension
    edge_input_dim = graph.edata['feat'].shape[1] if 'feat' in graph.edata else 0  # Get edge feature dimension
    
    train_loader = dgl.dataloading.GraphDataLoader(
        dataset[train_idx],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    
    valid_loader = dgl.dataloading.GraphDataLoader(
        dataset[test_idx],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle = True,
    )

    # print(f"Loading pre-trained model from {args.load_path}")
    # model = OGBGraphEncoder(
    #     positional_embedding_size=32,
    #     max_node_freq=16,
    #     max_edge_freq=16,
    #     max_degree=512,
    #     freq_embedding_size=16,
    #     degree_embedding_size=16,
    #     output_dim=64,
    #     node_hidden_dim=64,
    #     edge_hidden_dim=64,
    #     num_layers=5,
    #     norm=True,
    #     gnn_model="gin",
    #     degree_input=True,
    # ).to(device)



    os.makedirs(args.model_path, exist_ok=True)

    best_f1 = 0
    rocs = []
    for i in tqdm(range(10), colour='red'):
        model, criterion, optimizer, output_layer = get_model(node_input_dim, edge_input_dim, 1, device, args)
        summarize_model(model)
        quit()
        model = model.to(device)
        output_layer = output_layer.to(device)
        pbar_epochs = tqdm(range(0, args.epochs), colour='green', leave = False)
        for epoch in pbar_epochs:
            train_finetune(epoch, train_loader, model, output_layer, criterion, optimizer, device)
            if epoch % 10 == 0 or epoch == 0:
                val_loss, val_f1, roc = test_finetune(valid_loader, model, output_layer, criterion, device, evaluator)
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
