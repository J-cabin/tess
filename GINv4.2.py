import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold, KFold
from dgl import sum_nodes, broadcast_nodes
from dgl.data import GINDataset
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch.conv import GINConv, APPNPConv, GATConv, GraphConv, SAGEConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
import dgl.nn as dglnn
import argparse
from collections import Counter
import time
import copy
from datetime import datetime
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score, accuracy_score
from bisect import bisect_left
from sat2gv4 import SATDataset
import os
from torch.nn import LSTM
import random


class Data:
    def __init__(self, X, y):
        self.X = X
        self.y = y


from dgl.nn.pytorch import (
    AvgPooling,
    GlobalAttentionPooling,
    MaxPooling,
    Set2Set,
    SumPooling,
)
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
# from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


### GIN convolution along the graph structure
class GINConv(nn.Module):
    def __init__(self, emb_dim):
        """
        emb_dim (int): node embedding dimensionality
        """

        super(GINConv, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.eps = nn.Parameter(torch.Tensor([0]))

        # self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, g, x):
        with g.local_scope():
            # edge_embedding = self.bond_encoder(edge_attr)
            g.ndata["x"] = x
            g.apply_edges(fn.copy_u("x", "m"))
            g.edata["m"] = F.relu(g.edata["m"])
            g.update_all(fn.copy_e("m", "m"), fn.sum("m", "new_x"))
            out = self.mlp((1 + self.eps) * x + g.ndata["new_x"])

            return out


### GCN convolution along the graph structure
class GCNConv(nn.Module):
    def __init__(self, emb_dim):
        """
        emb_dim (int): node embedding dimensionality
        """

        super(GCNConv, self).__init__()

        self.linear = nn.Linear(emb_dim, emb_dim)
        self.root_emb = nn.Embedding(1, emb_dim)
        # self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, g, x):
        with g.local_scope():
            x = self.linear(x)
            # edge_embedding = self.bond_encoder(edge_attr)

            # Molecular graphs are undirected
            # g.out_degrees() is the same as g.in_degrees()
            degs = (g.out_degrees().float() + 1).to(x.device)
            norm = torch.pow(degs, -0.5).unsqueeze(-1)  # (N, 1)
            g.ndata["norm"] = norm
            g.apply_edges(fn.u_mul_v("norm", "norm", "norm"))

            g.ndata["x"] = x
            g.apply_edges(fn.copy_u("x", "m"))
            g.edata["m"] = g.edata["norm"] * F.relu(g.edata["m"])
            g.update_all(fn.copy_e("m", "m"), fn.sum("m", "new_x"))
            out = g.ndata["new_x"] + F.relu(
                x + self.root_emb.weight
            ) * 1.0 / degs.view(-1, 1)

            return out


class GeniePathConv(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_heads=1, residual=False):
        super(GeniePathConv, self).__init__()
        self.breadth_func = GATConv(
            in_dim, hid_dim, num_heads=num_heads, residual=residual, allow_zero_in_degree=True
        )
        self.depth_func = LSTM(hid_dim, out_dim)

    def forward(self, graph, x, h, c):
        x = self.breadth_func(graph, x)
        x = torch.tanh(x)
        x = torch.mean(x, dim=1)
        x, (h, c) = self.depth_func(x.unsqueeze(0), (h, c))
        x = x[0]
        return x, (h, c)



### GNN to generate node embedding
class GNN_node(nn.Module):
    """
    Output:
        node representations
    """

    def __init__(
        self,
        num_layers,
        emb_dim,
        drop_ratio=0.5,
        JK="last",
        residual=False,
        gnn_type="gin",
    ):
        """
        num_layers (int): number of GNN message passing layers
        emb_dim (int): node embedding dimensionality
        """

        super(GNN_node, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # self.atom_encoder = AtomEncoder(emb_dim)

        ###List of GNNs
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for layer in range(num_layers):
            if gnn_type == "gin":
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == "gcn":
                self.convs.append(GCNConv(emb_dim))
            elif gnn_type == 'geniepath':
                self.convs.append(GeniePathConv(
                    emb_dim,
                    emb_dim,
                    emb_dim,
                ))
            else:
                ValueError("Undefined GNN type called {}".format(gnn_type))

            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

    def forward(self, g, x):
        ### computing input node embedding
        h_list = [x]
        if self.gnn_type == 'geniepath':
            h = torch.zeros(1, x.shape[0], self.hid_dim).to(x.device)
            c = torch.zeros(1, x.shape[0], self.hid_dim).to(x.device)
        for layer in range(self.num_layers):
            if self.gnn_type == 'geniepath':
                x, (h, c) = self.convs[layer](g, h_list[layer], h, c)
            else:
                x = self.convs[layer](g, h_list[layer])
            x = self.batch_norms[layer](x)

            if layer == self.num_layers - 1:
                # remove relu for the last layer
                x = F.dropout(x, self.drop_ratio, training=self.training)
            else:
                x = F.dropout(
                    F.relu(x), self.drop_ratio, training=self.training
                )

            if self.residual:
                x += h_list[layer]

            h_list.append(x)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layers):
                node_representation += h_list[layer]

        return node_representation


class GNN_node_Virtualnode(nn.Module):
    """
    Output:
        node representations
    """

    def __init__(
        self,
        num_layers,
        input_dim,
        emb_dim,
        drop_ratio=0.5,
        JK="sum",
        residual=False,
        gnn_type="gin",
    ):
        """
        num_layers (int): number of GNN message passing layers
        emb_dim (int): node embedding dimensionality
        """

        super(GNN_node_Virtualnode, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual
        self.gnn_type = gnn_type
        self.hid_dim = emb_dim

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # self.atom_encoder = AtomEncoder(emb_dim)
        self.linear = nn.Linear(input_dim, emb_dim)

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = nn.Embedding(1, emb_dim)
        nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of GNNs
        self.convs = nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = nn.ModuleList()

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = nn.ModuleList()

        for layer in range(num_layers):
            if gnn_type == "gin":
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == "gcn":
                self.convs.append(GCNConv(emb_dim))
            elif gnn_type == 'geniepath':
                self.convs.append(GeniePathConv(
                    emb_dim,
                    emb_dim,
                    emb_dim,
                ))
            else:
                ValueError("Undefined GNN type called {}".format(gnn_type))

            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        for layer in range(num_layers - 1):
            self.mlp_virtualnode_list.append(
                nn.Sequential(
                    nn.Linear(emb_dim, emb_dim),
                    nn.BatchNorm1d(emb_dim),
                    nn.ReLU(),
                    nn.Linear(emb_dim, emb_dim),
                    nn.BatchNorm1d(emb_dim),
                    nn.ReLU(),
                )
            )
        self.pool = SumPooling()

    def forward(self, g, x):
        ### virtual node embeddings for graphs
        # print(x.dtype)
        virtualnode_embedding = self.virtualnode_embedding(
            torch.zeros(g.batch_size).to(torch.int).to(x.device)
        )
        # virtualnode_embedding = self.virtualnode_embedding(
        #     torch.zeros(g.batch_size).to(x.device)
        # )
        # print(virtualnode_embedding.shape)
        # print(x.shape)
        x = self.linear(x)
        h_list = [x]
        if self.gnn_type == 'geniepath':
            h = torch.zeros(1, x.shape[0], self.hid_dim).to(x.device)
            c = torch.zeros(1, x.shape[0], self.hid_dim).to(x.device)
        batch_id = dgl.broadcast_nodes(
            g, torch.arange(g.batch_size).to(x.device)
        )
        for layer in range(self.num_layers):
            # print(h_list[layer])
            # print(virtualnode_embedding[batch_id])
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch_id]

            ### Message passing among graph nodes
            if self.gnn_type == 'geniepath':
                x, (h, c) = self.convs[layer](g, h_list[layer], h, c)
            else:
                x = self.convs[layer](g, h_list[layer])
            x = self.batch_norms[layer](x)
            if layer == self.num_layers - 1:
                # remove relu for the last layer
                x = F.dropout(x, self.drop_ratio, training=self.training)
            else:
                x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)

            if self.residual:
                x = x + h_list[layer]

            h_list.append(x)

            ### update the virtual nodes
            if layer < self.num_layers - 1:
                ### add message from graph nodes to virtual nodes
                # print(virtualnode_embedding.shape)
                # a = self.pool(g, h_list[layer])
                # print(a.shape)
                virtualnode_embedding_temp = (
                    self.pool(g, h_list[layer]) + virtualnode_embedding
                )
                # print('virtualnode_embedding', virtualnode_embedding.shape)
                # print('virtualnode_embedding_temp', virtualnode_embedding_temp.shape)
                ### transform virtual nodes using MLP
                virtualnode_embedding_temp = self.mlp_virtualnode_list[layer](
                    virtualnode_embedding_temp
                )

                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(
                        virtualnode_embedding_temp,
                        self.drop_ratio,
                        training=self.training,
                    )
                else:
                    virtualnode_embedding = F.dropout(
                        virtualnode_embedding_temp,
                        self.drop_ratio,
                        training=self.training,
                    )

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layers):
                node_representation += h_list[layer]

        return node_representation


class TESS(nn.Module):
    def __init__(
        self,
        num_tasks=1,
        num_layers=2,
        input_dim=3,
        emb_dim=300,
        gnn_type="geniepath",
        virtual_node=True,
        residual=False,
        drop_ratio=0,
        JK="sum",
        graph_pooling="sum",
    ):
        """
        num_tasks (int): number of labels to be predicted
        virtual_node (bool): whether to add virtual node or not
        """
        super(TESS, self).__init__()

        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(
                num_layers,
                input_dim,
                emb_dim,
                JK=JK,
                drop_ratio=drop_ratio,
                residual=residual,
                gnn_type=gnn_type,
            )
        else:
            self.gnn_node = GNN_node(
                num_layers,
                emb_dim,
                JK=JK,
                drop_ratio=drop_ratio,
                residual=residual,
                gnn_type=gnn_type,
            )

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = SumPooling()
        elif self.graph_pooling == "mean":
            self.pool = AvgPooling()
        elif self.graph_pooling == "max":
            self.pool = MaxPooling()
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttentionPooling(
                gate_nn=nn.Sequential(
                    nn.Linear(emb_dim, 2 * emb_dim),
                    nn.BatchNorm1d(2 * emb_dim),
                    nn.ReLU(),
                    nn.Linear(2 * emb_dim, 1),
                )
            )

        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, n_iters=2, n_layers=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = nn.Linear(2 * self.emb_dim, self.num_tasks)
            self.graph_class_linear = nn.Linear(2 * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = nn.Linear(self.emb_dim, self.num_tasks)
            self.graph_class_linear = nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, g, x):
        h_node = self.gnn_node(g, x)

        h_graph = self.pool(g, h_node)
        output_reg = self.graph_pred_linear(h_graph)
        output_class = self.graph_pred_linear(h_graph)
        output_reg = output_reg.squeeze(-1)
        output_class = output_class.squeeze(-1)

        if self.training:
            return output_reg, output_class
        else:
            return torch.clamp(output_reg, min=0, max=5000), output_class

    def combo_loss(self, pred_t, pred_c, true_t, thres=5000, alpha=0.01):
        # mse loss of uncensored data
        uncensored_idx = true_t < thres
        uncensored_pred_t = pred_t[uncensored_idx]
        uncensored_true_t = true_t[uncensored_idx]
        mse_loss = F.mse_loss(uncensored_pred_t, uncensored_true_t)

        # bce loss of all data
        true_c = torch.zeros_like(pred_c)
        true_c[true_t > thres] = 1
        bce_loss = F.binary_cross_entropy_with_logits(pred_c, true_c)

        # consist_loss
        # consist_mul = (pred_t - thres) * (1 - 2 * true_c)
        # consist_loss = torch.mean(torch.relu(consist_mul))

        return mse_loss + \
            alpha * bce_loss



class Logger(object):
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()        # 缓冲区的内容及时更新到log文件

    def flush(self):
        pass


def calculate_hardness(y):
    from bisect import bisect_left
    hardness = [bisect_left(segs, label) for label in y]
    return hardness


def split_dataset2(data):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    idx_list = []
    for idx in kf.split(np.zeros(len(data.y)), calculate_hardness(data.y)):
        idx_list.append(idx)
    train_idx, valid_idx = idx_list[0]
    return train_idx, valid_idx
    # train_data = Data(data.X[train_idx], data.y[train_idx])
    # valid_data = Data(data.X[valid_idx], data.y[valid_idx])
    # return train_data, valid_data

def evaluate(dataloader, device, model):
    def seg_acc(labels, predicts):
        seg_labels = np.array([bisect_left(segs, i) for i in labels])
        seg_predicts = np.array([bisect_left(segs, i) for i in predicts])
        return accuracy_score(seg_labels, seg_predicts)

    model.eval()
    total = 0

    # all_predict = np.array([])
    # all_ids = np.array([])
    all_benchmark_ids = np.array([])
    all_results = np.array([])
    all_labels = np.array([])

    for batched_graph, graph_features, labels, gt, benchmark_ids, censor_type in dataloader:
        batched_graph = batched_graph.to(device)
        # labels = labels.to(device)
        feat = batched_graph.ndata.pop('attr')
        # total += len(labels)
        pred_t, _ = model(batched_graph, feat)
        if len(all_results) == 0:
            all_results = pred_t.cpu().detach().numpy()
            all_labels = labels
        else:
            all_results = np.concatenate((all_results, pred_t.cpu().detach().numpy()), axis=0)
            all_labels = np.concatenate((all_labels, labels), axis=0)
        if args.label_scale == 'log':
            all_results = np.expm1(all_results)
    result_dict = {'mse': mean_squared_error(all_labels, all_results),
                   'rmse': np.sqrt(mean_squared_error(all_labels, all_results)),
                   'evs': explained_variance_score(all_labels, all_results),
                   'mae': mean_absolute_error(all_labels, all_results),
                   'r2': r2_score(all_labels, all_results),
                   'pc': pearsonr(all_labels, all_results)[0],
                   'sc': spearmanr(all_labels, all_results)[0],
                   'acc': seg_acc(all_labels, all_results)}
    result_dict['all_results'] = all_results
    result_dict['all_labels'] = all_labels
    result_dict['all_benchmark_ids'] = all_benchmark_ids

    return result_dict


def train(train_loader, val_loader, device, model):
    # loss function, optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    best_loss = 100000000
    best_mae = 10000000
    # best_results = 0
    best_epoch = 0
    all_start_time = time.time()

    # training loop
    for epoch in range(args.max_epoch):
        model.train()
        total_loss = 0
        start_time = time.time()
        for batch, (batched_graph, graph_features, labels, gt, benchmark_ids, censor_type) in enumerate(train_loader):
            batched_graph = batched_graph.to(device)
            gt = gt.to(device)
            # print(batched_graph)
            # print(labels)
            feat = batched_graph.ndata.pop('attr')
            # print('G:', batched_graph)
            # print('Feat: ', feat.shape)
            # print(feat)
            pred_t, pred_c = model(batched_graph, feat)
            # print(F.softmax(logits, dim=1))
            # _, predicted = torch.max(logits, 1)
            # print(logits.shape, predicted.shape, labels.shape)
            # assert (len(predicted) == len(labels))
            # print('logits: ', logits.shape)
            # print('gt: ', gt.shape)
            # loss = F.mse_loss(logits, gt)
            loss = model.combo_loss(pred_t, pred_c, gt, thres=args.thres, alpha=args.alpha)
            # print(logits.type(), labels.type(), loss)
            # if args.loss == 'CE':
            #     loss = nn.CrossEntropyLoss()(logits, labels)
            # elif args.loss == 'Time':
            #     loss = torch.sum(torch.mul(F.softmax(logits, dim=1), times))
            # elif args.loss == 'NormTime':
            #     times = F.normalize(times)  # 归一化
            #     loss = torch.sum(torch.mul(F.softmax(logits, dim=1), times))
            # else:
            #     raise ValueError(f'Undefined loss type')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()   # loss.detach.item()
        scheduler.step()
        train_results = evaluate(train_loader, device, model)
        # valid_results = evaluate(val_loader, device, model)
        # train_acc = train_results['acc']
        # valid_acc = valid_results['acc']
        end_time = time.time()
        print("Epoch {:03d} | Loss {:.4f} | Train: MSE/RMSE: {:.4f}/{:.4f}, MAE: {:.4f}, "
              "EVS: {:.4f}, R2: {:.4f}, Corr: {:.4f}/{:.4f}, Acc: {:.4f}, Time {:.4f} | {}".format(
            epoch, total_loss / (batch + 1),
            train_results["mse"], train_results["rmse"], train_results["mae"],
            train_results["evs"], train_results["r2"], train_results["pc"], train_results["sc"],
            train_results["acc"], end_time-start_time, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        # print("Valid: MSE/RMSE: {:.4f}/{:.4f}, MAE: {:.4f}, EVS: {:.4f}, R2: {:.4f}, Corr: {:.4f}/{:.4f}, Acc: {:.4f}".format(
        #     valid_results["mse"], valid_results["rmse"], valid_results["mae"],
        #     valid_results["evs"], valid_results["r2"], valid_results["pc"][0], valid_results["sc"][0], valid_results["acc"]))
        # print("Epoch {:03d} | Loss {:.4f} | Train r2: {:.4f} | Valid r2: {:.4f} | "
        #       "Train MSE. {:.4f} | Valid MSE. {:.4f} | "
        #       "Time {:.4f} | {}"
        #       .format(epoch, total_loss / (batch + 1), train_results['r2'], valid_results['r2'],
        #               train_results['mse'], valid_results['mse'],
        #               end_time-start_time, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        if train_results["mae"] < best_mae:
            best_loss = total_loss / (batch + 1)
            best_epoch = epoch
            # best_train_result = train_results
            # best_results = valid_results
            # best_sc = train_results["sc"]
            best_mae = train_results["mae"]
            state = dict([('model', copy.deepcopy(model.state_dict())),
                          ('optim', copy.deepcopy(optimizer.state_dict()))])
            # print(best_results['all_labels'])
            # print(best_results['all_results'])
            # visualize(best_results['all_labels'], best_results['all_results'])
        if epoch - best_epoch > args.patience:
            break
    print("Final results: Best Epoch {:03d}, Best Loss, All Time {:.4f}s".format(
        best_epoch, best_loss, time.time() - all_start_time))
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optim'])
    best_train_result = evaluate(train_loader, device, model)
    best_results = evaluate(val_loader, device, model)
    print("Train: MSE/RMSE: {:.4f}/{:.4f}, MAE: {:.4f}, EVS: {:.4f}, R2: {:.4f}, Corr: {:.4f}/{:.4f}, Acc: {:.4f}".format(
        best_train_result["mse"], best_train_result["rmse"], best_train_result["mae"], best_train_result["evs"],
        best_train_result["r2"], best_train_result["pc"], best_train_result["sc"], best_train_result["acc"]))
    print("Valid: MSE/RMSE: {:.4f}/{:.4f}, MAE: {:.4f}, EVS: {:.4f}, R2: {:.4f}, Corr: {:.4f}/{:.4f}, Acc: {:.4f}".format(
        best_results["mse"], best_results["rmse"], best_results["mae"], best_results["evs"], best_results["r2"],
        best_results["pc"], best_results["sc"], best_results["acc"]))
    # print("Final results: Best Epoch {:03d} | Loss {:.4f} | Valid r2: {:.4f} | Valid MSE. {:.4f} | All Time {:.4f}"
    #       .format(best_epoch, best_loss, best_results['r2'], best_results['mse'], time.time() - all_start_time))
    # train_predict = np.concatenate((best_train_result['all_benchmark_ids'].reshape(-1, 1),
    #                                 best_train_result['all_predict'].reshape(-1, 1)), axis=1)
    # valid_predict = np.concatenate((best_results['all_benchmark_ids'].reshape(-1, 1),
    #                                 best_results['all_predict'].reshape(-1, 1)), axis=1)
    # all_predict = np.concatenate((train_predict, valid_predict), axis=0)
    # all_predict = all_predict[all_predict[:, 0].argsort()]
    # save_path = Path.cwd().joinpath('predict', datetime.strftime(datetime.now(), "%Y-%m-%d"))
    # if not save_path.exists():
    #     save_path.mkdir()
    # predict_file = save_path.joinpath(args.model+'_'+args.pooling+'_'+args.loss+'_'+str(args.learning_rate)+'_'+
    #                                str(args.time_out)+'_'+"predict.txt")
    # np.savetxt(predict_file, all_predict, delimiter=',', fmt='%s')
    # visualize(best_results['all_labels'], best_results['all_results'])
    # np.savetxt(f'gt-{args.dataset}-{args.model}', gt, fmt='%.4f')
    # np.savetxt(f'pred-{args.dataset}-{args.model}', pred, fmt='%.4f')

    result_df = pd.DataFrame()
    result_df['model'] = [args.model]
    result_df['graph'] = [args.graph]
    result_df['label_scale'] = [args.label_scale]
    result_df['MSE'] = [best_results['mse']]
    result_df['RMSE'] = [best_results['rmse']]
    result_df['MAE'] = [best_results["mae"]]
    result_df['EVS'] = [best_results["evs"]]
    result_df['R2'] = [best_results["r2"]]
    result_df['Corr(pc)'] = [best_results['pc']]
    result_df['Corr(sc)'] = [best_results['sc']]
    result_df['ACC'] = [best_results["acc"]]
    if os.path.exists(f'./result/{args.dataset}.csv'):
        _df = pd.read_csv(f'./result/{args.dataset}.csv', sep='\t')
        result_df = pd.concat([_df, result_df])
        print(result_df)
    result_df.to_csv(f'./result/{args.dataset}.csv', index=False, header=True, sep='\t')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


if __name__ == '__main__':
    seed_everything(0)
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="aig_dataset",                       # aig_dataset, INDU
                        # choices=['MUTAG', 'PTC', 'NCI1', 'PROTEINS'],
                        help='aig_dataset, SAT2022')
    parser.add_argument('--batch_size', type=int, default=8)
    # parser.add_argument('--split_idx', type=int, default=0)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.01)                        # [0.001, 0.005, 0.01, 0.05]
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--gnn', type=str, default="geniepath", help='gcn, gin, geniepath')
    parser.add_argument('--model', type=str, default="TESS", help='')
    # parser.add_argument('--gnn', type=str, default="gcn, gin, geniepath", help='')
    parser.add_argument('--graph', type=str, default="vig", help='lcg, vcg, lig, wlig, vig, wvig')
    parser.add_argument('--JK', type=str, default="sum", help='last, sum')                  # last, sum
    parser.add_argument('--pooling', type=str, default="sum", help='sum, mean, max, attention, set2set')    # sum, mean, max, attention, set2set
    parser.add_argument('--thres', type=int, default=5000, help='censor threshold')
    parser.add_argument('--alpha', type=float, default=0.01, help='censor threshold')       # [0.005, 0.01, 0.02]
    # parser.add_argument('--beta', type=float, default=0.01, help='')
    parser.add_argument('--hid_size', type=int, default=16, help='')                        # [16, 32, 64]
    parser.add_argument('--out_size', type=int, default=1, help='')
    parser.add_argument('--num_layer', type=int, default=2, help='')                        # [1, 2, 3]
    parser.add_argument('--virtual', type=bool, default=True, help='True, False')                        # [1, 2, 3]
    # parser.add_argument('--loss', type=str, default="Time", help='CE, Time, NormTime')
    parser.add_argument('--label_scale', type=str, default="none", help='log, norm, none')
    # parser.add_argument('--num_solver', type=int, default=5)
    parser.add_argument('--time_out', type=int, default=5000)
    args = parser.parse_args()

    # log
    log_path = Path.cwd().joinpath('logs', datetime.strftime(datetime.now(), "%Y-%m-%d"))
    if not log_path.exists():
        log_path.mkdir()
    log_file = log_path.joinpath(args.model+'_'+args.graph+'_'+args.pooling+'_' +str(args.learning_rate) +
                                 '_' + str(args.time_out))
    # print(log_path)
    sys.stdout = Logger(log_file)

    # device
    # print(f'Training with DGL built-in GINConv module with a fixed epsilon = 0')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.device >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.device))
    else:
        device = torch.device("cpu")

    # load and split dataset
    dataset = SATDataset(args.dataset, args=args)
    if args.dataset == 'UUF250_1065_100':
        # segs = np.percentile(y, (20, 40, 60, 80), interpolation='midpoint')
        segs = [i for i in range(10, 200, 10)]
    elif args.dataset in ['Flat200-479', 'flat_all']:
        segs = [0.05, 0.10, 0.15, 0.20]
    else:
        segs = [i for i in range(100, 5000, 100)]

    train_idx, val_idx = split_dataset2(Data(dataset.gfeatures, dataset.labels))

    # create dataloader
    train_loader = GraphDataLoader(dataset, sampler=SubsetRandomSampler(train_idx),
                                   batch_size=args.batch_size, pin_memory=torch.cuda.is_available(), drop_last=True)
    val_loader = GraphDataLoader(dataset, sampler=SubsetRandomSampler(val_idx),
                                 batch_size=args.batch_size, pin_memory=torch.cuda.is_available(), drop_last=True)

    # create GNN model
    model = TESS(num_layers=args.num_layer, input_dim=dataset.dim_nfeats, emb_dim=args.hid_size,
                 num_tasks=args.out_size, graph_pooling=args.pooling, gnn_type=args.gnn, JK=args.JK, virtual_node=args.virtual).to(device)

    # model training/validating
    print('Training...')
    train(train_loader, val_loader, device, model)
