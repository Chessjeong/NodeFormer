import argparse
import sys
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from sklearn.neighbors import kneighbors_graph
import networkx as nx


from logger import Logger
from dataset import load_dataset
from data_utils import load_fixed_splits, adj_mul, get_gpu_memory_map
from eval import evaluate, eval_acc, eval_rocauc, eval_f1
from parse import parse_method, parser_add_main_args
import time

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def edge_index_to_adj_matrix(edge_index, num_nodes, device):
    adj_matrix = torch.zeros((num_nodes, num_nodes), device=device)
    for edge in edge_index:
        adj_matrix[edge[0], edge[1]] = 1
    for i in range(num_nodes):
        adj_matrix[i, i] = 1
    return adj_matrix


def generate_connected_adjacency_matrix(n, device='cuda:0'):
    # Create an empty adjacency matrix on the specified device (default is 'cuda')
    adj_matrix = torch.zeros((n, n), dtype=torch.float32, device=device)
    

    # Ensure the graph is connected by creating a spanning tree
    for i in range(1, n):
        j = torch.randint(0, i, (1,), device=device).item()  # Randomly connect to any previous node
        adj_matrix[i, j] = adj_matrix[j, i] = 1
    for i in range(n):
        adj_matrix[i, i] = 1

    # Randomly add some more edges to make it more random while keeping it connected
    num_edges = torch.randint(n, n*(n-1)//2, (1,), device=device).item()
    while torch.sum(adj_matrix) < 2 * num_edges:  # each edge is counted twice in the adjacency matrix
        i, j = torch.randint(0, n, (2,), device=device)
        if i != j:
            adj_matrix[i, j] = adj_matrix[j, i] = 1

    return adj_matrix


def calculate_attention_k(edge_index, attn_weight, num_nodes, max_k=9):
    adj_matrix = edge_index_to_adj_matrix(edge_index, num_nodes, device)
    # adj_matrix = torch.tensor(nx.adjacency_matrix(random_graph).todense(), device = device).float()
    attn_weight = torch.tensor(attn_weight, device=device)
    
    k_hop_neighbors = torch.eye(num_nodes, device=device)
    attn_sum = torch.zeros(max_k, device=device)
    attn_k_list = []
    max_degree_node = (adj_matrix > 0).sum(dim=1).max()
    # print(max_degree_node)

    for k in range(max_k):
        k_hop_neighbor_count = (k_hop_neighbors > 0).sum(dim=1)
        k_hop_neighbor_attn_avg = (k_hop_neighbors * attn_weight).sum(dim=1)/k_hop_neighbor_count


        k_hop_neighbor_attn_sum = k_hop_neighbor_attn_avg.mean()
        # k_hop_neighbor_attn_sum = k_hop_neighbor_attn_avg[max_degree_node]
        
        if k == 0:
            attn_k = k_hop_neighbor_attn_sum
        else:
            attn_k = k_hop_neighbor_attn_sum - attn_sum[k-1]
        
        attn_sum[k] = k_hop_neighbor_attn_sum
        attn_k_list.append(attn_k.item())
        
        k_hop_neighbors = torch.matmul(k_hop_neighbors, adj_matrix)
        k_hop_neighbors =  (k_hop_neighbors > 0).float()
    return attn_k_list

def attn_sum():
    return


### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

# os.environ["CUDA_VISIBLE_DEVICES"]= "1" 
# print(torch.__version__)
print(f"Using device: {device}")
print(f"Available: {torch.cuda.device_count()}")
### Load and preprocess data ###
dataset = load_dataset(args.data_dir, args.dataset, args.sub_dataset)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
dataset.label = dataset.label.to(device)

# get the splits for all runs
if args.rand_split:
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                     for _ in range(args.runs)]
elif args.rand_split_class:
    split_idx_lst = [dataset.get_idx_split(split_type='class', label_num_per_class=args.label_num_per_class)
                     for _ in range(args.runs)]
elif args.dataset in ['ogbn-proteins', 'ogbn-arxiv', 'ogbn-products', 'amazon2m']:
    split_idx_lst = [dataset.load_fixed_splits()
                     for _ in range(args.runs)]
else:
    split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset, protocol=args.protocol)

#
if args.dataset in ('mini', '20news'):
    adj_knn = kneighbors_graph(dataset.graph['node_feat'], n_neighbors=args.knn_num, include_self=True)
    edge_index = torch.tensor(adj_knn.nonzero(), dtype=torch.long)
    dataset.graph['edge_index']=edge_index

### Basic information of datasets ###
n = dataset.graph['num_nodes']
e = dataset.graph['edge_index'].shape[1]
# infer the number of classes for non one-hot and one-hot labels
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")

# whether or not to symmetrize
if not args.directed and args.dataset != 'ogbn-proteins':
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

dataset.graph['edge_index'], dataset.graph['node_feat'] = \
    dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)

### Load method ###
model = parse_method(args, dataset, n, c, d, device)

### Loss function (Single-class, Multi-class) ###
if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.NLLLoss()

### Performance metric (Acc, AUC, F1) ###
if args.metric == 'rocauc':
    eval_func = eval_rocauc
elif args.metric == 'f1':
    eval_func = eval_f1
else:
    eval_func = eval_acc

logger = Logger(args.runs, args)
### Adj storage for relational bias ###
adjs = []
adj, _ = remove_self_loops(dataset.graph['edge_index'])
adj, _ = add_self_loops(adj, num_nodes=n)
adjs.append(adj)
for i in range(args.rb_order - 1): # edge_index of high order adjacency
    adj = adj_mul(adj, adj, n)
    adjs.append(adj)
dataset.graph['adjs'] = adjs
model_path = '/home/minho/NodeFormer/cora-nodeformer.pkl'
model.eval()
# out, link_loss_ = model(dataset.graph['node_feat'], dataset.graph['adjs'], args.tau)
# random_adj = generate_connected_adjacency_matrix(n, device)
random_graph = nx.erdos_renyi_graph(n, p=0.01)
attn_mean = torch.zeros((args.num_layers * args.num_heads, n))
for i in range(1, args.num_layers+1):
    
    for j in range(args.num_heads):
        attn_path =  f'./model/cora/attention_weight_0_layer{i}_head{j}.csv'
        df = pd.read_csv(attn_path ,header=None)
        attn_weight = df.to_numpy()
        # print(attn_weight_.shape)
        # for attn_weight in attn_weight_:
        # attn_k_list = calculate_attention_k(dataset.graph['adjs'], attn_weight,dataset.graph['node_feat'].size(0))
        # print(f'L{i}_h{j}= {attn_k_list} ')
        attn_tensor = torch.tensor(attn_weight, device = device).mean(dim=0)
        attn_mean[i-1+j, :] = attn_tensor

print(attn_mean.size())
print(attn_mean)
norms = attn_mean.norm(p=2, dim=1, keepdim=True)
normalized_attn_mean = attn_mean / norms
transposed=torch.transpose(normalized_attn_mean, 0, 1)
similarity = torch.matmul(normalized_attn_mean, transposed)
print(similarity)