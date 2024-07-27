import argparse
import sys
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from sklearn.neighbors import kneighbors_graph

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
import itertools

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
    # adj_matrix = random_adj
    attn_weight = torch.tensor(attn_weight, device=device)
    
    k_hop_neighbors = torch.eye(num_nodes, device=device)
    attn_sum = torch.zeros(max_k, device=device)
    attn_k_list = []
    max_degree_node = (adj_matrix > 0).sum(dim=1).max()
    # print(max_degree_node)

    for k in range(max_k):
        k_hop_neighbor_count = (k_hop_neighbors > 0).sum(dim=1)
        k_hop_neighbor_attn_avg = (k_hop_neighbors * attn_weight).sum(dim=1)


        k_hop_neighbor_attn_sum = k_hop_neighbor_attn_avg.mean()
        
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

unique_labels = torch.unique(dataset.label)

# Dictionary to hold indices for each label
indices_by_label = {label.item(): [] for label in unique_labels}

# Populate the dictionary with indices
for idx, label in enumerate(dataset.label):
    indices_by_label[label.item()].append(idx)

# Optional: Convert lists to tensors (if needed)
for label in indices_by_label:
    indices_by_label[label] = torch.tensor(indices_by_label[label], dtype=torch.long)

# Now indices_by_label contains the indices for each label
ratio_over_threshold = []
below_threshold_pair = []
threshold = 0.9
for i in range(1,3):
    for j in range(4):
        attn_path =  f'./model/cora/attention_weight_0_layer{i}_head{j}.csv'
        df = pd.read_csv(attn_path ,header=None)
        attn_weight = df.to_numpy()
        plt.rcParams.update({
        'font.size': 32,         # 기본 글자 크기
        'axes.titlesize': 40,    # 제목 글자 크기
        'axes.labelsize': 36,    # 축 라벨 글자 크기
        'xtick.labelsize': 28,   # x축 틱 라벨 글자 크기
        'ytick.labelsize': 28,   # y축 틱 라벨 글자 크기
        'legend.fontsize': 32,   # 범례 글자 크기
        'figure.titlesize': 44   # Figure 제목 글자 크기
        })
        attn_tensor = torch.from_numpy(attn_weight).to(device)
        norms = attn_tensor.norm(p=2, dim=1, keepdim=True)
        normalized_attn_tensor = attn_tensor / norms
        transposed=torch.transpose(normalized_attn_tensor, 0, 1)
        similarity = torch.matmul(normalized_attn_tensor, transposed)
        count_above_threshold = (similarity < threshold).sum().item()
        indices = torch.nonzero(similarity < threshold, as_tuple=True)
        new_tensor = torch.zeros_like(similarity)

        # 추출한 index들에 대해 값을 1로 설정
        new_tensor[indices] = 1
        new_tensor=new_tensor.cpu()
        count_list = [(i, row.sum().item()) for i, row in enumerate(new_tensor)]

        # count를 기준으로 리스트를 정렬
        sorted_count_list = sorted(count_list, key=lambda x: x[1], reverse=True)
        # 전체 텐서의 요소 수
        total_elements = similarity.numel()
        #비율 계산
        ratio = count_above_threshold / total_elements
        similarity=similarity.cpu()
        for label in range(len(indices_by_label)):
            pair = list(itertools.product(indices_by_label[label], repeat=2))

            similarity_intra = torch.zeros((indices_by_label[label].size()[0], indices_by_label[label].size()[0]),device =device)
            # pair에 있는 인덱스에 해당하는 original_tensor의 값을 similarity_intra에 넣기
            for (k, (x, y)) in enumerate(pair):
                idx_x = (indices_by_label[label] == x).nonzero(as_tuple=True)[0].item()
                idx_y = (indices_by_label[label] == y).nonzero(as_tuple=True)[0].item()
                
                # original_tensor의 값을 similarity_intra의 해당 위치에 복사
                similarity_intra[idx_x, idx_y] = similarity[x, y]
            similarity_intra = similarity_intra.cpu()
            plt.figure(figsize=(50, 40))
            sns.heatmap(similarity_intra, annot=False, fmt=".2f", cmap='viridis')
            plt.title(f'Attention Weights Similarity Heatmap within lable{label}')
            plt.xlabel(f'Head {j}')
            plt.ylabel(f'Head {j}')
            plt.show()
            plt.savefig(f'./model/cora/layer_{i}_head{j}_similarity_label{label}.png')
            print('Save figure!')
            
