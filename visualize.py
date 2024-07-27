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
from matplotlib.colors import ListedColormap, LogNorm
# plt.rcParams.update({
#         'font.size': 32,         # 기본 글자 크기
#         'axes.titlesize': 40,    # 제목 글자 크기
#         'axes.labelsize': 36,    # 축 라벨 글자 크기
#         'xtick.labelsize': 28,   # x축 틱 라벨 글자 크기
#         'ytick.labelsize': 28,   # y축 틱 라벨 글자 크기
#         'legend.fontsize': 32,   # 범례 글자 크기
#         'figure.titlesize': 44   # Figure 제목 글자 크기
#         })
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ratio_over_threshold = []
below_threshold_pair = []
low_threshold = 0
high_threshold = 0.5
# attn_path =  f'./model/cora/attention_weight_0_layer1_head0.csv'
# df = pd.read_csv(attn_path ,header=None)
# attn_weight = df.to_numpy()
# attn_path2 =  f'./model/cora/attention_weight_0_layer1_head1.csv'
# df = pd.read_csv(attn_path2 ,header=None)
# attn_weight2 = df.to_numpy()
# attn_tensor = torch.from_numpy(attn_weight).to(device)
# norms = attn_tensor.norm(p=2, dim=1, keepdim=True)
# normalized_attn_tensor = attn_tensor / norms
# attn_tensor2 = torch.from_numpy(attn_weight2).to(device)
# norms2 = attn_tensor2.norm(p=2, dim=1, keepdim=True)
# normalized_attn_tensor2 = attn_tensor2 / norms2
# transposed=torch.transpose(normalized_attn_tensor2, 0, 1)
# similarity = torch.matmul(normalized_attn_tensor, transposed)
# similarity=similarity.cpu()
# plt.figure(figsize=(50, 40))
# sns.heatmap(similarity, annot=False, fmt=".2f", cmap='viridis')
# plt.title("Attention Weights Similarity Heatmap")
# plt.xlabel(f'Head 0')
# plt.ylabel(f'Head 1')
# plt.show()
# plt.savefig(f'./model/cora/layer_1_head0,1_similarity.png')
viridis = plt.cm.viridis
viridis_colors = viridis(np.arange(viridis.N))

# 검정색 추가
# viridis_colors[0] = np.array([0, 0, 0, 1])  # 검정색 (RGBA)

# 새로운 컬러맵 생성
new_cmap = ListedColormap(viridis_colors)
for i in range(1,3):
    for j in range(4):
        attn_path =  f'./model/cora/attention_weight_0_layer{i}_head{j}_end.csv'
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
        # print(attn_tensor.sum(dim=1))
        # count_between_thresholds = ((attn_tensor > low_threshold) & (attn_tensor < high_threshold)).sum(dim=1)

        # print(count_between_thresholds)
        # print(attn_tensor.max())
        # print((count_between_thresholds == 1).sum())
        # print((count_between_thresholds == 0).sum())
        # norms = attn_tensor.norm(p=2, dim=1, keepdim=True)
        # normalized_attn_tensor = attn_tensor / norms
        # transposed=torch.transpose(normalized_attn_tensor, 0, 1)
        # similarity = torch.matmul(normalized_attn_tensor, transposed)
        # count_above_threshold = (attn_tensor > threshold).sum().item()
        # indices = torch.nonzero(attn_tensor > threshold, as_tuple=True)
        # new_tensor = torch.zeros_like(attn_tensor)

        # # 추출한 index들에 대해 값을 1로 설정
        # new_tensor[indices] = 1
        # new_tensor=new_tensor.cpu()
        # count_list = [(i, row.sum().item()) for i, row in enumerate(new_tensor)]

        # # count를 기준으로 리스트를 정렬
        # sorted_count_list = sorted(count_list, key=lambda x: x[1], reverse=True)
        # print(count_above_threshold)
        # print(sorted_count_list)
        # # 전체 텐서의 요소 수
        # total_elements = similarity.numel()
        # #비율 계산
        # ratio = count_above_threshold / total_elements
        # similarity=similarity.cpu()
        # ratio_over_threshold.append(ratio)
        # indices, counts = zip(*sorted_count_list)
        # plt.bar(indices, counts)
        # plt.xlabel('Node')
        # plt.ylabel('Count')
        # plt.title('Count of below threshold of node')
        # plt.savefig(f'./model/cora/layer_{i}_head{j}_similarity_lowpasscount.png')
        # # plt.figure(figsize=(50, 40))
        # sns.heatmap(new_tensor, annot=False, fmt=".2f", cmap='viridis')
        # plt.title("Attention Weights Similarity Heatmap")
        # plt.xlabel(f'Head {j}')
        # plt.ylabel(f'Head {j}')
        # plt.show()
        # plt.savefig(f'./model/cora/layer_{i}_head{j}_similarity_lowpass.png')
        # plt.figure(figsize=(50, 40))
        # sns.heatmap(similarity, annot=False, fmt=".2f", cmap='viridis')
        # plt.title("Attention Weights Similarity Heatmap")
        # plt.xlabel(f'Head {j}')
        # plt.ylabel(f'Head {j}')
        # plt.show()
        # plt.savefig(f'./model/cora/layer_{i}_head{j}_similarity.png')
        # print(f'Save figure!')

        plt.figure(figsize=(50, 40))
        sns.heatmap(attn_weight, annot=False, fmt=".2f", cmap=new_cmap)
        plt.title("Attention Weights Heatmap")
        plt.xlabel(f'Head {j}')
        plt.ylabel(f'Head {j}')
        plt.show()
        plt.savefig(f'./model/cora/layer_{i}_head{j}_end.png')
        print(f'Save figure!')

# print(f"Similarity ratio above {threshold}: {ratio_over_threshold}")
# print(below_threshold_pair)