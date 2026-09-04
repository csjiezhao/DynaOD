import os
import numpy as np
import torch
import dgl
import random
from torch.utils.data import Dataset
from random import shuffle, choice
random.seed(42)

import sys
from models.WeDAN import utils
sys.modules['utils'] = utils

import pickle
# 加载标准化对象
with open("ckpts/train_scalers.pkl", "rb") as f:
    SCALERS = pickle.load(f)


def load_city_graph(city_dir, add_self_loop=False):
    """
    从单个城市目录读取 demos/pois/adj 构建 DGLGraph。
    - adj.npy: (N,N) 接壤关系（0/1 或权重）
    - demos.npy: (N,D_demo)
    - pois.npy: (N,D_poi)
    返回: (city_name, graph)
    """
    demos_path = os.path.join(city_dir, "demos.npy")
    pois_path  = os.path.join(city_dir, "pois.npy")
    adj_path   = os.path.join(city_dir, "adj.npy")

    demos = np.load(demos_path).astype(np.float32)   # (N, D_demo)
    pois  = np.load(pois_path).astype(np.float32)    # (N, D_poi)
    adj   = np.load(adj_path)
    np.fill_diagonal(adj, 0)
    n = adj.shape[0]

    src, dst = np.nonzero(adj)
    g = dgl.graph((src, dst), num_nodes=n)
    if add_self_loop:
        g = dgl.add_self_loop(g)

    x = np.concatenate([demos, pois], axis=1).astype(np.float32)  # (N, D_demo + D_poi)
    x = SCALERS['feat'].transform(x)
    g.ndata["x"] = torch.from_numpy(x)

    return g


class CityGraphDataset(Dataset):
    """
    自监督：一个样本 = 一个城市图（region graph）
    """
    def __init__(self, data_path, max_cities=500, shuffle_cities=True, add_self_loop=False):
        super().__init__()
        self.data_path = data_path
        self.add_self_loop = add_self_loop

        cities = os.listdir(data_path)[:max_cities]
        if shuffle_cities == 1:
            shuffle(cities)
        self.cities = cities

        # 缓存每个城市的节点数，避免 sampler 每次读 adj.npy
        self._sizes = []
        for city in self.cities:
            adj_path = os.path.join(self.data_path, city, "adj.npy")
            adj = np.load(adj_path, mmap_mode="r")
            self._sizes.append(int(adj.shape[0]))

    def __len__(self):
        return len(self.cities)

    def __getitem__(self, idx: int):
        city = self.cities[idx]
        city_dir = os.path.join(self.data_path, city)
        g = load_city_graph(city_dir, add_self_loop=self.add_self_loop)
        return city, g

    def get_size(self, idx: int) -> int:
        return self._sizes[idx]


def collate_graphs(samples):
    """
    samples: list[(city, graph)]
    返回：
      cities: list[str]
      batched_graph: DGLGraph (disjoint union)
    """
    cities, graphs = map(list, zip(*samples))
    bg = dgl.batch(graphs)
    return cities, bg


class MaxNodeBatchSampler:
    def __init__(self, dataset: CityGraphDataset, batch_size: int, max_nodes: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_nodes = max_nodes

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        while indices:
            batch = []
            total_nodes = 0

            # 尝试填满 batch；如果随机挑不到合适的，就退化为取一个最小的
            tried = 0
            while indices and len(batch) < self.batch_size:
                idx = choice(indices)
                n = self.dataset.get_size(idx)
                if total_nodes + n <= self.max_nodes:
                    batch.append(idx)
                    total_nodes += n
                    indices.remove(idx)
                else:
                    tried += 1
                    # 防止在“剩下的都太大”时死循环：试几次不行就强行取一个（让它单独成 batch）
                    if tried > 50:
                        # 取一个节点数最小的，尽量不超
                        idx_min = min(indices, key=lambda j: self.dataset.get_size(j))
                        batch = [idx_min]
                        indices.remove(idx_min)
                        break

            if batch:
                yield batch

    def __len__(self):
        # 简单估计：按 sizes 贪心累加（不随机），更稳定
        sizes = list(self.dataset._sizes)
        sizes.sort()
        count, i = 0, 0
        while i < len(sizes):
            total, k = 0, 0
            while i < len(sizes) and k < self.batch_size and total + sizes[i] <= self.max_nodes:
                total += sizes[i]
                i += 1
                k += 1
            count += 1
        return count



if __name__ == '__main__':
    graphs = CityGraphDataset(data_path="../data")
    print(graphs[0])