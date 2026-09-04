import os
import dgl
import pickle
import torch
import datetime
import holidays
import pandas as pd
import numpy as np
import random
from random import shuffle, choice
from tqdm import tqdm
import json

import sys
from models.WeDAN import utils
sys.modules['utils'] = utils
random.seed(42)

# 加载标准化对象
with open("ckpts/train_scalers.pkl", "rb") as f:
    SCALERS = pickle.load(f)


def persist_cities(run_dir, cities, split_ratio):
    os.makedirs(run_dir, exist_ok=True)
    space_split_point = int(len(cities) * split_ratio)
    payload = {
        "cities": cities,
        "split_ratio": split_ratio,
        "seen_cities": cities[:space_split_point],
        "unseen_cities": cities[space_split_point:],
    }
    path = os.path.join(run_dir, "cities_split.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def load_persisted_cities(run_dir):
    import json, os
    path = os.path.join(run_dir, "cities_split.json")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["cities"], payload["seen_cities"], payload["unseen_cities"]


def extract_date_features(date_str):
    date_obj = datetime.datetime.strptime(date_str, "%Y_%m_%d")
    day_of_week = date_obj.weekday()          # 0–6
    us_holidays = holidays.US(years=date_obj.year)
    is_holiday = 1 if date_obj.date() in us_holidays else 0

    # 7 维 one-hot
    dow_onehot = np.zeros(7, dtype=np.float32)
    dow_onehot[day_of_week] = 1.0

    # 拼接成 8 维
    return np.concatenate([dow_onehot, [is_holiday]])   # shape (8,)


def load_city_data(data_path, city, date, llm):
    """
    加载并标准化指定城市和日期的数据
    """
    demos = np.load(os.path.join(data_path, city, "demos.npy")).astype(np.float32)
    pois = np.load(os.path.join(data_path, city, "pois.npy")).astype(np.float32)
    dis = np.load(os.path.join(data_path, city, "dis.npy")).astype(np.float32)
    od = np.load(os.path.join(data_path, city, f'ods/{date}_od.npy')).astype(np.float32)

    t_context = extract_date_features(date)
    n_nodes = od.shape[0]
    t_context = np.tile(t_context, (n_nodes, 1)).astype(np.float32)

    poi_ctrl_vec = np.load(os.path.join(data_path, city, "poi_vecs",  f"{llm}_poi_vec_{date}.npy")).astype(np.float32)
    demo_ctrl_vec = np.load(os.path.join(data_path, city, "demo_vecs", f"{llm}_demo_vec_{date}.npy")).astype(np.float32)


    # 特征处理
    nfeat = np.concatenate((demos, pois), axis=1)
    np.fill_diagonal(od, 0)
    scaled_od = SCALERS['od_normer'].fit_transform(od)
    nfeat = SCALERS['feat'].transform(nfeat)
    dis = SCALERS['dis'].transform(dis.reshape(-1, 1)).reshape(dis.shape)
    scaled_od = SCALERS['od'].transform(scaled_od.reshape(-1, 1)).reshape(scaled_od.shape)

    # 创建图
    graph = dgl.graph(od.nonzero(), num_nodes=od.shape[0])
    graph.ndata["demo"] = torch.from_numpy(nfeat)
    graph.ndata["t_context"] = torch.from_numpy(t_context)
    graph.ndata["poi_vec"] = torch.from_numpy(poi_ctrl_vec)
    graph.ndata["demo_vec"] = torch.from_numpy(demo_ctrl_vec)

    poi_shp_path = os.path.join(data_path, city, "poi_shps",  f"poi_shp_{date}.npy")
    demo_shp_path = os.path.join(data_path, city, "demo_shps", f"demo_shp_{date}.npy")
    near_score_path = os.path.join(data_path, city, f"near_k_scores.npy")
    if os.path.exists(poi_shp_path):
        k_scores = np.load(near_score_path).astype(np.float32)
        poi_shp = np.load(poi_shp_path).astype(np.float32)
        demo_shp = np.load(demo_shp_path).astype(np.float32)

        poi_shp, demo_shp, _ = weighted_shape_average(poi_shp, demo_shp, k_scores)
        graph.ndata["poi_shp"] = torch.from_numpy(poi_shp)
        graph.ndata["demo_shp"] = torch.from_numpy(demo_shp)

    return city, graph, torch.from_numpy(dis), torch.from_numpy(scaled_od)

def load_city_data_no_shape(data_path, city, date, llm):
    """
    加载并标准化指定城市和日期的数据
    """
    demos = np.load(os.path.join(data_path, city, "demos.npy")).astype(np.float32)
    pois = np.load(os.path.join(data_path, city, "pois.npy")).astype(np.float32)
    dis = np.load(os.path.join(data_path, city, "dis.npy")).astype(np.float32)
    od = np.load(os.path.join(data_path, city, f'ods/{date}_od.npy')).astype(np.float32)

    t_context = extract_date_features(date)
    n_nodes = od.shape[0]
    t_context = np.tile(t_context, (n_nodes, 1)).astype(np.float32)

    poi_ctrl_vec = np.load(os.path.join(data_path, city, "poi_vecs",  f"{llm}_poi_vec_{date}.npy")).astype(np.float32)
    demo_ctrl_vec = np.load(os.path.join(data_path, city, "demo_vecs", f"{llm}_demo_vec_{date}.npy")).astype(np.float32)


    # 特征处理
    nfeat = np.concatenate((demos, pois), axis=1)
    np.fill_diagonal(od, 0)
    scaled_od = SCALERS['od_normer'].fit_transform(od)
    nfeat = SCALERS['feat'].transform(nfeat)
    dis = SCALERS['dis'].transform(dis.reshape(-1, 1)).reshape(dis.shape)
    scaled_od = SCALERS['od'].transform(scaled_od.reshape(-1, 1)).reshape(scaled_od.shape)

    # 创建图
    graph = dgl.graph(od.nonzero(), num_nodes=od.shape[0])
    graph.ndata["demo"] = torch.from_numpy(nfeat)
    graph.ndata["t_context"] = torch.from_numpy(t_context)
    graph.ndata["poi_vec"] = torch.from_numpy(poi_ctrl_vec)
    graph.ndata["demo_vec"] = torch.from_numpy(demo_ctrl_vec)

    return city, graph, torch.from_numpy(dis), torch.from_numpy(scaled_od)

def weighted_shape_average(
    poi_shp: np.ndarray,      # (N,K,poi_dim)
    demo_shp: np.ndarray,     # (N,K,demo_dim)
    k_scores: np.ndarray,     # (N,K)
    eps: float = 1e-8,):
    """
    Return:
      poi_agg  : (N, poi_dim)
      demo_agg : (N, demo_dim)
      weights  : (N, K)
    """
    # ---- 1) normalize scores -> weights ----
    weights = k_scores / (k_scores.sum(axis=1, keepdims=True) + eps)
    # (N,K)

    # ---- 2) weighted sum ----
    poi_agg = np.sum(poi_shp * weights[:, :, None], axis=1)
    demo_agg = np.sum(demo_shp * weights[:, :, None], axis=1)
    return poi_agg, demo_agg, weights


def load_samples(data_path, shuffle_cities, split_ratio, mode, is_simplify=False, llm='gpt-4o-mini'):
    """
    获取样本，支持不同模式：'train', 'test1', 'test2', 'test3'
    """
    # # 获取城市列表
    # cities = os.listdir(data_path)[:500]
    # if shuffle_cities == 1:
    #     shuffle(cities)
    #
    # # 切分城市
    # space_split_point = int(len(cities) * split_ratio)
    # seen_cities = cities[:space_split_point]
    # unseen_cities = cities[space_split_point:]
    cities, seen_cities, unseen_cities = load_persisted_cities("ckpts/")

    # 获取日期范围并切分
    dates = pd.date_range(start='2019-01-01', end='2019-01-31').strftime('%Y_%m_%d').tolist()
    time_split_point = int(len(dates) * split_ratio)
    seen_dates = dates[:time_split_point]
    unseen_dates = dates[time_split_point:]

    # 模式选择
    sample_settings = {
        'train': (seen_cities, seen_dates),
        'test1': (seen_cities, unseen_dates),
        'test2': (unseen_cities, seen_dates),
        'test3': (unseen_cities, unseen_dates)
    }

    # 获取当前模式对应的城市和日期
    sample_cities, sample_dates = sample_settings.get(mode, ([], []))

    if is_simplify:
        return sample_cities, sample_dates
    else:
        # 创建样本
        geoids, graphs, dises, ods = [], [], [], []
        print("*"*20, f"loading {mode} samples", "*"*20)
        for city in tqdm(sample_cities):
            for date in sample_dates:
                geoid, graph, dis, od = load_city_data(data_path, city, date, llm=llm)
                geoids.append(geoid)
                graphs.append(graph)
                dises.append(dis)
                ods.append(od)
        return geoids, graphs, dises, ods


def load_samples_ijcai(data_path, shuffle_cities, split_ratio, mode, is_simplify=False, llm='gpt-4o-mini'):
    """
    获取样本，支持不同模式：'train', 'test1', 'test2', 'test3'
    """
    # # 获取城市列表
    # cities = os.listdir(data_path)[:500]
    # if shuffle_cities == 1:
    #     shuffle(cities)
    #
    # # 切分城市
    # space_split_point = int(len(cities) * split_ratio)
    # seen_cities = cities[:space_split_point]
    # unseen_cities = cities[space_split_point:]
    cities, seen_cities, unseen_cities = load_persisted_cities("ckpts/")

    # 获取日期范围并切分
    dates = pd.date_range(start='2019-01-01', end='2019-04-30').strftime('%Y_%m_%d').tolist()
    time_split_point = int(len(dates) * split_ratio)
    seen_dates = dates[:time_split_point]
    unseen_dates = dates[time_split_point:]

    # 模式选择
    sample_settings = {
        'train': (seen_cities, seen_dates),
        'test1': (seen_cities, unseen_dates),
        'test2': (unseen_cities, seen_dates),
        'test3': (unseen_cities, unseen_dates)
    }

    # 获取当前模式对应的城市和日期
    sample_cities, sample_dates = sample_settings.get(mode, ([], []))

    if is_simplify:
        return sample_cities, sample_dates
    else:
        # 创建样本
        geoids, graphs, dises, ods = [], [], [], []
        print("*"*20, f"loading {mode} samples", "*"*20)
        for city in tqdm(sample_cities):
            for date in sample_dates:
                geoid, graph, dis, od = load_city_data(data_path, city, date, llm=llm)
                geoids.append(geoid)
                graphs.append(graph)
                dises.append(dis)
                ods.append(od)
        return geoids, graphs, dises, ods


class CityDataset(dgl.data.DGLDataset):
    """
    County-level Dataset:
    每个样本是一个县 (county)，包含图结构、OD矩阵、气象特征、控制信号等
    """
    def __init__(self, GEOIDs, graphs, dises, ods):
        self.GEOIDs = GEOIDs
        self.graphs = graphs
        self.dises = dises
        self.ods = ods
        super(CityDataset, self).__init__(name="CityOD_dataset")

    def process(self):
        pass

    def get_size(self, index):
        """
        用于 batch sampler 判断规模
        返回 tract 数量
        """
        return self.dises[index].shape[0]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        """
        返回单个 county 的数据
        """
        return (
            self.GEOIDs[index],         # str
            self.graphs[index],         # DGLGraph
            self.dises[index],          # Tensor (N_tr, N_tr)
            self.ods[index],            # Tensor (N_tr, N_tr)
        )


def collate_fn(samples):
    geoids, graphs, dises, ods = map(list, zip(*samples))
    # batched graph
    batched_graph = dgl.batch(graphs)
    # block-diagonal distance matrix
    batched_dis = torch.block_diag(*dises)
    # 非对角部分默认设为 2（如果 dis 里空白是 0）
    batched_dis[batched_dis == 0] = 2.0
    return geoids, batched_graph, batched_dis, ods


class MyBatchSampler:
    def __init__(self, dataset, batch_size, max_value):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_value = max_value

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        while indices:
            batch = []
            current_sum = 0
            for _ in range(len(indices)):
                index = choice(indices)
                item_size = self.dataset.get_size(index)
                if (current_sum + item_size <= self.max_value) and (len(batch) < self.batch_size):
                    batch.append(index)
                    current_sum += item_size
                    indices.remove(index)
                else:
                    break

            if len(batch) > 0:
                yield batch
        if batch:
            yield batch

    def __len__(self):
        indices = list(range(len(self.dataset)))
        count = 0
        while indices:
            batch = []
            current_sum = 0
            for _ in range(len(indices)):
                index = choice(indices)
                item_size = self.dataset.get_size(index)
                if (current_sum + item_size <= self.max_value) and (len(batch) < self.batch_size):
                    batch.append(index)
                    current_sum += item_size
                    indices.remove(index)
                else:
                    break

            if len(batch) > 0:
                count += 1
        return count


def split_into_windows(date_list, T=7):
    """
    把 date_list 按 T=7 分块；最后一块不够 T 则直接保留（可变长度）。
    e.g. 21 days -> 3 windows of 7
         10 days -> 1 window of 10
    """
    windows = []
    i = 0
    while i < len(date_list):
        windows.append(date_list[i:i+T])
        i += T
    return windows


def load_window_samples(data_path, shuffle_cities, split_ratio, mode, is_simplify=False, T=7, llm='gpt-4o-mini'):
    cities, seen_cities, unseen_cities = load_persisted_cities("ckpts/")

    # dates (31 days)
    dates = pd.date_range(start='2019-01-01', end='2019-01-31').strftime('%Y_%m_%d').tolist()
    time_split_point = int(len(dates) * split_ratio)  # 31*0.7=21
    seen_dates = dates[:time_split_point]  # 21 days
    unseen_dates = dates[time_split_point:][-7:]  # 7 days

    seen_windows = split_into_windows(seen_dates, T=T)  # 3 windows of len 7
    unseen_windows = split_into_windows(unseen_dates, T=T)

    sample_settings = {
        'train': (seen_cities, seen_windows),
        'test1': (seen_cities, unseen_windows),
        'test2': (unseen_cities, seen_windows),
        'test3': (unseen_cities, unseen_windows),
    }
    sample_cities, sample_windows = sample_settings.get(mode, ([], []))
    if is_simplify:
        return sample_cities, sample_windows

    geoids, graphs_seqs, dises, ods_seqs = [], [], [], []
    print("*" * 20, f"loading {mode} windowed samples", "*" * 20)

    for city in tqdm(sample_cities):
        for win_id, date_list in enumerate(sample_windows):
            day_graphs, day_ods = [], []
            dis0 = None

            for d in date_list:
                geoid, g, dis, od = load_city_data(data_path, city, d, llm=llm)
                day_graphs.append(g)
                day_ods.append(od)
                if dis0 is None:
                    dis0 = dis

            geoids.append(f"{city}_{win_id}")
            graphs_seqs.append(day_graphs)  # list length = len(date_list)
            dises.append(dis0)
            ods_seqs.append(day_ods)  # list length = len(date_list)

    return geoids, graphs_seqs, dises, ods_seqs


def load_window_samples_ijcai(data_path, shuffle_cities, split_ratio, mode, is_simplify=False, T=7, llm='gpt-4o-mini'):
    cities, seen_cities, unseen_cities = load_persisted_cities("ckpts/")

    dates = pd.date_range(start='2019-01-01', end='2019-04-30').strftime('%Y_%m_%d').tolist()
    time_split_point = int(len(dates) * split_ratio)  # 31*0.7=21
    seen_dates = dates[:time_split_point]
    unseen_dates = dates[time_split_point:][-7:]  # 7 days

    seen_windows = split_into_windows(seen_dates, T=T)  # 3 windows of len 7
    unseen_windows = split_into_windows(unseen_dates, T=T)

    sample_settings = {
        'train': (seen_cities, seen_windows),
        'test1': (seen_cities, unseen_windows),
        'test2': (unseen_cities, seen_windows),
        'test3': (unseen_cities, unseen_windows),
    }
    sample_cities, sample_windows = sample_settings.get(mode, ([], []))
    if is_simplify:
        return sample_cities, sample_windows

    geoids, graphs_seqs, dises, ods_seqs = [], [], [], []
    print("*" * 20, f"loading {mode} windowed samples", "*" * 20)

    for city in tqdm(sample_cities):
        for win_id, date_list in enumerate(sample_windows):
            day_graphs, day_ods = [], []
            dis0 = None

            for d in date_list:
                geoid, g, dis, od = load_city_data_no_shape(data_path, city, d, llm=llm)
                day_graphs.append(g)
                day_ods.append(od)
                if dis0 is None:
                    dis0 = dis

            geoids.append(f"{city}_{win_id}")
            graphs_seqs.append(day_graphs)  # list length = len(date_list)
            dises.append(dis0)
            ods_seqs.append(day_ods)  # list length = len(date_list)

    return geoids, graphs_seqs, dises, ods_seqs


class CityWindowDataset(dgl.data.DGLDataset):
    """
    Window-level Dataset:
    每个样本 = (city, 7-day window)
    - graphs_seq: List[DGLGraph] length T(=7)
    - ods_seq:    List[Tensor]   length T(=7)
    - dis:        Tensor (N,N)   (static)
    """
    def __init__(self, GEOIDs, graphs_seqs, dises, ods_seqs):
        self.GEOIDs = GEOIDs
        self.graphs_seqs = graphs_seqs
        self.dises = dises
        self.ods_seqs = ods_seqs
        super().__init__(name="CityOD_window_dataset")

    def process(self):
        pass

    def get_size(self, index):
        return self.dises[index].shape[0]

    def __len__(self):
        return len(self.graphs_seqs)

    def __getitem__(self, index):
        return (
            self.GEOIDs[index],          # str
            self.graphs_seqs[index],     # List[DGLGraph], len=T
            self.dises[index],           # Tensor (N,N)
            self.ods_seqs[index],        # List[Tensor], len=T, each (N,N)
        )


def collate_fn_window(samples):
    geoids, graphs_seqs, dises, ods_seqs = map(list, zip(*samples))

    # 统一窗口长度（你要求固定 7 天）
    T = len(graphs_seqs[0])
    assert all(len(gs) == T for gs in graphs_seqs), "Window length mismatch in batch."

    # 1) 每天一个 batched graph
    batched_graphs_by_day = []
    for t in range(T):
        graphs_t = [graphs_seqs[b][t] for b in range(len(graphs_seqs))]
        batched_graphs_by_day.append(dgl.batch(graphs_t))

    # 2) batched distance: block diag（按样本）
    batched_dis = torch.block_diag(*dises)
    batched_dis[batched_dis == 0] = 2.0

    # 3) ods 按天组织：ods_by_day[t] 是长度=B 的 list，每个是 (Ni,Ni)
    ods_by_day = [[] for _ in range(T)]
    for b in range(len(ods_seqs)):
        for t in range(T):
            ods_by_day[t].append(ods_seqs[b][t])

    return geoids, batched_graphs_by_day, batched_dis, ods_by_day




if __name__ == '__main__':
    from dgl.dataloading import GraphDataLoader
    import torch
    import dgl

    # -----------------------------
    # 1) build loader + fetch one batch
    # -----------------------------
    train_data = load_window_samples(
        data_path="data/",
        shuffle_cities=1,  # your loader uses 1/0
        split_ratio=0.7,
        mode="train",
        T=7
    )
    train_set = CityWindowDataset(*train_data)
    sampler = MyBatchSampler(train_set, batch_size=8, max_value=1500)
    dataloader = GraphDataLoader(train_set, batch_sampler=sampler, collate_fn=collate_fn_window)

    batch = next(iter(dataloader))
    geoids, bgs_by_day, batched_dis, ods_by_day = batch

    print("=== Batch Summary ===")
    print("num samples in batch:", len(geoids))
    print("T (days):", len(bgs_by_day))
    print("batched_dis:", tuple(batched_dis.shape))
    print("ods_by_day lens:", [len(x) for x in ods_by_day])


    # -----------------------------
    # 2) helpers
    # -----------------------------
    def edge_keys(bg: dgl.DGLGraph) -> torch.Tensor:
        """
        Return sorted edge keys for exact comparison.
        key = src*(N+1) + dst
        """
        src, dst = bg.edges()
        N = bg.num_nodes()
        key = src.to(torch.int64) * (N + 1) + dst.to(torch.int64)
        key, _ = torch.sort(key)
        return key


    def edge_jaccard(bg1: dgl.DGLGraph, bg2: dgl.DGLGraph, sample_k: int = 200000) -> float:
        """
        Approx Jaccard similarity of edge sets using random sampling from smaller set.
        If graphs are huge, this avoids full set operations.
        """
        k1 = edge_keys(bg1)
        k2 = edge_keys(bg2)
        # Always sample from smaller
        if k1.numel() > k2.numel():
            k1, k2 = k2, k1

        if k1.numel() == 0 and k2.numel() == 0:
            return 1.0
        if k1.numel() == 0 or k2.numel() == 0:
            return 0.0

        # sample subset of k1
        if k1.numel() > sample_k:
            idx = torch.randint(0, k1.numel(), (sample_k,))
            samp = k1[idx]
            samp, _ = torch.sort(samp)
        else:
            samp = k1

        # intersection approx: count how many sampled edges appear in k2 (binary search)
        # torch.searchsorted requires CPU or same device; keep on CPU for simplicity
        samp_cpu = samp.cpu()
        k2_cpu = k2.cpu()
        pos = torch.searchsorted(k2_cpu, samp_cpu)
        pos = torch.clamp(pos, 0, k2_cpu.numel() - 1)
        hit = (k2_cpu[pos] == samp_cpu).float().mean().item()

        # Approx Jaccard: hit approximates |E1∩E2|/|E1|
        # J ≈ (hit*|E1|) / (|E1| + |E2| - hit*|E1|)
        E1 = float(k1.numel())
        E2 = float(k2.numel())
        inter = hit * E1
        j = inter / (E1 + E2 - inter + 1e-12)
        return float(j)


    def od_stats(od: torch.Tensor):
        od_cpu = od.detach().cpu()
        return {
            "shape": tuple(od_cpu.shape),
            "mean": float(od_cpu.mean()),
            "std": float(od_cpu.std()),
            "min": float(od_cpu.min()),
            "max": float(od_cpu.max()),
            "nonzero_ratio": float((od_cpu != 0).float().mean()),
        }


    def od_diff_metrics(od_a: torch.Tensor, od_b: torch.Tensor):
        diff = (od_a - od_b).detach().cpu()
        return {
            "L1_mean": float(diff.abs().mean()),
            "L2_rmse": float((diff.pow(2).mean()).sqrt()),
            "max_abs": float(diff.abs().max()),
        }


    # -----------------------------
    # 3) Graph structure diagnostics across days
    # -----------------------------
    print("\n=== Graph Diagnostics (Structure across days) ===")
    # Basic per-day graph stats
    for t, bg in enumerate(bgs_by_day):
        print(f"day {t}: num_nodes={bg.num_nodes()}, num_edges={bg.num_edges()}")

    # Exact edge-set equality checks vs day0
    print("\n[Exact] Edge-set equality vs day0:")
    k0 = edge_keys(bgs_by_day[0])
    for t in range(1, len(bgs_by_day)):
        kt = edge_keys(bgs_by_day[t])
        same = (k0.numel() == kt.numel()) and torch.equal(k0, kt)
        print(f"day0 vs day{t}: identical_edges={same}")

    # Approx Jaccard similarity (useful if exact is too strict / for sanity)
    print("\n[Approx] Edge-set Jaccard similarity vs day0:")
    for t in range(1, len(bgs_by_day)):
        j = edge_jaccard(bgs_by_day[0], bgs_by_day[t], sample_k=200000)
        print(f"day0 vs day{t}: Jaccard≈{j:.6f}")

    # -----------------------------
    # 4) OD diagnostics across days (pick a few samples in the batch)
    # -----------------------------
    print("\n=== OD Diagnostics (Values across days) ===")
    T = len(ods_by_day)
    B = len(ods_by_day[0])
    print(f"T={T}, batch_size(B)={B}")

    # choose up to 3 samples to inspect
    inspect_bs = list(range(min(3, B)))

    for b in inspect_bs:
        print(f"\n--- Sample b={b} (geo_id={geoids[b]}) ---")
        # per-day stats
        for t in range(T):
            st = od_stats(ods_by_day[t][b])
            print(f"day{t} stats:", st)

        # differences vs day0
        od0 = ods_by_day[0][b]
        print("\nDiff vs day0:")
        for t in range(1, T):
            m = od_diff_metrics(od0, ods_by_day[t][b])
            print(f"day0 vs day{t} diff:", m)

    # -----------------------------
    # 5) Optional: sanity checks for ctrl/t_context variability (node-level)
    # -----------------------------
    print("\n=== Optional: Node-level ctrl / time context variability ===")
    # Use first 10 nodes to avoid huge prints
    n_show = 10
    t0 = bgs_by_day[0].ndata["t_context"][:n_show]
    t6 = bgs_by_day[-1].ndata["t_context"][:n_show]
    print("t_context day0 vs dayLast abs-sum:", float((t0 - t6).abs().sum().cpu()))

    p0 = bgs_by_day[0].ndata["poi_vec"]
    p6 = bgs_by_day[-1].ndata["poi_vec"]
    print("poi_vec day0 vs dayLast change ratio:", float((p0 != p6).float().mean().cpu()))

    d0 = bgs_by_day[0].ndata["demo_vec"]
    d6 = bgs_by_day[-1].ndata["demo_vec"]
    print("demo_vec day0 vs dayLast change ratio:", float((d0 != d6).float().mean().cpu()))
