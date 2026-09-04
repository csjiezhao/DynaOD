from models.DynaOD.data_load import load_window_samples, CityWindowDataset, MyBatchSampler, collate_fn_window
from dgl.dataloading import GraphDataLoader
import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':
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

    T = len(bgs_by_day)
    B = len(geoids)

    # 选 batch 里的第 k 个城市
    k = 0
    print("Selected geo_id:", geoids[k])

    # ========== 1️⃣ 用 day 0 的 OD 选代表节点 ==========
    ods_day0 = ods_by_day[0][k]  # (n_k, n_k)

    # 节点总流量（in + out）
    node_flow = ods_day0.sum(dim=0) + ods_day0.sum(dim=1)
    rep_idx = torch.argmax(node_flow).item()

    print(f"Representative node index (city {k}):", rep_idx)

    # ========== 2️⃣ 收集 7 天该节点的 poi_ctrl vec ==========
    poi_seq = []  # (T, 34)

    for t in range(T):
        bg_t = bgs_by_day[t]
        poi_all = bg_t.ndata["poi_vec"]  # (N_total, 34)

        # 计算 batched node offset
        sizes = [od.shape[0] for od in ods_by_day[t]]
        l = sum(sizes[:k])
        r = l + sizes[k]

        poi_city = poi_all[l:r]  # (n_k, 34)
        poi_rep = poi_city[rep_idx]  # (34,)

        poi_seq.append(poi_rep)

    poi_seq = torch.stack(poi_seq, dim=0)  # (T, 34)
    print("poi_seq shape:", poi_seq.shape)

    # ========== 3️⃣ 画前 5 维 ==========
    days = list(range(T))
    plt.figure(figsize=(8, 4))
    for d in range(34):
        plt.plot(days, poi_seq[:, d].detach().cpu().numpy(),
                 marker="o", label=f"poi_dim_{d}")
        print(poi_seq[:, d].detach().cpu().numpy(), f"poi_dim_{d}")

    plt.xlabel("day")
    plt.ylabel("poi ctrl value")
    plt.title(f"Representative node POI ctrl (first 5 dims)\ngeo_id={geoids[k]}, node={rep_idx}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()