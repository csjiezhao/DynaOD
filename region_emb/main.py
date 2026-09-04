import os
import argparse
from typing import Dict, List, Tuple
import torch
from dgl.dataloading import GraphDataLoader
from region_emb.data_load import CityGraphDataset, MaxNodeBatchSampler, collate_graphs, load_city_graph
from region_emb.model import BGRL, drop_edge, drop_feature, drop_feature_dim
import torch.nn.functional as F


@torch.no_grad()
def quick_collapse_metrics(z: torch.Tensor, num_pairs: int = 20000):
    # z: [N, d] (raw)
    zn = F.normalize(z, dim=-1)
    n = zn.shape[0]
    i = torch.randint(0, n, (num_pairs,), device=zn.device)
    j = torch.randint(0, n, (num_pairs,), device=zn.device)
    mask = i != j
    i, j = i[mask], j[mask]
    cos = (zn[i] * zn[j]).sum(-1)
    return cos.mean().item(), cos.std(unbiased=False).item()


def train_bgrl(
    model: BGRL,
    dataloader: GraphDataLoader,
    device: str,
    epochs: int,
    lr: float,
    feat_drop: float,
    edge_drop: float,
    ckpt_path: str,
    save_every: int = 1,
):
    opt = torch.optim.AdamW(
        list(model.student.parameters()) + list(model.predictor.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    best_loss = float("inf")

    # 估计每个 epoch 有多少个 batch（不一定完全准确，但足够用来设置 probe 频率）
    try:
        num_batches = len(dataloader)
    except TypeError:
        num_batches = 0

    # 目标：每个 epoch probe ~3 次；如果太小就每 5 step probe
    probe_every = max(1, num_batches // 3) if num_batches and num_batches > 0 else 5

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        # aug-probe stats
        aug_mean_sum = 0.0
        aug_std_sum = 0.0

        # clean-eval probe stats
        clean_mean_sum = 0.0
        clean_std_sum = 0.0

        probe_cnt = 0
        last_g1 = last_x1 = None
        last_bg = last_x = None

        for step, (cities, bg) in enumerate(dataloader, start=1):
            bg = bg.to(device)
            x = bg.ndata["x"].to(device)

            last_bg, last_x = bg, x

            # ✅ asymmetric augmentation strengths (weak/strong views)
            edge_drop_1, edge_drop_2 = edge_drop, min(0.3, edge_drop * 2)
            feat_drop_1, feat_drop_2 = feat_drop, min(0.5, feat_drop * 2)

            # two augmented views (graph + features)
            g1 = drop_edge(bg.clone(), edge_drop_1)
            g2 = drop_edge(bg.clone(), edge_drop_2)
            x1 = drop_feature_dim(x, feat_drop_1)
            x2 = drop_feature_dim(x, feat_drop_2)

            last_g1, last_x1 = g1, x1

            loss = model(g1, x1, g2, x2)

            opt.zero_grad()
            loss.backward()
            opt.step()
            model.update_teacher()

            epoch_loss += loss.item()

            # ✅ probe a few times per epoch (adaptive): aug + clean-eval
            if step % probe_every == 0:
                with torch.no_grad():
                    # aug-probe
                    z_aug = model.student(g1, x1)
                    cm_aug, cs_aug = quick_collapse_metrics(z_aug)

                    # clean-eval probe (no augmentation)
                    model.student.eval()
                    z_clean = model.student(bg, x)
                    cm_clean, cs_clean = quick_collapse_metrics(z_clean)
                    model.student.train()

                aug_mean_sum += cm_aug
                aug_std_sum += cs_aug
                clean_mean_sum += cm_clean
                clean_std_sum += cs_clean
                probe_cnt += 1

        # avg loss
        denom = (num_batches if num_batches and num_batches > 0 else step)
        avg_loss = epoch_loss / max(1, denom)

        # epoch-end fallback probe (ensure at least 1 probe)
        if probe_cnt == 0 and (last_bg is not None) and (last_g1 is not None):
            with torch.no_grad():
                z_aug = model.student(last_g1, last_x1)
                cm_aug, cs_aug = quick_collapse_metrics(z_aug)

                model.student.eval()
                z_clean = model.student(last_bg, last_x)
                cm_clean, cs_clean = quick_collapse_metrics(z_clean)
                model.student.train()

            aug_mean_sum += cm_aug
            aug_std_sum += cs_aug
            clean_mean_sum += cm_clean
            clean_std_sum += cs_clean
            probe_cnt = 1

        aug_pm = aug_mean_sum / max(1, probe_cnt)
        aug_ps = aug_std_sum / max(1, probe_cnt)
        clean_pm = clean_mean_sum / max(1, probe_cnt)
        clean_ps = clean_std_sum / max(1, probe_cnt)

        print(
            f"[Epoch {epoch:03d}] loss={avg_loss:.4f} | "
            f"aug-probe mean={aug_pm:.4f} std={aug_ps:.4f} | "
            f"clean-eval mean={clean_pm:.4f} std={clean_ps:.4f} "
            f"(n={probe_cnt}, every={probe_every})"
        )

        # collapse warning based on clean-eval (更关键)
        if clean_pm > 0.98 and clean_ps < 0.02:
            print("   ⚠️ clean-eval looks highly anisotropic (near-collapse). Consider LN/centering or post-process.")

        # save
        if epoch % save_every == 0:
            state = {
                "epoch": epoch,
                "loss": avg_loss,
                "model": model.state_dict(),
                "config": {
                    "feat_drop": feat_drop,
                    "edge_drop": edge_drop,
                    "lr": lr,
                    "epochs": epochs,
                    "tau": model.tau,
                    "weight_decay": 1e-4,
                    "probe_every": probe_every,
                    "asym_aug": {
                        "edge_drop_1": edge_drop,
                        "edge_drop_2": min(0.3, edge_drop * 2),
                        "feat_drop_1": feat_drop,
                        "feat_drop_2": min(0.5, feat_drop * 2),
                    },
                }
            }
            torch.save(state, ckpt_path)

        if avg_loss < best_loss:
            best_loss = avg_loss

    print(f"✅ Training done. Best (approx) loss={best_loss:.4f}")
    return best_loss




@torch.no_grad()
def export_all_zone_embeddings(
    model: BGRL,
    dataset: CityGraphDataset,
    data_path: str,
    device: str,
    out_path: str,
    add_self_loop: bool = False,
    normalize: bool = True,
):
    """
    导出全库所有 zone embedding，便于后续建向量库（RAG）。
    输出：
      - keys: List[Tuple[str,int]]  (city, local_zone_idx)
      - emb:  FloatTensor [total_zones, d]
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    model.eval()
    keys: List[Tuple[str, int]] = []
    embs: List[torch.Tensor] = []

    for city in dataset.cities:
        city_dir = os.path.join(data_path, city)
        g = load_city_graph(city_dir, add_self_loop=add_self_loop)
        g = g.to(device)
        x = g.ndata["x"].to(device)

        z = model.encode(g, x, normalize=normalize)  # (N, d)
        z = z.detach().cpu()
        embs.append(z)

        keys.extend([(city, i) for i in range(z.shape[0])])

    emb = torch.cat(embs, dim=0)

    payload = {
        "keys": keys,
        "emb": emb,
        "dim": emb.shape[1],
        "normalize": normalize,
    }
    torch.save(payload, out_path)

    print(f"✅ Saved embeddings: {out_path}")
    print(f"   total_zones={emb.shape[0]}, dim={emb.shape[1]}")
    return out_path


def build_dataloader(
    data_path: str,
    max_cities: int,
    batch_size: int,
    max_nodes: int,
    add_self_loop: bool,
    shuffle_cities: bool,
    num_workers: int = 0,
):
    dataset = CityGraphDataset(
        data_path=data_path,
        max_cities=max_cities,
        shuffle_cities=1 if shuffle_cities else 0,
        add_self_loop=add_self_loop,
    )
    sampler = MaxNodeBatchSampler(dataset, batch_size=batch_size, max_nodes=max_nodes)
    loader = GraphDataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_graphs,
        num_workers=num_workers,
    )
    return dataset, loader


def infer_in_dim(dataset: CityGraphDataset, data_path: str, add_self_loop: bool) -> int:
    # 用第一座城市推断特征维度
    city0 = dataset.cities[0]
    g0 = load_city_graph(os.path.join(data_path, city0), add_self_loop=add_self_loop)
    return int(g0.ndata["x"].shape[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/")
    parser.add_argument("--max_cities", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda:0")

    # batching
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_nodes", type=int, default=1500
                        )
    parser.add_argument("--add_self_loop", action="store_true")
    parser.add_argument("--no_shuffle", action="store_true")

    # model
    parser.add_argument("--hid_dim", type=int, default=256)
    parser.add_argument("--out_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.995)

    # augmentation
    parser.add_argument("--feat_drop", type=float, default=0.2)
    parser.add_argument("--edge_drop", type=float, default=0.4)

    # train
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--ckpt_path", type=str, default="ckpts/bgrl_model.pth")

    # export
    parser.add_argument("--embed_out", type=str, default="ckpts/zone_embeddings.pt")
    parser.add_argument("--export_normalize", action="store_true")

    args = parser.parse_args()

    dataset, loader = build_dataloader(
        data_path=args.data_path,
        max_cities=args.max_cities,
        batch_size=args.batch_size,
        max_nodes=args.max_nodes,
        add_self_loop=args.add_self_loop,
        shuffle_cities=not args.no_shuffle,
        num_workers=4,
    )

    in_dim = infer_in_dim(dataset, args.data_path, args.add_self_loop)
    print(f"✅ inferred in_dim={in_dim}, num_cities={len(dataset)}")

    model = BGRL(
        in_dim=in_dim,
        hid_dim=args.hid_dim,
        out_dim=args.out_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        tau=args.tau,
    ).to(args.device)

    # train
    train_bgrl(
        model=model,
        dataloader=loader,
        device=args.device,
        epochs=args.epochs,
        lr=args.lr,
        feat_drop=args.feat_drop,
        edge_drop=args.edge_drop,
        ckpt_path=args.ckpt_path,
        save_every=1,
    )

    # export embeddings for RAG
    export_all_zone_embeddings(
        model=model,
        dataset=dataset,
        data_path=args.data_path,
        device=args.device,
        out_path=args.embed_out,
        add_self_loop=args.add_self_loop,
        normalize=args.export_normalize,
    )


if __name__ == "__main__":
    main()
