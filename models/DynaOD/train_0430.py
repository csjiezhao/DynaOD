from models.DynaOD.model import DynaOD
from models.DynaOD.data_load import (load_window_samples, load_window_samples_ijcai,
                                     CityWindowDataset, MyBatchSampler, collate_fn_window, SCALERS)
from models.DynaOD.loss import mse_loss
from models.WeDAN.utils.metrics import cal_od_metrics, average_listed_metrics

from setproctitle import setproctitle
import os
from tqdm import tqdm
from dgl.dataloading import GraphDataLoader
import torch


def train_process(config, dloader, model, optim, val_loader):
    device = config['device']

    # ===== fixed early-stopping settings (RMSE) =====
    patience = 8   # <-- 原来 8，fine-tune 建议 12
    min_delta = 1e-4  # require RMSE improvement >= min_delta
    best_rmse = float("inf")
    best_epoch = -1
    bad_epochs = 0

    for epoch in tqdm(range(config['shapenet_epoch'])):
        model.train()
        epoch_loss = 0.0

        for i, batch in enumerate(dloader):
            geoids, bgs_by_day, batched_dis, ods_by_day = batch
            T = len(bgs_by_day)  # num_days

            batched_dis = batched_dis.to(device)
            poi_vec_list, demo_vec_list, t_context_list = [], [], []
            net_list, batchlization_list = [], []
            ods_day_list = []

            for t in range(T):
                bg_t = bgs_by_day[t].to(device)

                ods_t_cpu = ods_by_day[t]
                ods_t = [od.to(device) for od in ods_t_cpu]
                ods_day_list.append(ods_t)

                n_t = bg_t.ndata["demo"].to(device)
                e_t = torch.block_diag(*ods_t)
                net_list.append((n_t, e_t))

                batchlization_t = torch.block_diag(*[torch.ones_like(od) for od in ods_t])
                batchlization_list.append(batchlization_t)

                poi_vec_list.append(bg_t.ndata["poi_vec"].to(device))
                demo_vec_list.append(bg_t.ndata["demo_vec"].to(device))
                t_context_list.append(bg_t.ndata["t_context"].to(device))

            od_pairs_seq, reg_loss = model(
                poi_vec_list, demo_vec_list, t_context_list, None, None,
                ods_day_list, net_list, batched_dis, batchlization_list, SCALERS,
                poi_control=True, demo_control=True, external_shape=False
            )

            loss_sum = None
            loss_cnt = 0
            for t in range(T):
                od_pairs_t = od_pairs_seq[t]
                for j in range(len(od_pairs_t)):
                    od_hat, od_gt = od_pairs_t[j]
                    l = mse_loss(od_hat, od_gt)
                    loss_sum = l if loss_sum is None else (loss_sum + l)
                    loss_cnt += 1
                    od_pairs_t[j] = None

            main_loss = loss_sum / max(1, loss_cnt)

            # reg_loss safety (keep your structure)
            if reg_loss is None:
                reg_loss = 0.0
            if not torch.is_tensor(reg_loss):
                reg_loss = torch.tensor(reg_loss, device=main_loss.device, dtype=main_loss.dtype)

            total_loss = main_loss + reg_loss

            optim.zero_grad(set_to_none=True)
            total_loss.backward()
            optim.step()
            epoch_loss += total_loss.item()

        avg_loss = epoch_loss / max(1, len(dloader))
        print(f"[Epoch {epoch + 1}] Train Avg Loss: {avg_loss:.6f}")

        # ===== validation (use RMSE for early stopping) =====
        config["DDIM_T_sample"] = 5
        config["sample_times"] = 2
        val_metrics = test_process(config, val_loader, model)
        val_rmse = float(val_metrics["RMSE"])
        print(f"[Epoch {epoch + 1}] Val CPC: {val_metrics['CPC']:.4f}, RMSE: {val_rmse:.2f}")

        # ===== save best & early stop =====
        improved = (best_rmse - val_rmse) > min_delta
        if improved:
            best_rmse = val_rmse
            best_epoch = epoch + 1
            bad_epochs = 0

            os.makedirs(config['ckpt_path'], exist_ok=True)
            torch.save(model.state_dict(), os.path.join(config['ckpt_path'], f'shapenet_model_0430.pth'))
            print(f"✅ New best RMSE={best_rmse:.2f} @ epoch {best_epoch}, saved shapenet_model__{config['llm']}_0430.pth")
        else:
            bad_epochs += 1
            print(f"⏳ No RMSE improvement. bad_epochs={bad_epochs}/{patience} (best={best_rmse:.2f} @ epoch {best_epoch})")

            if bad_epochs >= patience:
                print(f"🛑 Early stopping at epoch {epoch + 1}. Best epoch={best_epoch}, best RMSE={best_rmse:.2f}")
                break
        config["DDIM_T_sample"] = 2
        config["sample_times"] = 1

    print(f"✅ Training completed. Best epoch={best_epoch}, best RMSE={best_rmse:.2f}")


def test_process(
    config,
    dloader,
    model,
    poi_control=True,
    demo_control=True,
    external_shape=False,
    return_by_day=False,
):
    device = config["device"]
    model.eval()

    all_metrics = []
    per_day_metrics = None  # list[list[dict]]

    with torch.no_grad():
        for i, batch in tqdm(enumerate(dloader)):
            geoids, bgs_by_day, batched_dis, ods_by_day = batch
            T = len(bgs_by_day)

            if return_by_day and per_day_metrics is None:
                per_day_metrics = [[] for _ in range(T)]

            batched_dis = batched_dis.to(device)

            poi_vec_list, demo_vec_list, t_context_list = [], [], []
            poi_shp_list, demo_shp_list = [], []
            net_list, batchlization_list = [], []
            ods_day_list = []

            # ===== build per-day inputs =====
            for t in range(T):
                bg_t = bgs_by_day[t].to(device)

                ods_t_cpu = ods_by_day[t]  # list length B
                ods_t = [od.to(device) for od in ods_t_cpu]
                ods_day_list.append(ods_t)

                n_t = bg_t.ndata["demo"].to(device)
                e_t = torch.block_diag(*ods_t)
                net_list.append((n_t, e_t))

                batchlization_t = torch.block_diag(*[torch.ones_like(od) for od in ods_t])
                batchlization_list.append(batchlization_t)

                poi_vec_list.append(bg_t.ndata["poi_vec"].to(device))
                demo_vec_list.append(bg_t.ndata["demo_vec"].to(device))
                t_context_list.append(bg_t.ndata["t_context"].to(device))
                poi_shp_list.append(bg_t.ndata["poi_shp"].to(device))
                demo_shp_list.append(bg_t.ndata["demo_shp"].to(device))

            od_pairs_seq = model(
                poi_vec_list, demo_vec_list, t_context_list,
                poi_shp_list, demo_shp_list,
                ods_day_list, net_list, batched_dis, batchlization_list, SCALERS,
                poi_control=poi_control, demo_control=demo_control, external_shape=external_shape
            )

            for t in range(T):
                for od_hat, od_gt in od_pairs_seq[t]:
                    one_metrics = cal_od_metrics(od_hat, od_gt)
                    all_metrics.append(one_metrics)
                    if return_by_day:
                        per_day_metrics[t].append(one_metrics)

    avg_metrics = average_listed_metrics(all_metrics)

    if not return_by_day:
        return avg_metrics

    avg_metrics_by_day = [average_listed_metrics(ms) if len(ms) > 0 else None for ms in per_day_metrics]
    return avg_metrics, avg_metrics_by_day


if __name__ == '__main__':
    setproctitle("DynaOD-ShapeNet")

    model_config = {
        "device": "cuda:0",
        "data_path": "data/",
        "ckpt_path": "ckpts/",
        "odnet_ckpt": "ckpts/wedan_model_8400.pth",
        "split_ratio": 0.7,

        # parameters for ShapeNet
        "shapenet_lr": 5e-4,
        "shapenet_epoch": 500,
        "lambda_smooth": 0.1,
        "alpha_stable": 2,
        "llm": "qwen-2.5-1.5b-sft", # "qwen-2.5-1.5b-sft", "qwen-2.5-7b"

        # parameters for WeDAN
        "batch_size": 8,
        "max_nodes": 1000,
        "DDIM_T_sample": 2,
        "sample_times": 1,

        "attr_MinMax": 1,
        "od_MinMax": 1,
        "skew_norm": "log",
        "pert_node": 1,
        "sample_method": "DDIM",
        "valid_period": 10,
        "overfit_tolerance": 20,
        "hiddim": 32,
        "num_head": 4,
        "num_head_cross": 1,
        "num_layer": 4,
        "dropout": 0,
        "n_indim": 131,
        "e_indim": 2,
        "n_outdim": 131,
        "e_outdim": 1,
        "p_generation": 1,
        "p_featMissing": 0,
        "T": 1000,
        "DDIM_eta": 0,
        "beta_scheduler": "cosine",
        "EPOCH": 20000,
        "city_type_limit": "",
        "LaPE_dim": 0,
        "norm_type": "layer",
        "learning_rate": 1e-3,
        "optm": "AdamW",
        "loss": "mse",
    }

    train_data = load_window_samples_ijcai(data_path=model_config["data_path"], shuffle_cities=True,
                                     split_ratio=model_config["split_ratio"], mode='train', llm=model_config["llm"])
    train_set = CityWindowDataset(*train_data)
    sampler = MyBatchSampler(train_set, model_config["batch_size"], model_config["max_nodes"])
    dataloader = GraphDataLoader(train_set, batch_sampler=sampler, collate_fn=collate_fn_window)

    val_data = load_window_samples(data_path=model_config["data_path"], shuffle_cities=True,
                                     split_ratio=model_config["split_ratio"], mode='test3')
    val_set = CityWindowDataset(*val_data)
    val_sampler = MyBatchSampler(val_set, model_config["batch_size"], model_config["max_nodes"])
    val_loader = GraphDataLoader(val_set, batch_sampler=val_sampler, collate_fn=collate_fn_window)

    dyna_model = DynaOD(diff_config=model_config, pretrained_ckpt=model_config["odnet_ckpt"]).to(model_config["device"])
    optimizer = torch.optim.AdamW(dyna_model.shape_net.parameters(), lr=model_config["shapenet_lr"])
    train_process(model_config, dataloader, dyna_model, optimizer, val_loader)
