from models.DynaOD.model import ShapeNet
from models.DynaOD.data_load import load_window_samples, CityWindowDataset
from setproctitle import setproctitle
from tqdm import tqdm
import json
import torch
import os


def construct_memory(config, dset, model, all_geoids, out_path=None,
                     eps=1e-8, logvar_default=-4.0):
    device = config["device"]
    poi_dim = model.poi_dim
    demo_dim = model.demo_dim

    if out_path is None:
        out_path = os.path.join(config["ckpt_path"], "ShapeMem_weekday.pt")

    model.to(device)
    model.eval()

    # ---- accumulators: tract11 -> tensors on CPU ----
    mem_sum_mu_poi = {}
    mem_sum_m2_poi = {}
    mem_sum_mu_demo = {}
    mem_sum_m2_demo = {}
    mem_count = {}

    def _init_tract(tract11: str):
        mem_sum_mu_poi[tract11] = torch.zeros(7, poi_dim, dtype=torch.float32)
        mem_sum_m2_poi[tract11] = torch.zeros(7, poi_dim, dtype=torch.float32)
        mem_sum_mu_demo[tract11] = torch.zeros(7, demo_dim, dtype=torch.float32)
        mem_sum_m2_demo[tract11] = torch.zeros(7, demo_dim, dtype=torch.float32)
        mem_count[tract11] = torch.zeros(7, dtype=torch.int32)

    with torch.no_grad():
        for idx in tqdm(range(len(dset)), desc="Build ShapeMem"):
            geoid, graphs_seq, dis, ods_seq = dset[idx]
            county5 = geoid.split("_")[0]

            # county -> tract6 list
            if county5 not in all_geoids:
                continue
            tract6_list = all_geoids[county5]  # list[str] each is 6-digit tract id
            tracts_geoid11 = [county5 + str(t).zfill(6) for t in tract6_list]  # list[str]
            num_tracts = int(dis.shape[0])

            # sanity check: local order must match node order
            if len(tracts_geoid11) != num_tracts:
                raise ValueError(
                    f"[ShapeMem] tract count mismatch for county={county5}: "
                    f"county2tract has {len(tracts_geoid11)} tracts, but data has N={num_tracts}"
                )

            # ---- build ShapeNet inputs ----
            g0 = graphs_seq[0]
            n0 = g0.ndata["demo"].to(device)  # (N,131) = [demo, poi] (already scaled)
            poi_feat = n0[:, -poi_dim:].clone()
            demo_feat = n0[:, :-poi_dim].clone()

            poi_ctrl_seq = torch.stack([g.ndata["poi_vec"].to(device) for g in graphs_seq], dim=1)    # (N,7,34)
            demo_ctrl_seq = torch.stack([g.ndata["demo_vec"].to(device) for g in graphs_seq], dim=1)  # (N,7,97)
            t_context_seq = torch.stack([g.ndata["t_context"].to(device) for g in graphs_seq], dim=1) # (N,7,8)

            # weekday_id per day: all nodes share same t_context, so use node0
            weekday_ids = torch.argmax(t_context_seq[0, :, :7], dim=-1).to(torch.int64)  # (7,)

            out = model(poi_feat, demo_feat, poi_ctrl_seq, demo_ctrl_seq, t_context_seq)

            poi_mu = out["poi_mu_seq"].detach().to("cpu", dtype=torch.float32)         # (N,7,34)
            poi_lv = out["poi_logvar_seq"].detach().to("cpu", dtype=torch.float32)     # (N,7,34)
            demo_mu = out["demo_mu_seq"].detach().to("cpu", dtype=torch.float32)       # (N,7,97)
            demo_lv = out["demo_logvar_seq"].detach().to("cpu", dtype=torch.float32)   # (N,7,97)

            poi_var = torch.exp(poi_lv)    # (N,7,34)
            demo_var = torch.exp(demo_lv)  # (N,7,97)

            # ---- aggregate into weekday buckets per tract ----
            # For each day t, bucket = weekday_ids[t]
            for t in range(poi_mu.shape[1]):  # T=7
                w = int(weekday_ids[t].item())  # 0..6
                for i_tr, tract11 in enumerate(tracts_geoid11):
                    if tract11 not in mem_count:
                        _init_tract(tract11)

                    mu_p = poi_mu[i_tr, t]   # (34,)
                    v_p = poi_var[i_tr, t]   # (34,)
                    mu_d = demo_mu[i_tr, t]  # (97,)
                    v_d = demo_var[i_tr, t]  # (97,)

                    # sum of means
                    mem_sum_mu_poi[tract11][w] += mu_p
                    mem_sum_mu_demo[tract11][w] += mu_d

                    # sum of second moments: E[x^2] = var + mu^2
                    mem_sum_m2_poi[tract11][w] += (v_p + mu_p.pow(2))
                    mem_sum_m2_demo[tract11][w] += (v_d + mu_d.pow(2))

                    # count
                    mem_count[tract11][w] += 1

    # =========================
    # Finalize: moment matching
    # =========================
    tract11_list = sorted(mem_count.keys())
    M = len(tract11_list)

    poi_mu_week = torch.zeros(M, 7, poi_dim, dtype=torch.float32)
    poi_logvar_week = torch.zeros(M, 7, poi_dim, dtype=torch.float32)
    demo_mu_week = torch.zeros(M, 7, demo_dim, dtype=torch.float32)
    demo_logvar_week = torch.zeros(M, 7, demo_dim, dtype=torch.float32)
    count_week = torch.zeros(M, 7, dtype=torch.int32)

    for i, tract11 in enumerate(tract11_list):
        cnt = mem_count[tract11].to(torch.float32)          # (7,)
        count_week[i] = mem_count[tract11]                  # int32 (7,)

        cnt_safe = cnt.clamp_min(1.0).unsqueeze(-1)         # (7,1)

        # mu_final
        mu_p = mem_sum_mu_poi[tract11] / cnt_safe           # (7,34)
        mu_d = mem_sum_mu_demo[tract11] / cnt_safe          # (7,97)

        # var_final = E[x^2] - (E[x])^2
        var_p = mem_sum_m2_poi[tract11] / cnt_safe - mu_p.pow(2)
        var_d = mem_sum_m2_demo[tract11] / cnt_safe - mu_d.pow(2)

        var_p = var_p.clamp_min(eps)
        var_d = var_d.clamp_min(eps)

        lv_p = torch.log(var_p)
        lv_d = torch.log(var_d)

        # buckets never seen: set default (mu=0, logvar=logvar_default)
        unseen = (cnt == 0)  # (7,)
        if unseen.any():
            mu_p[unseen] = 0.0
            mu_d[unseen] = 0.0
            lv_p[unseen] = logvar_default
            lv_d[unseen] = logvar_default

        poi_mu_week[i] = mu_p
        poi_logvar_week[i] = lv_p
        demo_mu_week[i] = mu_d
        demo_logvar_week[i] = lv_d


    # ===== build concrete shapes =====
    # Option A: deterministic (recommended)
    # poi_shape_week = poi_mu_week.clone()
    # demo_shape_week = demo_mu_week.clone()

    # Option B: sampled aggregate (optional)
    torch.manual_seed(2026)  # uncomment if you want reproducible sampled shapes
    poi_shape_week, _ = model._sample_and_aggregate(poi_mu_week.to(device), poi_logvar_week.to(device), None)
    demo_shape_week, _ = model._sample_and_aggregate(demo_mu_week.to(device), demo_logvar_week.to(device), None)
    poi_shape_week = poi_shape_week.detach().cpu()
    demo_shape_week = demo_shape_week.detach().cpu()

    mem = {
        "tract11_list": tract11_list,       # List[str], length M
        "poi_mu_week": poi_mu_week,         # (M,7,34)
        "poi_logvar_week": poi_logvar_week, # (M,7,34)
        "demo_mu_week": demo_mu_week,       # (M,7,97)
        "demo_logvar_week": demo_logvar_week,# (M,7,97)
        "poi_shape_week": poi_shape_week,  # ✅ 新增
        "demo_shape_week": demo_shape_week,  # ✅ 新增
        "count_week": count_week,           # (M,7)
        "weekday_def": "0..6 from t_context one-hot",
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(mem, out_path)
    print(f"✅ Saved ShapeMem to {out_path}")
    print(f"   tracts={M}, poi_dim={poi_dim}, demo_dim={demo_dim}")


class ShapeMemLib:
    def __init__(self, path: str, device="cpu"):
        self.mem = torch.load(path, map_location="cpu")
        self.tract11_list = self.mem["tract11_list"]
        self.t2i = {t:i for i,t in enumerate(self.tract11_list)}
        self.device = device

        # tensors
        for k in ["poi_mu_week","poi_logvar_week","demo_mu_week","demo_logvar_week", "poi_shape_week", "demo_shape_week", "count_week"]:
            self.mem[k] = self.mem[k].to(device)

    def get(self, tract11: str, weekday: int):
        i = self.t2i.get(tract11)
        if i is None:
            return None  # not found
        w = int(weekday)

        return self.mem["poi_mu_week"][i, w], self.mem["demo_mu_week"][i, w]
        # return (
        #     self.mem["poi_mu_week"][i, w],
        #     self.mem["poi_logvar_week"][i, w],
        #     self.mem["demo_mu_week"][i, w],
        #     self.mem["demo_logvar_week"][i, w],
        #     self.mem["poi_shape_week"][i, w],
        #     self.mem["demo_shape_week"][i, w],
        #     self.mem["count_week"][i, w],
        # )


if __name__ == '__main__':
    setproctitle("DynaOD-ShapeMem")

    model_config = {
        "device": "cuda:3",
        "data_path": "data/",
        "ckpt_path": "ckpts/",
        "shape_ckpt": "ckpts/shapenet_model_best.pth",
        "split_ratio": 0.7,
        "geoid_path": "county2tract.json",
    }

    train_data = load_window_samples(
        data_path=model_config["data_path"],
        shuffle_cities=True,
        split_ratio=model_config["split_ratio"],
        mode='train'
    )
    train_set = CityWindowDataset(*train_data)

    # ---- load ShapeNet weights from DynaOD ckpt ----
    shape_model = ShapeNet()
    ckpt = torch.load(model_config["shape_ckpt"], map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    shape_state = {k.replace("shape_net.", "", 1): v for k, v in state.items() if k.startswith("shape_net.")}
    missing, unexpected = shape_model.load_state_dict(shape_state, strict=False)
    print("missing keys:", missing)
    print("unexpected keys:", unexpected)

    with open(model_config["geoid_path"], encoding="utf-8") as f:
        city_geoids = json.load(f)

    out_path = os.path.join(model_config["ckpt_path"], "ShapeMem_weekday.pt")
    construct_memory(model_config, train_set, shape_model, city_geoids, out_path=out_path)
