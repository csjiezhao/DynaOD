from models.DynaOD.shape_memory import ShapeMemLib
from region_emb.emb_readout import CityEmbLib
from models.DynaOD.data_load import load_window_samples

from datetime import datetime
import json
import os
import numpy as np
from tqdm import tqdm


def weekday_from_datestr(d: str) -> int:
    return datetime.strptime(d, "%Y_%m_%d").weekday()  # 0..6

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def export_city_near_shapes(cities, windows, city2tract, city_emb, shp_mem, out_root="data", K=3, poi_dim=34, demo_dim=97):
    """
    For each city:
      - save near_k_scores.npy: (N,K)
      - for each date: save
          poi_shps/poi_shp_YYYY_MM_DD.npy: (N,K,poi_dim)
          demo_shps/demo_shp_YYYY_MM_DD.npy: (N,K,demo_dim)
    """

    all_dates = []
    for date_list in windows:
        all_dates.extend(date_list)
    all_dates = sorted(list(dict.fromkeys(all_dates)))  # preserve order & dedup

    for city_id in tqdm(cities, desc="export cities"):
        # 1) tract list (keep your way, no extra zfill)
        tract6_list = city2tract[city_id]
        tract_list = [str(city_id) + e for e in tract6_list]   # tract11 by your convention
        N = len(tract_list)

        # 2) compute neighbors once per tract
        near_ids = np.empty((N, K), dtype=object)
        near_scores = np.zeros((N, K), dtype=np.float32)

        for i, tract11 in enumerate(tract_list):
            nears = city_emb.topk_similar_tracts_in_set(
                tract11,
                k=K,
                allowed_tracts=shp_mem.tract11_list,
                return_scores=True,
            )
            # fill up to K
            for k in range(K):
                if k < len(nears):
                    tid, score = nears[k]
                    near_ids[i, k] = tid
                    near_scores[i, k] = float(score)
                else:
                    near_ids[i, k] = None
                    near_scores[i, k] = 0.0

        # 3) create output dirs
        city_dir = os.path.join(out_root, str(city_id))
        poi_dir = os.path.join(city_dir, "poi_shps")
        demo_dir = os.path.join(city_dir, "demo_shps")
        ensure_dir(poi_dir)
        ensure_dir(demo_dir)

        # 4) save scores once per city
        np.save(os.path.join(city_dir, "near_k_ids.npy"), near_ids)
        np.save(os.path.join(city_dir, "near_k_scores.npy"), near_scores)

        # 5) for each date, build (N,K,D) tensors from ShapeMem
        #    (mu as shape evidence)
        for date_str in all_dates:
            w = weekday_from_datestr(date_str)

            poi_arr = np.zeros((N, K, poi_dim), dtype=np.float32)
            demo_arr = np.zeros((N, K, demo_dim), dtype=np.float32)

            for i in range(N):
                for k in range(K):
                    tid = near_ids[i][k]
                    if tid is None:
                        continue
                    pack = shp_mem.get(tid, w)  # your simplified get: returns (poi_mu, demo_mu) or None
                    if pack is None:
                        continue
                    poi_mu, demo_mu = pack
                    # convert tensor -> numpy
                    poi_arr[i, k, :] = poi_mu.detach().cpu().numpy()
                    demo_arr[i, k, :] = demo_mu.detach().cpu().numpy()

            np.save(os.path.join(poi_dir, f"poi_shp_{date_str}.npy"), poi_arr)
            np.save(os.path.join(demo_dir, f"demo_shp_{date_str}.npy"), demo_arr)

        print(f"✅ done city={city_id}: N={N}, dates={len(all_dates)}, saved to {city_dir}")





if __name__ == '__main__':
    device = "cpu"

    test_cities, test_windows = load_window_samples(data_path='data/',
                                                    shuffle_cities=True, is_simplify=True,
                                                    split_ratio=0.7, mode='train')
    with open("county2tract.json", encoding="utf-8") as f:
        county2tract = json.load(f)

    CityEmb = CityEmbLib(emb_path='ckpts/zone_embeddings.pt', geoid_path="county2tract.json", device=device)
    ShapeEmb = ShapeMemLib(path='ckpts/ShapeMem_weekday.pt', device=device)

    export_city_near_shapes(
        cities=test_cities,
        windows=test_windows,
        city2tract=county2tract,
        city_emb=CityEmb,
        shp_mem=ShapeEmb,
        out_root="data",
        K=5
    )