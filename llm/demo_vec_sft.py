#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import numpy as np
from tqdm import tqdm
from json_repair import repair_json

from llm.llm_api import LLMCaller
from llm_distillation.ctrl_vec_readout import DEMO_CTRL_PROMPT
from models.DynaOD.data_load import load_samples

_thread_local = threading.local()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--platform", default="vLLM")
    p.add_argument("--model", default="qwen-2.5-1.5b-sft")
    p.add_argument("--data_path", default="data/")
    p.add_argument("--parallel", type=int, default=32, help="并发线程数（建议 16~128）")
    p.add_argument("--overwrite", action="store_true", help="覆盖已存在文件")
    p.add_argument("--max_tracts", type=int, default=None, help="调试：每个城市只跑前 N 个 tract")
    p.add_argument("--mode", type=str, default="train", choices=["train", "test1", "test2", "test3"])
    return p.parse_args()

def load_city2tract(json_path="county2tract.json"):
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def get_thread_llm(platform: str, model: str) -> LLMCaller:
    """每个线程一个 LLMCaller（thread-local 单例）"""
    llm = getattr(_thread_local, "llm", None)
    if llm is None:
        llm = LLMCaller(platform, model)
        _thread_local.llm = llm
    return llm


def call_one_tract(tract_id: str, date: str, poi_ctrl: list, platform: str, model: str) -> list[int]:
    """返回 97 维 [-1,0,1] 列表"""
    llm = get_thread_llm(platform, model)
    prompt = DEMO_CTRL_PROMPT.format(
        TRACT=tract_id,
        POIS_CTRL_VEC=poi_ctrl,
        DATE=date
    )
    try:
        raw = llm.get_response([{"role": "user", "content": prompt}])
        resp = json.loads(repair_json(raw))
        vec = resp.get(tract_id, [])
    except Exception as e:
        tqdm.write(f"❌ tract {tract_id} parse error: {e}")
        vec = []
    vec = (vec[:97] + [0] * (97 - len(vec)))[:97]
    return vec


def process_city_date(city_id: str, date: str, args, city2tract: dict, pool: ThreadPoolExecutor):
    save_file = Path(args.data_path) / city_id / "demo_vecs" / f"{args.model}_demo_vec_{date}.npy"
    if save_file.exists() and not args.overwrite:
        tqdm.write(f"⏩ Skip  {city_id}  {date}  (already exists)")
        return

    # 加载 poi_ctrl_vec（必须与 tract 顺序一致）
    poi_ctrl_path = Path(args.data_path) / city_id / "poi_vecs" / f"{args.model}_poi_vec_{date}.npy"
    if not poi_ctrl_path.exists():
        tqdm.write(f"⚠️  poi_vec not found for {city_id} {date}, skip")
        return
    poi_ctrl_vec = np.load(poi_ctrl_path)  # (N, 34)

    tracts = [city_id + e for e in city2tract[city_id]]
    if args.max_tracts:
        tracts = tracts[: args.max_tracts]
        poi_ctrl_vec = poi_ctrl_vec[: args.max_tracts]

    if len(tracts) != len(poi_ctrl_vec):
        tqdm.write(f"⚠️  tract count != poi_vec rows for {city_id} {date} "
                   f"({len(tracts)} vs {len(poi_ctrl_vec)}), skip")
        return

    city_vecs = [None] * len(tracts)

    futures = {
        pool.submit(
            call_one_tract,
            tid,
            date,
            poi_ctrl_vec[i].tolist(),
            args.platform,
            args.model
        ): i
        for i, tid in enumerate(tracts)
    }

    for fut in as_completed(futures):
        i = futures[fut]
        city_vecs[i] = fut.result()

    vec_arr = np.asarray(city_vecs, dtype=np.float32)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_file, vec_arr)

    tqdm.write(
        f"💾 {city_id} {date}  shape={vec_arr.shape}  "
        f"nonzero={np.count_nonzero(vec_arr) / vec_arr.size:.1%}"
    )


def main():
    args = parse_args()
    city2tract = load_city2tract()
    cities, dates = load_samples(args.data_path, True, 0.7, args.mode, True)

    tasks = [(c, d) for c in cities for d in dates]

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for city_id, date in tqdm(tasks, desc=f"city-date ({args.mode})"):
            process_city_date(city_id, date, args, city2tract, pool)


if __name__ == "__main__":
    main()
