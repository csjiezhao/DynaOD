#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from models.DynaOD.data_load import load_samples

# ------------------------
# 1) category names & prompts (verbatim from your message)
# ------------------------

POI_CTRL_PROMPT = """You are given a US tract GEOID and a date.
TASK: POI_CONTROL
Output JSON only. No extra text.
Key MUST be TRACT. Value MUST be a list of EXACTLY 34 integers in (-1,0,1).

TRACT: {TRACT}
DATE: {DATE}
"""


DEMO_CTRL_PROMPT = """You are given a US tract GEOID and a date.
TASK: DEMO_CONTROL
Output JSON only. No extra text.
Key MUST be TRACT. Value MUST be a list of EXACTLY 97 integers in (-1,0,1).

TRACT: {TRACT}
DATE: {DATE}
"""


# ------------------------
# 2) helpers
# ------------------------
def load_city2tract(json_path: str) -> Dict[str, List[str]]:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)

def pad_vec(vec: np.ndarray, dim: int) -> List[int]:
    """
    Ensure length=dim and values in {-1,0,1}.
    """
    v = vec.reshape(-1).astype(np.int64).tolist()
    if len(v) < dim:
        v = v + [0] * (dim - len(v))
    v = v[:dim]
    # clamp to {-1,0,1}
    v = [(-1 if x < 0 else (1 if x > 0 else 0)) for x in v]
    return v

def make_poi_prompt(tract11: str, date: str) -> str:
    return POI_CTRL_PROMPT.format(
        TRACT=tract11,
        DATE=date
    )

def make_demo_prompt(tract11: str, date: str, poi_ctrl_vec: List[int]) -> str:
    return DEMO_CTRL_PROMPT.format(
        TRACT=tract11,
        DATE=date
    )

def parse_args():
    p = argparse.ArgumentParser("Export SFT JSONL from teacher control vectors")
    p.add_argument("--data_path", type=str, default="data")
    p.add_argument("--geoid_path", type=str, default="county2tract.json")
    p.add_argument("--split_ratio", type=float, default=0.7)
    p.add_argument("--mode", type=str, default="test1", choices=["train","test1","test2","test3"])
    p.add_argument("--llm_tag", type=str, default="gpt-4o-mini")
    p.add_argument("--out_dir", type=str, default="sft_data")
    p.add_argument("--max_city_dates", type=int, default=None, help="debug: limit number of (city,date) pairs")
    return p.parse_args()

# ------------------------
# 3) main export
# ------------------------
def export_jsonl(args):
    data_path = args.data_path
    city2tract = load_city2tract(args.geoid_path)

    # load split cities, dates (is_simplify=True)
    cities, dates = load_samples(data_path, True, args.split_ratio, args.mode, True)
    cd_pairs = [(c, d) for c in cities for d in dates]
    if args.max_city_dates is not None:
        cd_pairs = cd_pairs[:args.max_city_dates]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    poi_out = out_dir / f"{args.mode}_poi.jsonl"
    demo_out = out_dir / f"{args.mode}_demo.jsonl"

    n_poi = 0
    n_demo = 0

    with open(poi_out, "w", encoding="utf-8") as f_poi, open(demo_out, "w", encoding="utf-8") as f_demo:
        for city_id, date in tqdm(cd_pairs, desc=f"export {args.mode}"):
            # load teacher vectors
            poi_path = os.path.join(data_path, city_id, "poi_vecs", f"{args.llm_tag}_poi_vec_{date}.npy")
            demo_path = os.path.join(data_path, city_id, "demo_vecs", f"{args.llm_tag}_demo_vec_{date}.npy")

            if (not os.path.exists(poi_path)) or (not os.path.exists(demo_path)):
                continue

            poi_vecs = np.load(poi_path).astype(np.float32)    # (N,34)
            demo_vecs = np.load(demo_path).astype(np.float32)  # (N,97)

            tracts6 = city2tract.get(city_id, None)
            if tracts6 is None:
                continue

            tracts11 = [str(city_id) + str(t) for t in tracts6]
            if len(tracts11) != poi_vecs.shape[0] or len(tracts11) != demo_vecs.shape[0]:
                # skip inconsistent city
                continue

            # one record per tract-day
            for i, tract11 in enumerate(tracts11):
                poi_vec = pad_vec(poi_vecs[i], 34)
                demo_vec = pad_vec(demo_vecs[i], 97)

                # POI sample
                poi_prompt = make_poi_prompt(tract11, date)
                poi_ans = json.dumps({tract11: poi_vec}, ensure_ascii=False)
                f_poi.write(json.dumps({
                    "messages": [
                        {"role": "user", "content": poi_prompt},
                        {"role": "assistant", "content": poi_ans},
                    ],
                    "meta": {"city": city_id, "date": date, "tract": tract11, "task": "poi"}
                }, ensure_ascii=False) + "\n")
                n_poi += 1

                # DEMO sample (conditioned on POI control)
                demo_prompt = make_demo_prompt(tract11, date, poi_vec)
                demo_ans = json.dumps({tract11: demo_vec}, ensure_ascii=False)
                f_demo.write(json.dumps({
                    "messages": [
                        {"role": "user", "content": demo_prompt},
                        {"role": "assistant", "content": demo_ans},
                    ],
                    "meta": {"city": city_id, "date": date, "tract": tract11, "task": "demo"}
                }, ensure_ascii=False) + "\n")
                n_demo += 1

    print(f"✅ Wrote {n_poi} POI samples -> {poi_out}")
    print(f"✅ Wrote {n_demo} DEMO samples -> {demo_out}")


if __name__ == "__main__":
    args = parse_args()
    export_jsonl(args)
