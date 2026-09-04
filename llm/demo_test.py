#!/usr/bin/env python3
import json
import time
import numpy as np
import tiktoken
import os

from json_repair import repair_json
from llm.llm_api import LLMCaller
from llm.prompts import POIS_CATE, DEMOS_CATE, DEMO_CTRL_PROMPT


# ===== 固定参数 =====

PLATFORM = "OpenAI"
MODEL = "gpt-4o-mini"

DATA_PATH = "data/"
CITY_ID = "10005"
TRACT = "10005050101"
DATE = "2019-01-07"

RUNS = 5


def main():

    encoder = tiktoken.encoding_for_model(MODEL)
    llm = LLMCaller(PLATFORM, MODEL)

    # 读取一个 tract 的 poi control 向量
    poi_vec_path = os.path.join(
        DATA_PATH,
        CITY_ID,
        "poi_vecs",
        f"gpt-4o-mini_poi_vec_2019_01_07.npy",
    )

    if not os.path.exists(poi_vec_path):
        print("❌ poi_vec file not found")
        return

    poi_vecs = np.load(poi_vec_path)

    # 只取第一个 tract 的向量
    poi_ctrl = poi_vecs[0].tolist()

    prompt = DEMO_CTRL_PROMPT.format(
        TRACT=TRACT,
        POIS_CATE=POIS_CATE,
        POIS_CTRL_VEC=poi_ctrl,
        DEMOS_CATE=DEMOS_CATE,
        DATE=DATE,
    )

    messages = [
        {"role": "user", "content": prompt}
    ]

    # 计算 input tokens（固定）
    input_tokens = 0
    for msg in messages:
        input_tokens += len(encoder.encode(msg["role"]))
        input_tokens += len(encoder.encode(msg["content"]))

    latencies = []
    outputs = []

    print("\n=== DEMO control single-tract test ===")
    print("Model:", MODEL)
    print("Tract:", TRACT)
    print("Date :", DATE)
    print()

    for i in range(RUNS):

        t0 = time.perf_counter()

        raw = llm.get_response(messages)

        latency = time.perf_counter() - t0

        output_tokens = len(encoder.encode(raw))

        latencies.append(latency)
        outputs.append(output_tokens)

        print(
            f"Run {i+1}: "
            f"time={latency:.3f}s | "
            f"in={input_tokens} | "
            f"out={output_tokens}"
        )

        # 第一次检查 parse
        if i == 0:
            try:
                resp = json.loads(repair_json(raw))
                vec = resp.get(TRACT, [])
                vec = (vec[:97] + [0] * (97 - len(vec)))[:97]
                print("Vector length:", len(vec))
            except Exception:
                print("⚠️ parse warning")

    print("\n===== Average =====")

    print("Avg latency:",
          f"{np.mean(latencies):.3f}s")

    print("Std latency:",
          f"{np.std(latencies):.3f}s")

    print("Avg input tokens:",
          input_tokens)

    print("Avg output tokens:",
          int(np.mean(outputs)))

    print("Avg total tokens:",
          int(input_tokens + np.mean(outputs)))


if __name__ == "__main__":
    main()