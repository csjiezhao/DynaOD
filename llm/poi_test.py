#!/usr/bin/env python3
import json
import time
import numpy as np
import tiktoken

from json_repair import repair_json
from llm.llm_api import LLMCaller
from llm.prompts import POIS_CATE, POI_CTRL_PROMPT


# ===== 固定参数（无需输入） =====

PLATFORM = "OpenAI"
MODEL = "gpt-4o-mini"

TRACT = "01001020100"
DATE = "2019-01-07"

RUNS = 5


def main():

    encoder = tiktoken.encoding_for_model(MODEL)
    llm = LLMCaller(PLATFORM, MODEL)

    latencies = []
    input_tokens_list = []
    output_tokens_list = []

    print("\n=== POI Control Demo (5 runs) ===")
    print(f"Model: {MODEL}")
    print(f"Tract: {TRACT}")
    print(f"Date:  {DATE}")
    print()

    prompt_text = POI_CTRL_PROMPT.format(
        TRACT=TRACT,
        POIS_CATE=POIS_CATE,
        DATE=DATE,
    )

    messages = [
        {"role": "user", "content": prompt_text}
    ]

    # input tokens (固定，不用重复算)
    input_tokens = 0
    for msg in messages:
        input_tokens += len(encoder.encode(msg["role"]))
        input_tokens += len(encoder.encode(msg["content"]))

    for i in range(RUNS):

        t0 = time.perf_counter()

        raw_resp = llm.get_response(messages)

        latency = time.perf_counter() - t0

        output_tokens = len(encoder.encode(raw_resp))

        latencies.append(latency)
        input_tokens_list.append(input_tokens)
        output_tokens_list.append(output_tokens)

        print(
            f"Run {i+1}: "
            f"time={latency:.3f}s | "
            f"in={input_tokens} | "
            f"out={output_tokens}"
        )

        # optional: verify parse once
        if i == 0:
            try:
                resp_json = json.loads(repair_json(raw_resp))
                vec = resp_json.get(TRACT, [])
                print(f"Vector length: {len(vec)}")
            except Exception:
                print("Parse warning")

    print("\n===== Average Results =====")

    print(f"Avg latency: {np.mean(latencies):.3f}s")
    print(f"Std latency: {np.std(latencies):.3f}s")

    print(f"Avg input tokens: {int(np.mean(input_tokens_list))}")
    print(f"Avg output tokens: {int(np.mean(output_tokens_list))}")
    print(
        f"Avg total tokens: "
        f"{int(np.mean(input_tokens_list) + np.mean(output_tokens_list))}"
    )


if __name__ == "__main__":
    main()