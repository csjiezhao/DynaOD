# DynaOD: Dynamic Origin-Destination Flow Generation with Discrete-to-Continuous Temporal Semantic Modeling

> This repo is the official implementation of our paper published in **IJCAI 2026 Main Track**.

<img src="./figs/dynaod.png">

DynaOD synthesizes dynamic origin-destination (OD) flows from regional attributes and temporal context. The framework uses LLM-derived discrete directional controls, ShapeNet continuous temporal feature evolution, ShapeMem retrieval-based shape priors, and a frozen pretrained OD generator such as WeDAN.

## Repository Layout

- `models/DynaOD/`: DynaOD, ShapeNet, ShapeMem, training, inference, and ablation scripts.
- `models/WeDAN/`: pretrained static OD generator backbone used by DynaOD.
- `llm/`: prompt templates and scripts for generating POI and demographic directional control vectors.
- `llm_distillation/`: scripts for exporting SFT data and visualizing control/shape trajectories.
- `region_emb/`: BGRL-style regional embedding model used for ShapeMem retrieval.
- `configs/`: LLaMA-Factory LoRA training and merge templates for the lightweight Qwen controller.
- `scripts/`: helper scripts such as the vLLM server launcher and safe dataset maintenance utilities.
- `docs/code_inventory.md`: notes on duplicate/legacy scripts and recommended cleanup.

## Data And Checkpoints

Large artifacts are intentionally not tracked by Git:

- `data/`: county-level tract graphs, OD matrices, LLM control vectors, and shape priors.
- `dy_od/`: raw or processed dynamic OD data.
- `ckpts/`: WeDAN, ShapeNet, ShapeMem, BGRL, and scaler checkpoints.
- `sft_data/`: exported POI/DEMO instruction tuning data.

Place these directories at the repository root, or pass explicit paths through the command-line arguments in the scripts.

## Environment

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Create a local `.env` from `.env.example` and fill only the platforms you use:

```bash
cp .env.example .env
```

## Typical Workflow

Generate LLM directional controls:

```bash
python -m llm.poi_vec_generation --platform OpenAI --model gpt-4o-mini --mode train
python -m llm.demo_vec_generation --platform OpenAI --model gpt-4o-mini --mode train
```

Generate control vectors with a distilled local controller:

```bash
python -m llm.poi_vec_sft --mode train
python -m llm.demo_vec_sft --mode train
```

Train ShapeNet with a frozen WeDAN checkpoint:

```bash
python -m models.DynaOD.train_shapenet --llm qwen-2.5-1.5b-sft
```

Build ShapeMem weekday priors:

```bash
python -m models.DynaOD.shape_memory
```

Run DynaOD inference:

```bash
python -m models.DynaOD.run_inference --mode test3 --external_shape
```

Run the classifier-controller ablation:

```bash
python -m models.DynaOD.run_inference_cls --mode test3
```

Export SFT JSONL data for the lightweight controller:

```bash
python -m llm_distillation.ctrl_vec_readout --mode train --llm_tag gpt-4o-mini
```

Launch a local vLLM server for the distilled controller:

```bash
MODEL_PATH=/path/to/qwen-or-merged-model vLLM_API_KEY=local-token scripts/vllm_server.sh
```

## Citation

```bibtex
@inproceedings{zhao2026dynaod,
  title={DynaOD: Dynamic Origin-Destination Flow Generation with Discrete-to-Continuous Temporal Semantic Modeling},
  author={Zhao, Jie and Dai, Xianqi and Feng, Jie and Wang, Huandong and Li, Yong},
  booktitle={Proceedings of the Thirty-Fifth International Joint Conference on Artificial Intelligence},
  year={2026}
}
```
