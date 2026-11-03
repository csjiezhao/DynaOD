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

## Data And Checkpoints

Large artifacts are intentionally not tracked by Git. The paper reproduction
bundle is available from [Google Drive](https://drive.google.com/file/d/1sXXbiJguqqUoH8XW2B9qBsDA0blABPAI/view?usp=drive_link):

[`DynaOD_paper_runtime_assets.tar.zst`](https://drive.google.com/file/d/1sXXbiJguqqUoH8XW2B9qBsDA0blABPAI/view?usp=drive_link) (831 MB)

Run the following command from the repository root after downloading the file.
It extracts the three required paths directly into the expected locations:

```bash
zstd -dc DynaOD_paper_runtime_assets.tar.zst | tar -xf -
```

| Path | Contents | Required by |
| --- | --- | --- |
| `county2tract.json` | County-to-tract mapping for all 500 counties. | Control-vector generation and ShapeMem retrieval. |
| `data/` | January 2019 data for 500 counties: static POI/DEMO features, distance and adjacency matrices, daily OD matrices, GPT-4o-mini and Qwen-SFT control vectors, plus external shape priors. | DynaOD/ShapeNet training and inference. |
| `ckpts/` | City split, scalers, WeDAN, final SFT ShapeNet, ShapeMem, regional embeddings, and BGRL checkpoints. | Training, inference, and external-shape generation. |

The released `data/` contains 15,500 daily OD matrices (500 counties x 31
days) and two January control-vector sources: `gpt-4o-mini` and
`qwen-2.5-1.5b-sft`. Shape priors are included for January 1-21 and January
25-31, matching the 7-day windows used by the code. The final SFT ShapeNet
weight is available under both `shapenet_model_qwen-2.5-1.5b-sft.pth` and
`shapenet_model_best.pth`; the latter matches the default inference path.

The following large artifacts are not included in the paper reproduction bundle:

- `dy_od/`: raw dynamic OD source data, not required by the released pipeline.
- `sft_data/`: POI/DEMO instruction-tuning JSONL data.
- The Qwen LoRA adapter and merged serving model.

The archive checksum is:

```text
24ccd9a10d6473a5dcd51dc3ac79fd1ce5b10b574be451be7f0742c93b21a161
```

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
