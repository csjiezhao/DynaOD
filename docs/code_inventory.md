# DynaOD Code Inventory

This note records the current status of duplicate or legacy-looking scripts in
the public DynaOD codebase. It is intended to make the release easier to clean
without losing track of experiment variants.

## LLM Control Vector Scripts

The `0430` variants are almost identical to the non-`0430` SFT scripts. Their
main purpose is to switch from the original online LLM setting to the local
SFT/vLLM setting and to use the January-April 2019 data range.

| Script pair | Main differences |
| --- | --- |
| `llm/demo_vec_sft.py` vs `llm/demo_vec_sft0430.py` | The `0430` script imports `load_samples_ijcai`, defaults to `--platform vLLM`, defaults to `--model qwen-2.5-1.5b-sft`, and calls `load_samples_ijcai(...)` instead of `load_samples(...)`. The prompt and generation logic are otherwise the same. |
| `llm/poi_vec_sft.py` vs `llm/poi_vec_sft_0430.py` | The `0430` script has the same changes: `load_samples_ijcai`, `vLLM`, `qwen-2.5-1.5b-sft`, and January-April samples. The POI control generation logic is otherwise the same. |

Recommended cleanup:

- Merge each pair into one script with a `--date_range` or `--split_profile`
  argument, for example `jan2019` and `jan_apr_2019`.
- Keep `qwen-2.5-1.5b-sft` plus `vLLM` as the default only if the public release
  should represent the distilled/SFT pipeline.
- Remove the inconsistent filename style: `demo_vec_sft0430.py` lacks the
  underscore used by `poi_vec_sft_0430.py`.

## ShapeNet Training Scripts

`models/DynaOD/train_0430.py` is a variant of `models/DynaOD/train_shapenet.py`.
Most of the training loop is duplicated. The meaningful differences are:

- `train_0430.py` imports and uses `load_window_samples_ijcai(...)` for training.
- `train_shapenet.py` uses `load_window_samples(...)`, which covers January 2019.
- `train_0430.py` saves the best checkpoint to `ckpts/shapenet_model_0430.pth`.
- `train_shapenet.py` saves the best checkpoint to
  `ckpts/shapenet_model_{llm}.pth`.
- `train_0430.py` sets `test_process(..., return_by_day=False)` by default,
  while `train_shapenet.py` defaults to `return_by_day=True`.

Important inconsistency:

- `train_0430.py` trains with `load_window_samples_ijcai(...)`, but its validation
  data still uses `load_window_samples(...)`, so training uses January-April 2019
  while validation uses the January 2019 split.
- Both training scripts pass `llm=model_config["llm"]` for training data, but the
  validation loaders omit `llm`, so validation falls back to the default
  `gpt-4o-mini` control vectors. This is likely wrong when training with
  `qwen-2.5-1.5b-sft`.

Recommended cleanup:

- Replace `train_0430.py` with one unified `train_shapenet.py` entry point that
  accepts `--split_profile jan2019|jan_apr_2019`, `--llm`, and `--ckpt_name`.
- Make training and validation use the same split profile and the same `llm`
  value unless an ablation explicitly requests otherwise.
- Keep `train_0430.py` only as a deprecated wrapper if older run scripts depend
  on that filename.

## Dataset Loading Variants

`models/DynaOD/data_load.py` contains both January-only and January-April data
loaders:

- `load_samples(...)`: city/date samples for January 1-31, 2019.
- `load_samples_ijcai(...)`: city/date samples for January 1-April 30, 2019.
- `load_window_samples(...)`: windowed samples for January 2019.
- `load_window_samples_ijcai(...)`: windowed samples for January-April 2019.
- `load_city_data(...)`: loads control vectors and optional ShapeMem priors
  (`poi_shp`, `demo_shp`) when those files exist.
- `load_city_data0430(...)`: loads control vectors but does not load ShapeMem
  prior tensors.

Potential issue:

- `load_samples_ijcai(..., is_simplify=False)` currently hardcodes
  `llm='gpt-4o-mini'` and calls `load_city_data(...)`, not `load_city_data0430`.
  This path is not used by the current `0430` SFT scripts because they call it
  with `is_simplify=True`, but it is still confusing and should be unified.

## `dset_move.py`

`models/DynaOD/dset_move.py` is not required by the training or inference
pipeline.

Current status:

- No other file imports or calls it.
- The first half is commented-out legacy code for building split directories.
- The active function reads `cities_split.json` and deletes numeric city
  directories under `data/` that are not in the split.
- Because it calls `shutil.rmtree(...)`, it is a destructive maintenance helper,
  not a required dataset component.

Recommended cleanup:

- Remove it from `models/DynaOD/`, or move it to a clearly named maintenance
  script such as `scripts/remove_orphan_city_dirs.py`.
- If kept, add `--dry-run` as the default and require an explicit `--apply` flag
  before deleting anything.

## Release Recommendation

For a clean public release, the preferred structure is:

- One POI SFT vector generation script.
- One demographic SFT vector generation script.
- One ShapeNet training script with explicit split/model arguments.
- One inference path for the final model.
- Deprecated wrappers only where they protect reproducibility of old commands.
- Destructive data maintenance scripts outside `models/` and dry-run by default.
