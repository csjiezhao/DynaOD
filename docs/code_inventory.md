# DynaOD Code Inventory

This note records how the public DynaOD codebase was cleaned from the original
experiment snapshot. It is intended to keep the final release easy to navigate
without losing track of experiment variants.

## LLM Control Vector Scripts

The former `0430` variants were almost identical to the non-`0430` SFT scripts.
Their main purpose was to switch from the original online LLM setting to the
local SFT/vLLM setting and to use the January-April 2019 data range.

The cleaned release keeps one POI SFT script and one demographic SFT script:

- `llm/poi_vec_sft.py`
- `llm/demo_vec_sft.py`

Both scripts now expose `--split_profile jan2019|jan_apr_2019`, default to the
distilled `qwen-2.5-1.5b-sft` controller through `vLLM`, and replace the deleted
`*_0430.py` entry points.

## ShapeNet Training Scripts

The former `models/DynaOD/train_0430.py` duplicated
`models/DynaOD/train_shapenet.py`. It existed mainly to train on the
January-April 2019 window, but it mixed January-April training with January-only
validation and omitted the validation `llm` argument.

The cleaned release removes `train_0430.py` and uses one ShapeNet training entry
point:

- `models/DynaOD/train_shapenet.py`

This script now accepts `--split_profile jan2019|jan_apr_2019`, `--llm`, and
`--shape_ckpt_name`. Training and validation use the same split profile and the
same `llm` value.

## Dataset Loading Variants

`models/DynaOD/data_load.py` contains both January-only and January-April data
loaders:

- `load_samples(...)`: city/date samples for January 1-31, 2019.
- `load_samples_ijcai(...)`: city/date samples for January 1-April 30, 2019.
- `load_window_samples(...)`: windowed samples for January 2019.
- `load_window_samples_ijcai(...)`: windowed samples for January-April 2019.
- `load_city_data(...)`: loads control vectors and optional ShapeMem priors
  (`poi_shp`, `demo_shp`) when those files exist.
- `load_city_data_no_shape(...)`: loads control vectors but does not load ShapeMem
  prior tensors.

The non-window loaders now also accept an `llm` argument instead of hardcoding
`gpt-4o-mini`.

## Dataset Maintenance

The former `models/DynaOD/dset_move.py` was not required by the training or
inference pipeline.

It has been replaced by:

- `scripts/remove_orphan_city_dirs.py`

The new script is dry-run by default and requires `--apply` before deleting any
directory.

## Release Recommendation

The cleaned public release now follows this structure:

- One POI SFT vector generation script.
- One demographic SFT vector generation script.
- One ShapeNet training script with explicit split/model arguments.
- One inference path for the final model.
- Destructive data maintenance scripts outside `models/` and dry-run by default.
