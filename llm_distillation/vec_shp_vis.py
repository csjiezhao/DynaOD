# -*- coding: utf-8 -*-
"""
Publication-quality visualization for DynaOD (fixed version)
- Read files using your original date format: YYYY_%m_%d
- Display x-axis as date strings (YYYY-MM-DD)
- Bigger fonts / ticks / titles
- Export PNG + PDF
- Separate output folders per LLM
- Use your split logic (seen/unseen cities + window) via load_persisted_cities + split_into_windows
"""

from models.DynaOD.data_load import load_persisted_cities, split_into_windows, weighted_shape_average

from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# =========================
# Global style (paper-level)
# =========================
plt.rcParams.update({
    "font.size": 30,
    "axes.titlesize": 30,
    "axes.labelsize": 30,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    "legend.fontsize": 30,
    "figure.titlesize": 30,
})

# =========================
# Utils
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_figure(fig, path_base: str, dpi=250, save_png=True, save_pdf=True):
    if save_png:
        fig.savefig(path_base + ".png", dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(path_base + ".pdf", bbox_inches="tight")

def normalize_TD(x_TD: np.ndarray, mode=None, eps=1e-6) -> np.ndarray:
    if mode is None:
        return x_TD
    if mode == "zscore":
        mu = x_TD.mean(axis=0, keepdims=True)
        sd = x_TD.std(axis=0, keepdims=True)
        return (x_TD - mu) / (sd + eps)
    if mode == "minmax":
        mn = x_TD.min(axis=0, keepdims=True)
        mx = x_TD.max(axis=0, keepdims=True)
        return (x_TD - mn) / (mx - mn + eps)
    raise ValueError(f"Unknown normalize mode: {mode}")

def date_display(date_ymd_underscore: str) -> str:
    # "2019_01_25" -> "2019-01-25"
    return date_ymd_underscore.replace("_", "-")

# =========================
# Curve plots (2x2)
# =========================
def draw_step_jitter(ax, x_TD: np.ndarray, jitter: float, title: str):
    T, D = x_TD.shape
    for d in range(D):
        ax.step(np.arange(T), x_TD[:, d] + d * jitter, where="post", linewidth=1.6, alpha=0.95)
    ax.set_title(title)
    ax.set_ylabel("value (+ jitter)")
    ax.set_yticks([])
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

def draw_lines(ax, x_TD: np.ndarray, title: str):
    T, D = x_TD.shape
    for d in range(D):
        ax.plot(np.arange(T), x_TD[:, d], linewidth=1.6, alpha=0.9)
    ax.set_title(title)
    ax.set_ylabel("value")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

def plot_region_curves_4panels(
    poi_vec_TD: np.ndarray,
    poi_shp_TD: np.ndarray,
    demo_vec_TD: np.ndarray,
    demo_shp_TD: np.ndarray,
    title: str,
    save_base: str,
    x_labels: list[str],
    poi_jitter: float = 0.04,
    demo_jitter: float = 0.02,
    normalize_shp=None,
):
    # normalize only shp
    poi_shp_TD = normalize_TD(poi_shp_TD, normalize_shp)
    demo_shp_TD = normalize_TD(demo_shp_TD, normalize_shp)

    T = poi_vec_TD.shape[0]
    t = np.arange(T)

    fig, axes = plt.subplots(2, 2, figsize=(20, 11), sharex=True)

    draw_step_jitter(axes[0, 0], poi_vec_TD, poi_jitter, "POI control")
    draw_lines(axes[0, 1], poi_shp_TD, "POI shape" + (f" | norm={normalize_shp}" if normalize_shp else ""))

    draw_step_jitter(axes[1, 0], demo_vec_TD, demo_jitter, "DEMO control")
    draw_lines(axes[1, 1], demo_shp_TD, "DEMO shape" + (f" | norm={normalize_shp}" if normalize_shp else ""))

    # x-axis: use dates
    for ax in axes[1, :]:
        ax.set_xlabel("Date")
        ax.set_xticks(t)
        ax.set_xticklabels(x_labels, rotation=30, ha="right")

    for ax in axes[0, :]:
        ax.set_xticks(t)
        ax.set_xticklabels([])

    fig.suptitle(title, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, save_base)
    plt.close(fig)

# =========================
# Heatmaps (2x2) + two colorbars at right
# =========================
def plot_region_heatmaps_4panels_two_cbars(
    poi_vec_TD: np.ndarray,
    poi_shp_TD: np.ndarray,
    demo_vec_TD: np.ndarray,
    demo_shp_TD: np.ndarray,
    title: str,
    save_base: str,
    x_labels: list[str],
    normalize_shp=None,
):
    # normalize only shp
    poi_shp_TD = normalize_TD(poi_shp_TD, normalize_shp)
    demo_shp_TD = normalize_TD(demo_shp_TD, normalize_shp)

    T = poi_vec_TD.shape[0]
    t = np.arange(T)

    # unified vec color range
    vec_all = np.concatenate([poi_vec_TD.reshape(-1), demo_vec_TD.reshape(-1)], axis=0)
    vec_vmin, vec_vmax = float(vec_all.min()), float(vec_all.max())

    # unified shp color range (after norm)
    shp_all = np.concatenate([poi_shp_TD.reshape(-1), demo_shp_TD.reshape(-1)], axis=0)
    shp_vmin, shp_vmax = float(shp_all.min()), float(shp_all.max())

    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(
        nrows=2, ncols=3,
        width_ratios=[1, 1, 0.06],
        wspace=0.25, hspace=0.25
    )

    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])

    cax_vec = fig.add_subplot(gs[0, 2])
    cax_shp = fig.add_subplot(gs[1, 2])

    im00 = ax00.imshow(poi_vec_TD.T, aspect="auto", origin="lower", vmin=vec_vmin, vmax=vec_vmax)
    ax00.set_title("POI control (raw)")

    im01 = ax01.imshow(poi_shp_TD.T, aspect="auto", origin="lower", vmin=shp_vmin, vmax=shp_vmax)
    ax01.set_title("POI shape" + (f" | norm={normalize_shp}" if normalize_shp else ""))

    im10 = ax10.imshow(demo_vec_TD.T, aspect="auto", origin="lower", vmin=vec_vmin, vmax=vec_vmax)
    ax10.set_title("DEMO control (raw)")

    im11 = ax11.imshow(demo_shp_TD.T, aspect="auto", origin="lower", vmin=shp_vmin, vmax=shp_vmax)
    ax11.set_title("DEMO shape" + (f" | norm={normalize_shp}" if normalize_shp else ""))

    for ax in [ax00, ax01, ax10, ax11]:
        ax.set_xlabel("Date")
        ax.set_ylabel("Dimension")
        ax.set_yticks([])  # too dense otherwise
        ax.set_xticks(t)
        ax.set_xticklabels(x_labels, rotation=30, ha="right")

    cb1 = fig.colorbar(im00, cax=cax_vec)
    cb1.set_label("control value", fontsize=14)

    cb2 = fig.colorbar(im01, cax=cax_shp)
    cb2.set_label("shape value", fontsize=14)

    fig.suptitle(title, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, save_base)
    plt.close(fig)

# =========================
# Data loading
# =========================
def load_city_window_arrays(city, date_list_underscore, data_path="data", llm="gpt-4o-mini"):
    """
    date_list_underscore: list[str] like ["2019_01_25", ...]
    Returns:
      poi_vecs: (N,T,Dp)
      poi_shps: (N,T,Dp)
      demo_vecs: (N,T,Dd)
      demo_shps: (N,T,Dd)
    """
    poi_vecs_list, demo_vecs_list, poi_shps_list, demo_shps_list = [], [], [], []

    score_path = os.path.join(data_path, city, "near_k_scores.npy")
    if not os.path.exists(score_path):
        raise FileNotFoundError(f"Missing near_k_scores.npy for city={city}: {score_path}")
    k_scores = np.load(score_path).astype(np.float32)

    for d in date_list_underscore:
        poi_path = os.path.join(data_path, city, "poi_vecs", f"{llm}_poi_vec_{d}.npy")
        demo_path = os.path.join(data_path, city, "demo_vecs", f"{llm}_demo_vec_{d}.npy")
        poi_shp_path = os.path.join(data_path, city, "poi_shps", f"poi_shp_{d}.npy")
        demo_shp_path = os.path.join(data_path, city, "demo_shps", f"demo_shp_{d}.npy")

        # if any missing, skip this date (robust)
        if not (os.path.exists(poi_path) and os.path.exists(demo_path) and os.path.exists(poi_shp_path) and os.path.exists(demo_shp_path)):
            print(f"[WARN] missing files for city={city}, llm={llm}, date={d}, skip date.")
            continue

        poi_vec = np.load(poi_path).astype(np.float32)   # (N,34)
        demo_vec = np.load(demo_path).astype(np.float32) # (N,97)
        poi_shp = np.load(poi_shp_path).astype(np.float32)   # (N,K,34)
        demo_shp = np.load(demo_shp_path).astype(np.float32) # (N,K,97)

        # weighted average over K neighbors (your original logic)
        poi_shp, demo_shp, _ = weighted_shape_average(poi_shp, demo_shp, k_scores)

        poi_vecs_list.append(poi_vec)
        demo_vecs_list.append(demo_vec)
        poi_shps_list.append(poi_shp)
        demo_shps_list.append(demo_shp)

    if len(poi_vecs_list) == 0:
        raise RuntimeError(f"No valid dates loaded for city={city}, llm={llm}. Check file names and date format.")

    poi_vecs = np.stack(poi_vecs_list, axis=1)   # (N,T,D)
    demo_vecs = np.stack(demo_vecs_list, axis=1)
    poi_shps = np.stack(poi_shps_list, axis=1)
    demo_shps = np.stack(demo_shps_list, axis=1)

    return poi_vecs, poi_shps, demo_vecs, demo_shps, date_list_underscore[:poi_vecs.shape[1]]

# =========================
# Main driver
# =========================
def plot_one_city_multiple_regions(
    mode: str,
    split_ratio: float = 0.7,
    T: int = 7,
    data_path: str = "data",
    llm: str = "gpt-4o-mini",
    out_root: str = "vis_out",
    city_name=None,
    win_id: int = 0,
    region_ids=None,
    max_regions: int = 5,
    plot_curves: bool = True,
    plot_heatmaps: bool = True,
    normalize_shp="minmax",
):
    _, seen_cities, unseen_cities = load_persisted_cities("ckpts/")

    dates = pd.date_range(start="2019-01-01", end="2019-01-31").strftime("%Y_%m_%d").tolist()
    time_split_point = int(len(dates) * split_ratio)
    seen_dates = dates[:time_split_point]
    unseen_dates = dates[time_split_point:][-7:]

    seen_windows = split_into_windows(seen_dates, T=T)
    unseen_windows = split_into_windows(unseen_dates, T=T)

    sample_settings = {
        "train": (seen_cities, seen_windows),
        "test1": (seen_cities, unseen_windows),
        "test2": (unseen_cities, seen_windows),
        "test3": (unseen_cities, unseen_windows),
        "test": (unseen_cities, unseen_windows),  # allow your new single-test naming if desired
    }
    sample_cities, sample_windows = sample_settings.get(mode, ([], []))
    if len(sample_cities) == 0 or len(sample_windows) == 0:
        raise ValueError(f"Empty cities/windows for mode={mode}.")

    city = city_name if city_name is not None else sample_cities[0]
    date_list = sample_windows[win_id]

    # load arrays
    poi_vecs, poi_shps, demo_vecs, demo_shps, loaded_dates = load_city_window_arrays(
        city, date_list, data_path=data_path, llm=llm
    )
    x_labels = [date_display(d) for d in loaded_dates]

    N = poi_vecs.shape[0]
    if region_ids is None:
        region_ids = list(range(min(max_regions, N)))
    else:
        region_ids = [int(r) for r in region_ids if 0 <= int(r) < N]
        if len(region_ids) == 0:
            raise ValueError("Provided region_ids are empty after filtering by range.")

    # separate folder per LLM
    base_out = os.path.join(out_root, city, mode, f"win{win_id}", llm)
    ensure_dir(base_out)

    for rid in tqdm(region_ids, desc=f"Plotting regions for {city} ({llm})"):
        poi_vec_TD = poi_vecs[rid]
        poi_shp_TD = poi_shps[rid]
        demo_vec_TD = demo_vecs[rid]
        demo_shp_TD = demo_shps[rid]

        out_dir = os.path.join(base_out, f"rid{rid}")
        ensure_dir(out_dir)

        prefix = f"{city}_{mode}_win{win_id}_rid{rid}_{loaded_dates[0]}__{loaded_dates[-1]}"
        title = f"{city} | {llm} | Region {rid} | {x_labels[0]}–{x_labels[-1]}"

        if plot_curves:
            save_base = os.path.join(out_dir, f"{prefix}_curves_4panels_shpnorm-{normalize_shp}")
            plot_region_curves_4panels(
                poi_vec_TD, poi_shp_TD, demo_vec_TD, demo_shp_TD,
                title=title + " | Curves",
                save_base=save_base,
                x_labels=x_labels,
                normalize_shp=normalize_shp,
            )

        if plot_heatmaps:
            save_base = os.path.join(out_dir, f"{prefix}_heatmaps_4panels_shpnorm-{normalize_shp}")
            plot_region_heatmaps_4panels_two_cbars(
                poi_vec_TD, poi_shp_TD, demo_vec_TD, demo_shp_TD,
                title=title + " | Heatmaps",
                save_base=save_base,
                x_labels=x_labels,
                normalize_shp=normalize_shp,
            )

    print("[Done] Output root:", base_out)


if __name__ == "__main__":
    # run multiple models, saved separately
    for llm_name in ["gpt-4o-mini", "qwen-2.5-1.5b-sft"]:
        plot_one_city_multiple_regions(
            mode="test3",        # or "test" if you want single test name
            city_name=None,
            win_id=0,
            region_ids=None,
            max_regions=1000,      # adjust
            plot_curves=True,
            plot_heatmaps=True,
            normalize_shp="minmax",
            T=7,
            data_path="data",
            llm=llm_name,
            out_root="vis_out",
        )
