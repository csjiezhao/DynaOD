from models.DynaOD.model import DynaOD
from models.DynaOD.data_load import (
    load_window_samples,
    CityWindowDataset,
    MyBatchSampler,
    collate_fn_window,
)
from models.DynaOD.train_shapenet import test_process

from setproctitle import setproctitle
from dgl.dataloading import GraphDataLoader
import torch
import argparse


def parse_args():
    p = argparse.ArgumentParser("DynaOD Inference")
    p.add_argument("--mode", type=str, default="test3", choices=["test1", "test2", "test3", "train"],
                   help="dataset split mode")
    p.add_argument("--external_shape", action="store_true",
                   help="use external shape (load shape_ckpt) instead of internal ShapeNet output")
    p.add_argument("--device", type=str, default="cuda:2")
    p.add_argument("--data_path", type=str, default="data/")
    p.add_argument("--ckpt_path", type=str, default="ckpts/")
    p.add_argument("--odnet_ckpt", type=str, default="ckpts/wedan_model_8400.pth")
    p.add_argument("--shape_ckpt", type=str, default="ckpts/shapenet_model_best.pth")
    p.add_argument("--llm", type=str, default="qwen-2.5-1.5b-sft")

    # optional inference knobs
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_nodes", type=int, default=1000)
    p.add_argument("--ddim_t_sample", type=int, default=10)
    p.add_argument("--sample_times", type=int, default=5)

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    setproctitle(f"DynaOD-Inference@{args.mode}")

    model_config = {
        "device": args.device,
        "data_path": args.data_path,
        "ckpt_path": args.ckpt_path,
        "odnet_ckpt": args.odnet_ckpt,
        "shape_ckpt": args.shape_ckpt,
        "split_ratio": 0.7,
        "llm": args.llm,

        # parameters for ShapeNet (not used in inference but kept for compatibility)
        "shapenet_lr": 5e-4,
        "shapenet_epoch": 20,

        # parameters for WeDAN
        "batch_size": args.batch_size,
        "max_nodes": args.max_nodes,
        "DDIM_T_sample": args.ddim_t_sample,
        "sample_times": args.sample_times,

        "attr_MinMax": 1,
        "od_MinMax": 1,
        "skew_norm": "log",
        "pert_node": 1,
        "sample_method": "DDIM",
        "valid_period": 10,
        "overfit_tolerance": 20,
        "hiddim": 32,
        "num_head": 4,
        "num_head_cross": 1,
        "num_layer": 4,
        "dropout": 0,
        "n_indim": 131,
        "e_indim": 2,
        "n_outdim": 131,
        "e_outdim": 1,
        "p_generation": 1,
        "p_featMissing": 0,
        "T": 1000,
        "DDIM_eta": 0,
        "beta_scheduler": "cosine",
        "EPOCH": 20000,
        "city_type_limit": "",
        "LaPE_dim": 0,
        "norm_type": "layer",
        "learning_rate": 1e-3,
        "optm": "AdamW",
        "loss": "mse",
    }

    # ---- load dataset ----
    test_data = load_window_samples(
        data_path=model_config["data_path"],
        shuffle_cities=True,
        split_ratio=model_config["split_ratio"],
        mode=args.mode,
        llm=args.llm,
    )
    test_set = CityWindowDataset(*test_data)
    sampler = MyBatchSampler(test_set, model_config["batch_size"], model_config["max_nodes"])
    dataloader = GraphDataLoader(test_set, batch_sampler=sampler, collate_fn=collate_fn_window)

    # ---- load model ----
    dyna_model = DynaOD(diff_config=model_config, pretrained_ckpt=model_config["odnet_ckpt"]).to(model_config["device"])
    dyna_model.load_state_dict(torch.load(model_config["shape_ckpt"], map_location="cpu"), strict=False)
    dyna_model.eval()

    print(f"Beginning inference... mode={args.mode} external_shape={args.external_shape}")
    avg_all, avg_by_day = test_process(model_config, dataloader, dyna_model, poi_control=True, demo_control=True,
                       external_shape=args.external_shape, return_by_day=True)

    print("ALL:", avg_all)
    for t, m in enumerate(avg_by_day):
        print(f"Day {t}:", m)
