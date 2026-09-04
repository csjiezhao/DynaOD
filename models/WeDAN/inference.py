from model import Diffusion
from data_load import prepare_test_data, MyDataset, MyBatchSampler, collate_fn
from utils.MyLogger import Logger
from eval import test

import os
import torch
import pickle
from setproctitle import setproctitle
from dgl.dataloading import GraphDataLoader


config = {
    "device": "cuda:2",
    "exp_path": "exp/",
    "data_path" : "data/",
    "shuffle_cities": 0,

    "attr_MinMax": 1,
    "od_MinMax": 1,
    "skew_norm": "log",

    "pert_node": 1,
    "sample_method": "DDIM",

    "train_set": 0.8,
    "valid_set": 0.1,
    "test_set": 0.1,

    "valid_period": 100,
    "overfit_tolerance": 200,

    "hiddim": 32,
    "num_head": 4,
    "num_head_cross": 1,
    "num_layer": 4,
    "dropout": 0,
    "n_indim": 131,
    "e_indim": 2,
    "n_outdim": 131,
    "e_outdim": 1,

    "T": 1000,
    "DDIM_T_sample": 100,
    "sample_times": 10,
    "DDIM_eta": 0,
    "beta_scheduler": "cosine",

    "max_nodes": 1500,
    "city_type_limit": "",
    "LaPE_dim": 0,
    "batch_size": 32,
    "norm_type": "layer",
    "learning_rate": 1e-3,
    "optm": "AdamW",
    "loss": "mse",
}

setproctitle("WEDANInference")

scaler_path = os.path.join( "ckpts/train_scalers.pkl")
with open(scaler_path, "rb") as f:
    train_scalers = pickle.load(f)
testSet, scalers = prepare_test_data(config, external_scalers=train_scalers)

# logger
print("  ** preparing logger...", end="")
logger = Logger(config)
print("done")

print("  ** loading trained model...", end="")
diff_model = Diffusion(config).to(config["device"])
diff_model.load_state_dict(torch.load(os.path.join("ckpts/wedan_model_8400.pth")))
print("done")

print("  ** constructing dataloader...", end="")
test_set = MyDataset(testSet, config, "test")
ttBS = MyBatchSampler(test_set, config["batch_size"], config["max_nodes"])
test_Dloader = GraphDataLoader(test_set, batch_sampler=ttBS, collate_fn=collate_fn)
print("done")

test(config, diff_model, test_Dloader, logger, scalers, config["device"])
