import torch

QTR_BODY_ATT_ARGS = {
    "embed_dim": 1024,
    "hidden_size": 512,
    "output_dim": 512,
    "n_heads": 8,
    "num_layers": 4,
    "order_sensitive": True,
    "dropout": 0.1,
    "train_mode": "normal"
}

CONV_ATT_ARGS = {
    "query_dim": 1024,
    "dim": 512,
    "num_layers": 4,
    "num_heads": 8
}

QTR_HEAD_ARGS = {
    "in_dim": 64
}

ADAPTER_ARGS = {
    "input_dim": 1024,
    "bottleneck": 128,
    "output_dim": 64,
    "dropout": 0.1,
    "init_scale": 1.0
}

QTR_BODY_CHECKPOINT = "./checkpoints/qtr_body/qtr_body.pt"
QTR_HEAD_CHECKPOINT = "./checkpoints/qtr_head/qtr_head.pt"
CONV_CHECKPOINT = "./checkpoints/conv/conv.pt"
SCORER_CHECKPOINT = "./checkpoints/scorer/scorer.pt"

QTR_PRETRAIN_DATALOADER_BUILDING_ARGS = {
    "k_neg": 10,
    "p_hard": 0.2,
    "batch_size": 32,
    "test_ratio": 0.2,
    "seed": None,
}

QTR_PRETRAIN_OPTIMIZER_ARGS = {
    "lr": 1e-5,
    "weight_decay": 1e-3,
}


SCORER_ARGS = {
    "input_size": 512,
    "hidden_size": 512,
    "output_size": 1
}

SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("current device:", DEVICE)
