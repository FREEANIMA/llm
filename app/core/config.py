import torch
from dataclasses import dataclass

@dataclass
class GlobalConfig:
    lr = 1e-4
    batch_size = 32

    dtype = torch.bfloat16
    device = "cuda"

@dataclass
class Config(GlobalConfig):
    hidden_size:int = 1024
    nhaed:int = 16
    num_layers:int = 16
    rope_hz:int = 10_000
    max_len:int = 256

    weight_decay = 0.01
    drop_out = 0.1

    