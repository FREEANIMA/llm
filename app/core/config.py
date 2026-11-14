import torch
from dataclasses import dataclass

@dataclass
class GlobalConfig:
    lr: float = 1e-4
    batch_size: int = 64
    dtype: torch.dtype = torch.bfloat16
    use_amp:bool = True
    device: str = "cuda"

@dataclass
class Config(GlobalConfig):
    hidden_size: int = 1024
    nhead: int = 16
    num_layers: int = 16
    rope_hz: int = 10_000
    max_len: int = 256

    weight_decay: float = 0.01
    drop_out: float = 0.1