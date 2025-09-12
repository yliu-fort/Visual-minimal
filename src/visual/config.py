from __future__ import annotations
from dataclasses import dataclass, is_dataclass, asdict
from typing import Literal, Tuple, Optional, Dict, Any
import yaml

@dataclass
class CudnnCfg:
    benchmark: bool
    deterministic: bool

@dataclass
class RunCfg:
    experiment_name: str
    output_dir: str
    seed: int
    device: Literal["auto", "cpu", "cuda"]
    cudnn: CudnnCfg

@dataclass
class DataCfg:
    name: Literal["cifar10", "mnist", "riichi"]
    num_samples: int
    batch_size: int
    num_workers: int
    cfg: Dict[str, Any]

@dataclass
class ModelCfg:
    input_dim: int = 2
    pretrained: bool = False
    name: str = "resnet18"
    common: Optional[Dict[str, Any]] = None

@dataclass
class TrainCfg:
    epochs: int
    lr: float
    weight_decay: float
    grad_clip: float
    log_interval: int
    ckpt_interval: int
    sample_interval: int
    sample_size: int
    checkpoint_dir: str
    save_every_steps: int
    keep_last_k: int
    resume_from: str
    save_best_on: str
    save_rng_state: bool
    warmup: int

@dataclass
class MlflowCfg:
    enabled: bool
    tracking_uri: str
    experiment_name: str

@dataclass
class TrackingCfg:
    tensorboard: bool
    mlflow: MlflowCfg

@dataclass
class Cfg:
    run: RunCfg
    data: DataCfg
    model: ModelCfg
    train: TrainCfg
    tracking: TrackingCfg


def load_config(path: str) -> Cfg:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = Cfg(
        run=RunCfg(**{k: v for k, v in raw["run"].items() if k not in {"cudnn",}}, cudnn=CudnnCfg(**raw["run"]["cudnn"])),
        data=DataCfg(
            **{k: v for k, v in raw["data"].items() if k not in {}},
        ),
        model=ModelCfg(**raw.get("model", {})),
        train=TrainCfg(**raw["train"]),
        tracking=TrackingCfg(tensorboard=raw["tracking"]["tensorboard"], mlflow=MlflowCfg(**raw["tracking"]["mlflow"])),
    )
    return cfg


def cfg_to_dict(obj):
    """
    Recursively convert nested config objects (dataclasses or custom Cfg types) 
    into plain Python dicts/lists/primitives.
    """
    # handle dataclasses (if your configs use @dataclass)
    if is_dataclass(obj):
        return {k: cfg_to_dict(v) for k, v in asdict(obj).items()}

    # handle dicts
    if isinstance(obj, dict):
        return {k: cfg_to_dict(v) for k, v in obj.items()}

    # handle lists / tuples
    if isinstance(obj, (list, tuple)):
        return [cfg_to_dict(v) for v in obj]

    # handle custom Cfg-like classes with __dict__
    if hasattr(obj, "__dict__"):
        return {k: cfg_to_dict(v) for k, v in vars(obj).items()}

    # base case: primitive
    return obj
