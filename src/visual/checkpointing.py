import os, io, tempfile, shutil, glob, re, random
from typing import Optional, Dict, Any, List
import torch
import numpy as np


def _atomic_save(obj: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # write to tmp then rename for atomicity
    with tempfile.NamedTemporaryFile(dir=os.path.dirname(path), delete=False) as tmp:
        torch.save(obj, tmp.name)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp.name, path)

def _rng_state_dump():
    return {
        "py_random": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": np.random.get_state(),
    }

def _rng_state_load(state: Dict[str, Any]):
    if not state: return
    random.setstate(state["py_random"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    np.random.set_state(state["numpy"])

def prune_old(checkpoint_dir: str, keep_last_k: int):
    if keep_last_k <= 0: return
    files = sorted(glob.glob(os.path.join(checkpoint_dir, "step_*.pt")),
                   key=lambda p: int(re.search(r"step_(\d+)\.pt$", p).group(1)))
    for p in files[:-keep_last_k]:
        try: os.remove(p)
        except OSError: pass
        
def save_checkpoint(
    path: str,
    *,
    model,
    optimizer,
    scheduler=None,
    scaler=None,
    ema=None,
    step: int,
    epoch: int,
    config: Dict[str, Any],
    metrics: Optional[Dict[str, float]] = None,
    save_rng_state: bool = True,
):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "ema": ema.state_dict() if ema else None,
        "step": step,
        "epoch": epoch,
        "config": config,
        "metrics": metrics or {},
        "rng": _rng_state_dump() if save_rng_state else None,
        "torch_version": torch.__version__,
    }
    _atomic_save(payload, path)

def latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    candidates = glob.glob(os.path.join(checkpoint_dir, "step_*.pt"))
    if not candidates: return None
    return max(candidates, key=lambda p: int(re.search(r"step_(\d+)\.pt$", p).group(1)))

def maybe_resume(
    resume_from: str,
    checkpoint_dir: str,
    model,
    optimizer,
    scheduler=None,
    scaler=None,
    ema=None,
    strict: bool = True,
    restore_rng: bool = True,
):
    ckpt_path = None
    if resume_from:
        ckpt_path = resume_from if resume_from != "latest" else latest_checkpoint(checkpoint_dir)
    if not ckpt_path or not os.path.exists(ckpt_path):
        return 0, 1, {}
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=strict)
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"): scheduler.load_state_dict(ckpt["scheduler"])
    if scaler and ckpt.get("scaler"): scaler.load_state_dict(ckpt["scaler"])
    if ema and ckpt.get("ema"): ema.load_state_dict(ckpt["ema"])
    if restore_rng and ckpt.get("rng"): _rng_state_load(ckpt["rng"])
    step, epoch = ckpt.get("step", 0), ckpt.get("epoch", 1)
    return step, epoch, ckpt.get("metrics", {})
