from __future__ import annotations
import argparse, os, signal, copy
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW

from visual.config import load_config, cfg_to_dict
from visual.seed import set_all_seeds
from visual.logger import RunLogger
from visual.riichi_data import build_riichi_dataloader
from visual.riichi_dataset_loader import NUM_FEATURES

from visual.model import VisualClassifier

from visual.checkpointing import (
    save_checkpoint,
    maybe_resume,
    latest_checkpoint,
    prune_old,
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/riichi.yaml")
    return p.parse_args()

def asdict_maybe(obj):
    if obj is None:
        return {}
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    if isinstance(obj, dict):
        return obj
    return {}

# ----------------------------
# Loss utility (masked CE)
# ----------------------------
class MaskedCrossEntropy(torch.nn.Module):
    """Cross-entropy with a (34,) mask where 0-weight classes are ignored.
    Assumes logits shape (B, 34) and targets shape (B,).
    """
    def __init__(self, eps: float = 1e-9):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # logits: (B,34); mask: (B,34) or (34,)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0).expand(logits.size(0), -1)
        # set very negative logits on illegal classes so softmax prob ~0
        neg_inf = -1e9
        masked_logits = logits.clone()
        masked_logits[mask <= 0.0] = neg_inf
        return torch.nn.functional.cross_entropy(masked_logits, targets)
    
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # Device and Seed
    if cfg.run.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.run.device)

    set_all_seeds(cfg.run.seed, cfg.run.cudnn.benchmark, cfg.run.cudnn.deterministic)

    # Data logger
    logger = RunLogger(cfg.run.output_dir, cfg.run.experiment_name, cfg.tracking.tensorboard, mlflow_cfg=cfg.tracking.mlflow.__dict__)
    logger.log_params({"config_path": args.config, **cfg.__dict__["run"].__dict__})

    # Data
    if cfg.data.name == "riichi":
        dl, dl_tst = build_riichi_dataloader(
            root=cfg.data.cfg[cfg.data.name]["root"],
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            download=cfg.data.cfg[cfg.data.name]["download"],
            class_conditional=cfg.data.cfg[cfg.data.name]["class_conditional"],
            img_size=cfg.data.cfg[cfg.data.name]["img_size"],
            cf_guidance_p=cfg.data.cfg[cfg.data.name]["cf_guidance_p"],
        )
        sample_shape = (cfg.train.sample_size, 29, cfg.data.cfg[cfg.data.name]["img_size"], cfg.data.cfg[cfg.data.name]["img_size"])
    else:
        return
    
    # Model & Diffusion
    name = getattr(cfg.model, "name", "resnet18")
    common = asdict_maybe(getattr(cfg.model, "common", None))
    model = VisualClassifier(backbone=name, in_chans=NUM_FEATURES, **common).to(device)
    criterion = nn.CrossEntropyLoss()

    opt = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda step: min(1.0, step/cfg.train.warmup)
    )
    global_step = 0

    # ---------- checkpoint dir & resume ----------
    run_dir = Path(logger.run_dir)
    ckpt_dir = Path(getattr(cfg.train, "checkpoint_dir", "") or (run_dir / "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_step, start_epoch, _ = maybe_resume(
        resume_from=getattr(cfg.train, "resume_from", ""),
        checkpoint_dir=str(ckpt_dir),
        model=model,
        optimizer=opt,
        scheduler=scheduler,
        scaler=None,
        ema=None,
        strict=True,
        restore_rng=getattr(cfg.train, "save_rng_state", True),
    )
    global_step = start_step

    # handle SIGINT/SIGTERM → save & exit cleanly
    stop_flag = {"stop": False}
    def _handle(sig, frame):
        stop_flag["stop"] = True
    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        model.train()
        tr_loss_sum, tr_correct, tr_total = 0.0, 0, 0
        for images, labels, masks in dl:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            scheduler.step()

            preds = logits.argmax(dim=1)
            tr_correct += (preds == labels).sum().item()
            bs = images.size(0)
            tr_total += bs
            tr_loss_sum += loss.item() * bs
            epoch_train_loss = tr_loss_sum / max(tr_total, 1)
            epoch_train_acc = tr_correct / max(tr_total, 1)

            if global_step % cfg.train.log_interval == 0:
                # metric
                logger.log_metric("train/loss", loss.item(), global_step)
            global_step += 1

            # ---------- autosave by step ----------
            save_every = int(getattr(cfg.train, "save_every_steps", 0))
            if save_every and (global_step % save_every == 0 or stop_flag["stop"]):
                step_path = ckpt_dir / f"step_{global_step}.pt"
                save_checkpoint(
                    path=str(step_path),
                    model=model,
                    optimizer=opt,
                    scheduler=scheduler,
                    scaler=None,
                    ema=None,
                    step=global_step,
                    epoch=epoch,
                    config=cfg_to_dict(cfg),
                    metrics=None,
                    save_rng_state=getattr(cfg.train, "save_rng_state", True),
                )
                # refresh latest.pt (hardlink, fallback copy)
                try:
                    latest = ckpt_dir / "latest.pt"
                    if latest.exists(): latest.unlink()
                    os.link(step_path, latest)
                except Exception:
                    import shutil as _sh
                    _sh.copy2(step_path, ckpt_dir / "latest.pt")
                prune_old(str(ckpt_dir), int(getattr(cfg.train, "keep_last_k", 3)))
                if stop_flag["stop"]:
                    print("[signal] checkpoint saved; exiting.")
                    logger.close()
                    return

            if global_step % save_every == 0:
                with torch.no_grad():
                    model.eval()
                    # Validation
                    val_loss_sum, val_correct, val_total = 0.0, 0, 0
                    for images, labels, masks in dl_tst:
                        images, labels = images.to(device), labels.to(device)
                        logits = model(images)
                        loss = criterion(logits, labels)
                        val_loss_sum += loss.item() * images.size(0)
                        val_correct += (logits.argmax(dim=1) == labels).sum().item()
                        val_total += images.size(0)
                    val_loss = val_loss_sum / max(val_total, 1)
                    val_acc = val_correct / max(val_total, 1)

                    logger.log_metric("train/epoch_loss", epoch_train_loss, global_step)
                    logger.log_metric("train/epoch_acc", epoch_train_acc, global_step)
                    logger.log_metric("val/loss", val_loss, global_step)
                    logger.log_metric("val/acc", val_acc, global_step)

        if epoch % cfg.train.sample_interval == 0 or epoch == cfg.train.epochs:
            with torch.no_grad():
                model.eval()
                # Validation
                val_loss_sum, val_correct, val_total = 0.0, 0, 0
                for images, labels, masks in dl_tst:
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images)
                    loss = criterion(logits, labels)
                    val_loss_sum += loss.item() * images.size(0)
                    val_correct += (logits.argmax(dim=1) == labels).sum().item()
                    val_total += images.size(0)
                val_loss = val_loss_sum / max(val_total, 1)
                val_acc = val_correct / max(val_total, 1)

                logger.log_metric("train/epoch_loss", epoch_train_loss, global_step)
                logger.log_metric("train/epoch_acc", epoch_train_acc, global_step)
                logger.log_metric("val/loss", val_loss, global_step)
                logger.log_metric("val/acc", val_acc, global_step)


        if epoch % cfg.train.ckpt_interval == 0 or epoch == cfg.train.epochs:
            # keep your epoch checkpoint, but include optimizer/step/rng to make it resumable too
            ckpt_path = Path(logger.run_dir) / f"model-epoch{epoch}.pt"
            save_checkpoint(
                path=str(ckpt_path),
                model=model,
                optimizer=opt,
                scheduler=scheduler,
                scaler=None,
                ema=None,
                step=global_step,
                epoch=epoch,
                config=cfg_to_dict(cfg),
                metrics=None,
                save_rng_state=getattr(cfg.train, "save_rng_state", True),
            )

    logger.close()


if __name__ == "__main__":
    main()
