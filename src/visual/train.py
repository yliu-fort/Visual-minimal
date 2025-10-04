from __future__ import annotations
import argparse, os, signal, copy
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW

from visual.config import load_config, cfg_to_dict
from visual.seed import set_all_seeds
from visual.logger import RunLogger
from visual.dataset_loader_web import (
    build_riichi_dataloader,
    NUM_FEATURES,
    NUM_ACTIONS,
    compute_resize_shape,
    resize_batch_on_device,
)

from visual.model import VisualClassifier
from tqdm import tqdm

from visual.checkpointing import (
    save_checkpoint,
    maybe_resume,
    latest_checkpoint,
    prune_old,
)

class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        model_state = self.model.state_dict(keep_vars=True)
        self.shadow = {
            name: tensor.detach().clone()
            for name, tensor in model_state.items()
        }

    @torch.no_grad()
    def update(self):
        model_state = self.model.state_dict(keep_vars=True)
        for name, param in model_state.items():
            if not isinstance(param, torch.Tensor) or not param.dtype.is_floating_point:
                continue

            param_data = param.detach()
            shadow = self.shadow.get(name)
            if shadow is None:
                shadow = param_data.clone()
            else:
                if shadow.device != param_data.device or shadow.dtype != param_data.dtype:
                    shadow = shadow.to(device=param_data.device, dtype=param_data.dtype)

            shadow.mul_(self.decay).add_(param_data, alpha=1 - self.decay)
            self.shadow[name] = shadow

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=False)

    def state_dict(self):
        """Save EMA parameter"""
        return {
            "decay": self.decay,
            "shadow": self.shadow
        }

    def load_state_dict(self, state_dict):
        """Restore EMA parameter"""
        self.decay = state_dict["decay"]
        model_state = self.model.state_dict(keep_vars=True)
        restored = {}
        for name, tensor in state_dict["shadow"].items():
            target = model_state.get(name)
            clone = tensor.clone().detach()
            if isinstance(target, torch.Tensor):
                target_data = target.detach()
                if clone.device != target_data.device or clone.dtype != target_data.dtype:
                    clone = clone.to(device=target_data.device, dtype=target_data.dtype)
            restored[name] = clone

        for name, param in model_state.items():
            if name not in restored and isinstance(param, torch.Tensor):
                restored[name] = param.detach().clone()

        self.shadow = restored

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
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # logits: (B,34); mask: (B,34) or (34,)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0).expand(logits.size(0), -1)

        # 用 masked_fill + 可靠的负大数（或 -inf）
        neg_large = -torch.finfo(logits.dtype).max  # 避免混合精度下的数值问题
        masked_logits = logits.masked_fill(mask <= 0, neg_large)

        return torch.nn.functional.cross_entropy(masked_logits, targets, label_smoothing=self.label_smoothing)
    

def classify(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # logits: (B,34); mask: (B,34) or (34,)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0).expand(logits.size(0), -1)

    # 用 masked_fill + 可靠的负大数（或 -inf）
    neg_large = -torch.finfo(logits.dtype).max  # 避免混合精度下的数值问题
    masked_logits = logits.masked_fill(mask <= 0, neg_large)

    return masked_logits.argmax(dim=1)
    

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
        sample_shape = (cfg.train.sample_size, NUM_FEATURES, cfg.data.cfg[cfg.data.name]["img_size"], cfg.data.cfg[cfg.data.name]["img_size"])
        target_resize = compute_resize_shape(cfg.data.cfg[cfg.data.name]["img_size"])
    else:
        return
    
    # Model & Diffusion
    name = getattr(cfg.model, "name", "resnet18")
    common = asdict_maybe(getattr(cfg.model, "common", None))
    if cfg.data.name == "riichi" and cfg.data.cfg[cfg.data.name]["class_conditional"]:
        current_num_classes = common.get("num_classes")
        if current_num_classes not in (None, NUM_ACTIONS):
            print(
                f"[train] Overriding num_classes={current_num_classes} with NUM_ACTIONS={NUM_ACTIONS} "
                "to match dataset mask."
            )
        common["num_classes"] = NUM_ACTIONS

    model = VisualClassifier(backbone=name, in_chans=NUM_FEATURES, **common).to(device)
    criterion = MaskedCrossEntropy(label_smoothing=cfg.train.label_smoothing)
    opt = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    # 线性 warmup：从极小倍率线性升到 1.0
    warmup = torch.optim.lr_scheduler.LinearLR(
        opt,
        start_factor=1e-3,            # 避免 0 / 小到几乎 0
        end_factor=1.0,
        total_iters=cfg.train.warmup,
    )
    # 余弦衰减：从 base_lr 下降到 eta_min
    cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt,
        T_mult=2,
        T_0=cfg.data.num_samples // (cfg.data.batch_size * cfg.train.gradient_accumulation_steps),
        eta_min=cfg.train.lr * cfg.train.min_lr_ratio,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup, cosine], milestones=[cfg.train.warmup])
    ema = EMA(model, decay=cfg.train.ema_decay)

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
        ema=ema,
        strict=True,
        restore_rng=getattr(cfg.train, "save_rng_state", True),
    )
    global_step = start_step
    accum_steps = getattr(cfg.train, "gradient_accumulation_steps", 1)

    # handle SIGINT/SIGTERM → save & exit cleanly
    stop_flag = {"stop": False}
    def _handle(sig, frame):
        stop_flag["stop"] = True
    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        model.train()
        tr_loss_sum, tr_correct, tr_total = 0.0, 0, 0

        opt.zero_grad(set_to_none=True)
        for it, (images, labels, masks) in enumerate(dl):
            images = images.to(device, non_blocking=True)
            images = resize_batch_on_device(images, target_resize)
            labels = labels.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels, masks)

            # === 梯度累积：反传用缩放后的 loss ===
            (loss / accum_steps).backward()

            # 仅在累计满时做一次优化器更新 & 相关操作
            if ((it + 1) % accum_steps) == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.train.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                scheduler.step()
                ema.update()
                global_step += 1

                preds = classify(logits, masks)
                correct_ = (preds == labels).sum().item()
                tr_correct += correct_
                bs = images.size(0)
                tr_total += bs
                tr_loss_sum += loss.item() * bs
                epoch_train_loss = tr_loss_sum / max(tr_total, 1)
                epoch_train_acc = tr_correct / max(tr_total, 1)

                if global_step % cfg.train.log_interval == 0:
                    # metric
                    logger.log_metric("train/loss", loss.item(), global_step)
                    logger.log_metric("train/acc", correct_/bs, global_step)
                    logger.log_metric("train/learning_rate", scheduler.get_last_lr()[0], global_step)


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
                        ema=ema,
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
                        ema_model = copy.deepcopy(model)
                        ema.copy_to(ema_model)
                        # Validation
                        val_loss_sum, val_correct, val_total = 0.0, 0, 0
                        for images, labels, masks in tqdm(dl_tst):
                            images = images.to(device, non_blocking=True)
                            images = resize_batch_on_device(images, target_resize)
                            labels = labels.to(device, non_blocking=True)
                            masks = masks.to(device, non_blocking=True)
                            logits = ema_model(images)
                            loss = criterion(logits, labels, masks)
                            val_loss_sum += loss.item() * images.size(0)
                            val_correct += (classify(logits, masks) == labels).sum().item()
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
                ema=ema,
                step=global_step,
                epoch=epoch,
                config=cfg_to_dict(cfg),
                metrics=None,
                save_rng_state=getattr(cfg.train, "save_rng_state", True),
            )

        if epoch % cfg.train.sample_interval == 0 or epoch == cfg.train.epochs:
            with torch.no_grad():
                model.eval()
                ema_model = copy.deepcopy(model)
                ema.copy_to(ema_model)
                # Validation
                val_loss_sum, val_correct, val_total = 0.0, 0, 0
                for images, labels, masks in dl_tst:
                    images = images.to(device, non_blocking=True)
                    images = resize_batch_on_device(images, target_resize)
                    labels = labels.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)
                    logits = ema_model(images)
                    loss = criterion(logits, labels, masks)
                    val_loss_sum += loss.item() * images.size(0)
                    val_correct += (classify(logits, masks) == labels).sum().item()
                    val_total += images.size(0)
                val_loss = val_loss_sum / max(val_total, 1)
                val_acc = val_correct / max(val_total, 1)

                logger.log_metric("train/epoch_loss", epoch_train_loss, global_step)
                logger.log_metric("train/epoch_acc", epoch_train_acc, global_step)
                logger.log_metric("val/loss", val_loss, global_step)
                logger.log_metric("val/acc", val_acc, global_step)
                

    logger.close()


if __name__ == "__main__":
    main()
