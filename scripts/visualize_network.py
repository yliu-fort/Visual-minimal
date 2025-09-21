#!/usr/bin/env python3
"""Utility script to visualise the tensor shapes flowing through a model.

The script reads a YAML configuration file (identical to the training
configuration) and instantiates the model defined there.  It then performs a
single forward pass with a dummy tensor of the requested shape and prints a
compact summary containing the output shapes of every leaf module.
"""
from __future__ import annotations

import argparse
import inspect
from typing import List, Sequence, Tuple

import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:  # Prefer the project's config dataclasses when available.
    from visual.config import load_config as _load_config  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _load_config = None

from visual.model import VisualClassifier


def parse_shape(shape: str) -> Tuple[int, ...]:
    """Parse a textual representation of a tensor shape.

    Accepts strings such as ``"1,3,224,224"`` or ``"(1, 3, 224, 224)``.
    """
    cleaned = shape.strip().replace(" ", "")
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = cleaned[1:-1]
    if not cleaned:
        raise ValueError("Input shape must contain at least one dimension")

    dims: List[int] = []
    for chunk in cleaned.split(","):
        if not chunk:
            continue
        try:
            dim = int(chunk)
        except ValueError as exc:  # pragma: no cover - defensive branch
            raise ValueError(f"Invalid dimension '{chunk}' in shape '{shape}'") from exc
        if dim <= 0:
            raise ValueError("All dimensions must be positive integers")
        dims.append(dim)

    if len(dims) < 2:
        raise ValueError("Expected at least batch and channel dimensions")
    return tuple(dims)


def format_shape_list(shapes: Sequence[Sequence[int]]) -> str:
    if not shapes:
        return "-"
    formatted = ["(" + ", ".join(str(v) for v in shape) + ")" for shape in shapes]
    return ", ".join(formatted)


def collect_summary(model: torch.nn.Module, sample: torch.Tensor) -> Tuple[List[dict], Tuple[int, ...]]:
    summary: List[dict] = []
    hooks = []

    def register_hook(name: str, module: torch.nn.Module) -> None:
        def hook(mod: torch.nn.Module, inputs: Tuple[torch.Tensor, ...], outputs: torch.Tensor):
            input_shapes = [tuple(inp.shape) for inp in inputs if isinstance(inp, torch.Tensor)]
            if isinstance(outputs, torch.Tensor):
                output_shapes: List[Tuple[int, ...]] = [tuple(outputs.shape)]
            elif isinstance(outputs, (list, tuple)):
                output_shapes = [tuple(out.shape) for out in outputs if isinstance(out, torch.Tensor)]
            else:  # pragma: no cover - unusual case
                output_shapes = []

            params = sum(p.numel() for p in mod.parameters(recurse=False))
            trainable = sum(p.numel() for p in mod.parameters(recurse=False) if p.requires_grad)
            summary.append(
                {
                    "name": name,
                    "type": mod.__class__.__name__,
                    "input": input_shapes,
                    "output": output_shapes,
                    "params": params,
                    "trainable": trainable,
                }
            )

        hooks.append(module.register_forward_hook(hook))

    for module_name, module in model.named_modules():
        if module is model:
            continue
        if len(list(module.children())) == 0:
            register_hook(module_name, module)

    with torch.no_grad():
        outputs = model(sample)

    for hook_handle in hooks:
        hook_handle.remove()

    if isinstance(outputs, torch.Tensor):
        final_shape = tuple(outputs.shape)
    elif isinstance(outputs, (list, tuple)) and outputs and isinstance(outputs[0], torch.Tensor):
        final_shape = tuple(outputs[0].shape)
    else:
        final_shape = ()

    return summary, final_shape


def filter_common_kwargs(common: dict) -> dict:
    if not isinstance(common, dict):
        return {}
    signature = inspect.signature(VisualClassifier.__init__)
    valid = {
        name for name in signature.parameters if name not in {"self", "backbone", "in_chans"}
    }
    return {k: v for k, v in common.items() if k in valid}


def extract_model_config(config_path: str) -> Tuple[str, dict, int | None]:
    if _load_config is not None:
        try:
            cfg = _load_config(config_path)
        except Exception:
            pass
        else:
            name = getattr(cfg.model, "name", "resnet18")
            raw_common = getattr(cfg.model, "common", {}) or {}
            if hasattr(raw_common, "__dict__"):
                common = dict(raw_common.__dict__)
            else:
                common = dict(raw_common)
            input_dim = getattr(cfg.model, "input_dim", None)
            return name, common, input_dim

    with open(config_path, "r", encoding="utf-8") as handle:
        raw_cfg = yaml.safe_load(handle) or {}

    model_section = raw_cfg.get("model", {}) or {}
    name = model_section.get("name", "resnet18")
    common = dict(model_section.get("common", {}) or {})
    input_dim = model_section.get("input_dim")
    return name, common, input_dim


def build_model_from_config(config_path: str, in_chans: int) -> VisualClassifier:
    backbone, common_raw, input_dim = extract_model_config(config_path)
    common_cfg = filter_common_kwargs(common_raw)

    if "num_classes" not in common_cfg:
        inferred = input_dim if isinstance(input_dim, int) else None
        common_cfg["num_classes"] = inferred or common_cfg.get("num_classes", 10)

    model = VisualClassifier(backbone=backbone, in_chans=in_chans, **common_cfg)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise network tensor shapes")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/resnet18.yaml",
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--input-shape",
        type=str,
        default="1,3,224,224",
        help="Input tensor shape, e.g. '(1,3,224,224)'",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the dummy forward pass on",
    )
    args = parser.parse_args()

    tensor_shape = parse_shape(args.input_shape)
    device = torch.device(args.device)

    model = build_model_from_config(args.config, tensor_shape[1])
    model.to(device)
    model.eval()

    sample = torch.randn(*tensor_shape, device=device)

    summary, final_shape = collect_summary(model, sample)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Configuration: {args.config}")
    print(f"Backbone: {model.backbone_name}")
    print(f"Input shape: {tuple(tensor_shape)}")
    if final_shape:
        print(f"Output shape: {final_shape}")
    print("\nLayer summary:")
    header = f"{'Layer (name)':<45} {'Type':<30} {'Output shape(s)':<35} {'Param #':>12} {'Trainable':>12}"
    print(header)
    print("-" * len(header))
    for entry in summary:
        print(
            f"{entry['name']:<45} {entry['type']:<30} {format_shape_list(entry['output']):<35} "
            f"{entry['params']:>12,d} {entry['trainable']:>12,d}"
        )

    print("-" * len(header))
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")


if __name__ == "__main__":
    main()
