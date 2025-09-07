#!/usr/bin/env python
from __future__ import annotations
import argparse, json, sys, pathlib
# ensure we can import the project when placed at repo root; or run with --src to point to src dir
def _append_src(src_hint: str | None):
    if src_hint:
        sys.path.append(src_hint)
        return
    # try common layouts
    here = pathlib.Path(__file__).resolve().parent
    candidates = [
        here / "src",
        here / "src/diffusion_sandbox",
        here.parent / "src",
    ]
    for c in candidates:
        if c.exists():
            sys.path.append(str(c if c.name != "diffusion_sandbox" else c.parent))
            return

def main():
    ap = argparse.ArgumentParser(description="Count model parameter sizes from YAML config")
    ap.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    ap.add_argument("--src", type=str, default=None, help="Optional path to your src/ root (so we can import diffusion_sandbox)")
    args = ap.parse_args()
    _append_src(args.src)

    import torch
    from diffusion_sandbox.config import load_config
    from diffusion_sandbox.models import REGISTRY

    def asdict_maybe(obj):
        if obj is None: return {}
        if hasattr(obj, "__dict__"): return dict(obj.__dict__)
        if isinstance(obj, dict): return obj
        return {}

    cfg = load_config(args.config)
    name = getattr(cfg.model, "name", "mlp_baseline")
    common = asdict_maybe(getattr(cfg.model, "common", None))
    specific = asdict_maybe(getattr(cfg.model, name, None))
    if name not in REGISTRY:
        raise SystemExit(f"Unknown model name: {name}. Known: {list(REGISTRY.keys())}")
    ModelCls = REGISTRY[name]
    model = ModelCls(**common, **specific)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info = {
        "model_name": name,
        "common": common,
        "specific": specific,
        "total_params": int(total),
        "trainable_params": int(trainable),
        "total_params_million": round(total / 1_000_000, 6),
    }
    print(json.dumps(info, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
