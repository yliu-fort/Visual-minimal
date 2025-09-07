#!/usr/bin/env bash
set -e
#python -m diffusion_sandbox.train --config configs/default.yaml
PYTHONPATH="$PWD/src" python src/diffusion_sandbox/train.py --config configs/default.yaml
#tensorboard --logdir runs