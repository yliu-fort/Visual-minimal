#!/usr/bin/env bash
set -euo pipefail

# ===  ===
PY_VER="3.10"
ENV_NAME="visual"
REPO_URL="https://github.com/yliu-fort/Visual-minimal.git"
REPO_DIR="/data/Visual-minimal"

echo "[*] Updating packages..."
sudo apt-get update -y
sudo apt-get install -y git wget curl tmux htop unzip


echo "[*] Installing PyTorch (CUDA 12.1 wheels) ..."
pip install --upgrade pip

echo "[*] Cloning repo..."
# Backup runs if it exists
if [ -d "$REPO_DIR/runs" ]; then
  rm -rf /data/runs
  cp -r "$REPO_DIR/runs" /data
fi

# Remove old repo and clone fresh
rm -rf "$REPO_DIR"
git clone "$REPO_URL" "$REPO_DIR"

# Restore runs if backup exists
if [ -d "/data/runs" ]; then
  cp -r /data/runs "$REPO_DIR"
fi
cd "$REPO_DIR" || { echo " : $REPO_DIR"; exit 1; }

echo "[*] Installing project requirements..."
if [[ -f requirements.txt ]]; then
pip install -r requirements.txt
fi

echo "[*] Running unittests..."
python -m unittest discover -s ./tests -v || true



echo
echo "[] Init done."
echo "To start working:"
echo "cd $REPO_DIR"
echo "PYTHONPATH="$PWD/src" python src/visual/train.py --config configs/default.yaml"
echo "nohup env PYTHONPATH=/data/DDPM-minimal/src python src/visual/train.py --config configs/default.yaml > log.out 2>&1 &"
echo "nohup tensorboard --logdir runs --host 0.0.0.0 --port 6007 > tb.out 2>&1 &"

cd $REPO_DIR
nohup env PYTHONPATH="$PWD/src" python src/visual/train.py --config configs/default.yaml > log.out 2>&1 &
nohup tensorboard --logdir runs --host 0.0.0.0 --port 6007 > tb.out 2>&1 &