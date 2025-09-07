#!/usr/bin/env bash
set -euo pipefail

# === Disk Management ===
DEVICE="/dev/nvme2n1"
MOUNT_POINT="/data"

# Check the file system type
FSTYPE=$(lsblk -no FSTYPE "$DEVICE")

if [ "$FSTYPE" != "ext4" ]; then
  echo "[$DEVICE] is not ext4, start disk initialisation..."
  sudo mkfs -t ext4 -F "$DEVICE"
else
  echo "[$DEVICE] is ext4, skip initialisation."
fi

# Create mount point and mount the disk
sudo mkdir -p "$MOUNT_POINT" || true
sudo mount "$DEVICE" "$MOUNT_POINT" || true

# override the user permission
sudo chown ubuntu:ubuntu "$MOUNT_POINT"

# ===  ===
PY_VER="3.10"
ENV_NAME="diffusion"
REPO_URL="https://github.com/yliu-fort/DDPM-minimal.git"
REPO_DIR="/data/DDPM-minimal"

source /opt/pytorch/bin/activate

echo "[*] Updating packages..."
sudo apt-get update -y
sudo apt-get install -y git wget curl tmux htop unzip


echo "[*] Installing PyTorch (CUDA 12.1 wheels) ..."
pip install --upgrade pip
#pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

echo "[*] Cloning repo..."
# Backup runs if it exists
if [ -d "$REPO_DIR/runs" ]; then
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
echo "source /opt/pytorch/bin/activate"
echo "PYTHONPATH="$PWD/src" python src/diffusion_sandbox/train.py --config configs/cifar10_uncond.yaml"
echo "nohup env PYTHONPATH=/data/DDPM-minimal/src python src/diffusion_sandbox/train.py --config configs/cifar10_uncond.yaml > log.out 2>&1 &"
echo "nohup tensorboard --logdir runs --host 0.0.0.0 --port 6006 > tb.out 2>&1 &"

cd $REPO_DIR
source /opt/pytorch/bin/activate
nohup env PYTHONPATH="$PWD/src" python src/diffusion_sandbox/train.py --config configs/mnist_uncond.yaml > log.out 2>&1 &
nohup tensorboard --logdir runs --host 0.0.0.0 --port 6006 > tb.out 2>&1 &