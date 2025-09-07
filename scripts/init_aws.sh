#!/usr/bin/env bash
set -euo pipefail

# === Disk Management ===
DEVICE="/dev/nvme1n1"
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

echo "[*] Updating packages..."
sudo apt-get update -y
sudo apt-get install -y git wget curl tmux htop unzip

#  NVIDIA DLAMI 
if [[ -e /proc/driver/nvidia/version ]]; then
  echo "[*] NVIDIA driver detected."
else
  echo "[!] NVIDIA driver not detected. If you are on plain Ubuntu, consider using AWS Deep Learning AMI or install drivers manually:"
  echo "    sudo apt-get install -y ubuntu-drivers-common && sudo ubuntu-drivers autoinstall && sudo reboot"
  echo "    After reboot, re-run this script."
fi

#  Miniconda
source $HOME/.bashrc
if ! command -v conda >/dev/null 2>&1; then
  echo "[*] Installing Miniconda..."
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p "$HOME/miniconda"
  rm miniconda.sh
  echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> "$HOME/.bashrc"
  source $HOME/.bashrc
fi

# -------------------------------------
# 2)  conda activate 
# -------------------------------------
#  shell  activate  "Run 'conda init' before 'conda activate'"
#  (a) conda init bash (b)  shell  hook
echo "[*] Initializing conda for bash..."
"$HOME/miniconda/bin/conda" init bash || true
#  conda 
# A conda  hook
eval "$("$HOME/miniconda/bin/conda" shell.bash hook)" || {
  # Bfallback  profile.d
  # shellcheck disable=SC1091
  source "$HOME/miniconda/etc/profile.d/conda.sh" 2>/dev/null || true
}
source $HOME/.bashrc
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# -------------------------------------
# 3) /
# -------------------------------------
# 
conda env remove -n "$ENV_NAME" -y >/dev/null 2>&1 || true

echo "[*] Creating conda env ($ENV_NAME, python=$PY_VER)..."
conda create -n "$ENV_NAME" "python=${PY_VER}" -y

# --- 1 activate ---
if conda activate "$ENV_NAME" 2>/dev/null; then
  echo "[*] Activated env via 'conda activate $ENV_NAME'."

  echo "[*] Installing PyTorch (CUDA 12.1 wheels) ..."
  pip install --upgrade pip
  pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

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

else
  # --- 2 activate conda run ---
  echo "[!] 'conda activate' not available in this shell; proceeding with 'conda run' path."

  echo "[*] Installing PyTorch (CUDA 12.1 wheels) ..."
  "$HOME/miniconda/bin/conda" run -n "$ENV_NAME" python -m pip install --upgrade pip
  "$HOME/miniconda/bin/conda" run -n "$ENV_NAME" python -m pip install \
      --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

  echo "[*] Cloning repo..."
  # Backup runs if it exists
  if [ -d "$REPO_DIR/runs" ]; then
    cp -r "$REPO_DIR/runs" /data/runs
  fi

  # Remove old repo and clone fresh
  rm -rf "$REPO_DIR"
  git clone "$REPO_URL" "$REPO_DIR"

  # Restore runs if backup exists
  if [ -d "/data/runs" ]; then
    cp -r /data/runs "$REPO_DIR/runs"
  fi
  cd "$REPO_DIR" || { echo " : $REPO_DIR"; exit 1; }

  echo "[*] Installing project requirements..."
  if [[ -f requirements.txt ]]; then
    "$HOME/miniconda/bin/conda" run -n "$ENV_NAME" python -m pip install -r requirements.txt
  fi

  echo "[*] Running unittests..."
  "$HOME/miniconda/bin/conda" run -n "$ENV_NAME" python -m unittest discover -s ./tests -v || true
fi

echo
echo "[] Init done."
echo "To start working:"
echo "cd $REPO_DIR"
echo "source $HOME/.bashrc"
echo "conda activate $ENV_NAME"
echo "PYTHONPATH="$PWD/src" python src/diffusion_sandbox/train.py --config configs/cifar10_uncond.yaml"
echo "nohup env PYTHONPATH=$PWD/src python src/diffusion_sandbox/train.py --config configs/cifar10_uncond.yaml > log.out 2>&1 &"
echo "nohup tensorboard --logdir runs --host 0.0.0.0 --port 6006 > tb.out 2>&1 &"

cd $REPO_DIR
source $HOME/.bashrc
conda activate "$ENV_NAME"
nohup env PYTHONPATH="$PWD/src" python src/diffusion_sandbox/train.py --config configs/cifar10_uncond.yaml > log.out 2>&1 &
nohup tensorboard --logdir runs --host 0.0.0.0 --port 6006 > tb.out 2>&1 &