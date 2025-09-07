#!/usr/bin/env python
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from pathlib import Path
import torch
# import the dataset classes + cfgs you have defined
from diffusion_sandbox.data import (
    SyntheticGMM, SyntheticGMMCfg,
    SyntheticRing, SyntheticRingCfg,
    SyntheticTwoMoons, SyntheticTwoMoonsCfg,
    SyntheticConcentricCircles, SyntheticConcentricCirclesCfg,
    SyntheticSpiral, SyntheticSpiralCfg,
    SyntheticCheckerboard, SyntheticCheckerboardCfg,
    SyntheticPinwheel, SyntheticPinwheelCfg,
    SyntheticSwissRoll2D, SyntheticSwissRoll2DCfg,
)
from diffusion_sandbox.viz import scatter_2d

out_dir = Path(__file__).resolve().parents[1] / "examples" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

# GMM Visualisation
gmm = SyntheticGMM(2000, SyntheticGMMCfg(), seed=42)
fig = scatter_2d(torch.stack([gmm[i] for i in range(len(gmm))]), None, title="Synthetic GMM")
fig.savefig(out_dir / "gmm.png")

# Ring Visualisation
ring = SyntheticRing(2000, SyntheticRingCfg(), seed=42)
fig = scatter_2d(torch.stack([ring[i] for i in range(len(ring))]), None, title="Synthetic Ring")
fig.savefig(out_dir / "ring.png")

# Two_moons Visualisation
two_moons = SyntheticTwoMoons(2000, SyntheticTwoMoonsCfg(), seed=42)
fig = scatter_2d(torch.stack([two_moons[i] for i in range(len(two_moons))]), None, title="Synthetic Two_moons")
fig.savefig(out_dir / "two_moons.png")

# Concentric Visualisation
concentric = SyntheticConcentricCircles(2000, SyntheticConcentricCirclesCfg(), seed=42)
fig = scatter_2d(torch.stack([concentric[i] for i in range(len(concentric))]), None, title="Synthetic Concentric")
fig.savefig(out_dir / "concentric.png")

# Spiral Visualisation
spiral = SyntheticSpiral(2000, SyntheticSpiralCfg(), seed=42)
fig = scatter_2d(torch.stack([spiral[i] for i in range(len(spiral))]), None, title="Synthetic Spiral")
fig.savefig(out_dir / "spiral.png")

# Checkerboard Visualisation
checkerboard = SyntheticCheckerboard(2000, SyntheticCheckerboardCfg(), seed=42)
fig = scatter_2d(torch.stack([checkerboard[i] for i in range(len(checkerboard))]), None, title="Synthetic Checkerboard")
fig.savefig(out_dir / "checkerboard.png")

# Pinwheel Visualisation
pinwheel = SyntheticPinwheel(2000, SyntheticPinwheelCfg(), seed=42)
fig = scatter_2d(torch.stack([pinwheel[i] for i in range(len(pinwheel))]), None, title="Synthetic Pinwheel")
fig.savefig(out_dir / "pinwheel.png")

# Swiss_roll2d Visualisation
swiss_roll2d = SyntheticSwissRoll2D(2000, SyntheticSwissRoll2DCfg(), seed=42)
fig = scatter_2d(torch.stack([swiss_roll2d[i] for i in range(len(swiss_roll2d))]), None, title="Synthetic Swiss_roll2d")
fig.savefig(out_dir / "swiss_roll2d.png")

print(f"Saved figures to {out_dir}")