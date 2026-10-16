[← README](../README.md)

## Requirements

- Ubuntu 22.04 or 24.04.
- NVIDIA GPU with a driver that supports CUDA 12.8.
- Isaac Sim 4.5.0 (pip).
- Isaac Lab 2.2.0.
- Python 3.10.

## Installation

Run the commands below in order.

```bash
conda create -n humanoid-compliance python=3.10 -y
conda activate humanoid-compliance

pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128

pip install 'isaacsim[all,extscache]==4.5.0' \
  --extra-index-url https://pypi.nvidia.com
```

Install Isaac Lab 2.2.0 (the first two lines are required for its build):

```bash
pip install "setuptools<81"
pip install --no-build-isolation flatdict==4.0.1

git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.2.0
./isaaclab.sh -i none
cd -
```

Install this repository from its root directory:

```bash
cd /path/to/this/repository
pip install -e .
```

Do not upgrade `numpy` or `warp-lang`; `setup.py` pins them to match Isaac Sim 4.5.

Set these variables in every shell that runs the simulator:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export ISAACLAB_PATH=/path/to/IsaacLab
export PYTHONPATH="$ISAACLAB_PATH/source/isaaclab:$PWD"
```

Check the installation:

```bash
pip list | grep -E "^(isaaclab|isaacsim|numpy|warp-lang|torch) "
# Expect: isaaclab 0.44.x, isaacsim 4.5.0.0, numpy 1.26.x, torch 2.7.0+cu128, warp-lang 1.5.0
pytest tests
```

### W&B

Training stages are chained through W&B runs. Run `wandb login` once and set
your entity:

```bash
export WANDB_ENTITY=your-wandb-user-or-team
```

For local debugging, append `wandb.mode=disabled` to a `scripts/train.py` command.

## Quick smoke test

Prepare the [dataset](data.md) before running this check.

A one-minute single-GPU run that checks the installation and dataset:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  task=G1/G1_3kp_stiff +exp=train \
  wandb.mode=disabled task.num_envs=256 total_frames=40960
```

It finishes with `Saved checkpoint to .../checkpoint_final.pt`.

