[← 中文首页](../README_zh.md)

## 环境要求

- Ubuntu 22.04 或 24.04。
- 支持 CUDA 12.8 的 NVIDIA GPU。
- Isaac Sim 4.5.0（pip 安装）。
- Isaac Lab 2.2.0。
- Python 3.10。

## 安装

按以下顺序执行：

```bash
conda create -n humanoid-compliance python=3.10 -y
conda activate humanoid-compliance

pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128

pip install 'isaacsim[all,extscache]==4.5.0' \
  --extra-index-url https://pypi.nvidia.com
```

安装 Isaac Lab 2.2.0：

```bash
pip install "setuptools<81"
pip install --no-build-isolation flatdict==4.0.1

git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.2.0
./isaaclab.sh -i none
cd -
```

从仓库根目录安装本项目：

```bash
cd /path/to/this/repository
pip install -e .
```

不要升级 `numpy` 或 `warp-lang`；`setup.py` 中固定了与 Isaac Sim 4.5 匹配的版本。

每个运行 simulator 的 shell 都需要设置：

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export ISAACLAB_PATH=/path/to/IsaacLab
export PYTHONPATH="$ISAACLAB_PATH/source/isaaclab:$PWD"
```

检查安装：

```bash
pip list | grep -E "^(isaaclab|isaacsim|numpy|warp-lang|torch) "
pytest tests
```

### W&B 配置

训练流程通过 W&B run 传递 checkpoint。先登录 W&B，并设置自己的 entity：

```bash
wandb login
export WANDB_ENTITY=your-wandb-user-or-team
```

本地调试时，可以在 `scripts/train.py` 命令后添加 `wandb.mode=disabled`。

## 快速 smoke test

运行前请先完成[数据准备](data_zh.md)。

下面的命令使用单张 GPU 运行约一分钟，用于检查安装和数据集：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  task=G1/G1_3kp_stiff +exp=train \
  wandb.mode=disabled task.num_envs=256 total_frames=40960
```

成功后会显示 `Saved checkpoint to .../checkpoint_final.pt`。

