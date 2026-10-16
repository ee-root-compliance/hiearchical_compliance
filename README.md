<h1 align="center">CEER</h1>
<p align="center"><strong>Humanoid whole-body control with tunable compliance</strong></p>
<p align="center">
  <a href="https://arxiv.org/abs/2605.19981">CEER1 Paper</a> ·
  <a href="https://robotproject8.github.io/ceer_page/">CEER1 Website</a> ·
  <a href="https://arxiv.org/abs/2609.38709">CEER2 Paper</a> ·
  <a href="https://ee-root-compliance.github.io/">CEER2 Website</a> ·
  <a href="https://ee-root-compliance.github.io/live-demo/"><strong>Live Demo ↗</strong></a>
</p>
<p align="center">English · <a href="README_zh.md">简体中文</a></p>

[![Directional compliance, adjustable stiffness, and collaborative carrying on a humanoid](docs/assets/ceer2-overview.png)](https://ee-root-compliance.github.io/)

<p align="center">Directional compliance · Adjustable stiffness · Collaborative carrying</p>

## Control framework

**CEER1** turns end-effector and root commands into whole-body motion.

**CEER2** adds high-level policies and an analytical MoE to control how the robot responds to forces.

![CEER2 produces EE–root commands for the CEER1 low-level controller; teleoperation or a planner can also command the low-level controller directly](docs/assets/control-stack.svg)

CEER2 uses a frozen stiff backbone built on the CEER1 interface. High-level experts must be paired with the low-level checkpoint they were trained with.

## Usage guide

| Feature | Documentation |
|---|---|
| Try compliance in your browser | [Live Demo — no installation](https://ee-root-compliance.github.io/live-demo/) |
| Set up Isaac Lab and prepare motions | [Installation](docs/installation.md) · [Dataset](docs/data.md) |
| Train the compliant low-level controller | [CEER1 / compliance training](docs/ceer1.md) |
| Add directional EE and root compliance | [CEER2 / high-level training & MoE](docs/ceer2.md) |
| Control the robot interactively | [Keyboard teleoperation & scenes](docs/teleoperation.md) |
| Evaluate stiffness or compare baselines | [Evaluation](docs/evaluation.md) |

## Quick start

After [installation](docs/installation.md) and [data preparation](docs/data.md), check the training pipeline on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  task=G1/G1_3kp_stiff +exp=train \
  wandb.mode=disabled task.num_envs=256 total_frames=40960
```

For full training and evaluation, configure your W&B entity and checkpoint paths in the linked guides.

<details>
<summary>Code map</summary>

| Path | Purpose |
|---|---|
| `train.sh` | Low-level teacher → adapt → finetune |
| `train_hl_teacher_student.sh` | High-level experts and baselines |
| `cfg/moe/` | Analytical MoE expert manifest |
| `scripts/eval_manipulation.py` | Interactive control and compliance evaluation |
| `teleop_dummy_pub.py` | Keyboard / motion pose publisher |
| `active_adaptation/` | Environments, policies, and learning algorithms |
| `docs/` | English and Chinese usage guides |

</details>

## Papers

- **CEER (CEER1):** [Compliant End-Effector and Root Control as a Unified Interface for Hierarchical Humanoid Loco-Manipulation](https://arxiv.org/abs/2605.19981).
- **CEER2:** [Directional and Tunable End-Effector and Root Compliance for Humanoid Loco-Manipulation](https://arxiv.org/abs/2609.38709).

## Acknowledgments

This codebase builds on [GentleHumanoid](https://github.com/Axellwppr/gentle-humanoid-training). We thank its authors for open-sourcing their implementation, which provided the starting point for CEER's low-level controller.

Code: [MIT License](LICENSE). Motion datasets retain their original licenses. [Image source](docs/assets/README.md).
