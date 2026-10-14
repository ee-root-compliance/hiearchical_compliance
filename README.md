# Hierarchical Humanoid Compliance

This repository provides Isaac Lab training and evaluation code for a
two-layer humanoid compliance controller. A shared low-level whole-body policy
tracks motion references, while a high-level policy generates residual
end-effector or root commands. The repository also includes fixed-compliance,
ranged-compliance, force-estimator, and analytical MoE baselines.

## Repository layout

```text
active_adaptation/       Environment, commands, observations, policies, and utilities
cfg/                     Hydra task, algorithm, evaluation, object, and MoE configs
scripts/train.py         Training entry point
scripts/eval_manipulation.py
                         Manipulation, EE-compliance, and root-compliance evaluation
scripts/data_process/    Motion conversion and dataset allowlists
train.sh                 Low-level training launcher
train_hl_teacher_student.sh
                         High-level teacher-student launcher
teleop_dummy_pub.py      Dummy UDP pose publisher for interactive testing
```

## Workflow

1. [Install](#installation) Isaac Sim, Isaac Lab, and this repository.
2. [Convert the motion dataset](#dataset).
3. Run the [quick smoke test](#quick-smoke-test).
4. Train the shared [low-level policy](#low-level-training).
5. Train the [high-level experts and baselines](#high-level-teacher-student-training).
6. Evaluate the [analytical MoE](#analytical-moe) and the [baselines](#baselines-and-evaluation).

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

## Dataset

The motion datasets are not redistributed in this repository. Obtain them from
their official distribution channels, such as [AMASS](https://amass.is.tue.mpg.de/),
and follow their licenses.

### Input format

Each input `.npz` must be a motion clip **already retargeted to the Unitree G1**
and sampled at **50 Hz**. Raw AMASS SMPL files are not accepted, and the
converter does not resample.

A G1-retargeted file contains:

| Key | Shape | Notes |
|---|---|---|
| `joint_names` | (J,) | Must include all 29 G1 joints, in any order |
| `body_names` | (B,) | G1 links plus `head_mimic`, `left_hand_mimic`, `right_hand_mimic` |
| `root_pos` | (T, 3) | World position, metres |
| `root_rot` | (T, 4) | World orientation, quaternion **xyzw** |
| `dof_pos` | (T, J) | Radians, in `joint_names` order |
| `local_body_pos` | (T, B, 3) | Body positions in the root frame |
| `local_body_rot` | (T, B, 4) | Body orientations relative to the root, **xyzw** |
| `mocap_framerate`, `frequency`, or `fps` | scalar | Must be 50 |

The LIMMT keypoint format (`qpos`, `qvel`, `kpt2gv_pose`, `gv2wrd_pose`) is also
accepted, but lacks the head and wrist bodies used by the 3-keypoint tasks.

### Directory layout

Allowlist paths are relative to `--dataset-root`, e.g.
`<dataset-root>/ACCAD/Female1General_c3d/A14_-_stand_to_skip_stageii.npz`.
The root should contain only motion `.npz` files.

### Conversion

Create the dataset used by all default task configs:

```bash
python scripts/data_process/generate_dataset.py \
  --dataset-root /path/to/G1_retargeted_motions \
  --allowlist scripts/data_process/allowlist_limmt_no_foot_compliance.json \
  --mem-path dataset/limmt_no_foot_compliance_full
```

Or use the wrapper:

```bash
DATASET_ROOT=/path/to/G1_retargeted_motions bash generate_limmt_amass_full_dataset.sh
```

Datasets are loaded from `<repository>/dataset/`. To store them elsewhere, set
`MEMPATH=/path/to/datasets`.

## Quick smoke test

A one-minute single-GPU run that checks the installation and dataset:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  task=G1/G1_3kp_stiff +exp=train \
  wandb.mode=disabled task.num_envs=256 total_frames=40960
```

It finishes with `Saved checkpoint to .../checkpoint_final.pt`.

## Low-level training

`train.sh` runs the low-level teacher-student pipeline:

```text
train       4B environment frames
adapt       1B environment frames
finetune    2B environment frames
```

All settings are environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `WANDB_ENTITY` | (required) | W&B user or team |
| `LOW_WANDB_PROJECT` | `your-low-level-project` | W&B project for low-level runs |
| `PIPELINE` | `stiff30` | One of `stiff30`, `fixed_ee`, `range_100_600`, `5kp` |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3` | GPUs to use |
| `NPROC` | `4` | Processes per node; must match the number of GPUs |
| `MASTER_PORT` | `29501` | `torchrun` port |
| `RUN_TIMESTAMP` | current time | Suffix of the W&B run ids |

```bash
# Shared 3-keypoint stiff backbone; maximum external force is 30 N.
PIPELINE=stiff30 LOW_WANDB_PROJECT=my-low-level bash train.sh

# End-to-end fixed 200xyz compliance baseline.
PIPELINE=fixed_ee LOW_WANDB_PROJECT=my-low-level bash train.sh

# End-to-end independently ranged xyz compliance baseline over 100--600 N/m.
PIPELINE=range_100_600 LOW_WANDB_PROJECT=my-low-level bash train.sh

# Optional 5-keypoint locomotion policy.
PIPELINE=5kp LOW_WANDB_PROJECT=my-low-level bash train.sh

# Single GPU:
CUDA_VISIBLE_DEVICES=0 NPROC=1 PIPELINE=stiff30 LOW_WANDB_PROJECT=my-low-level bash train.sh
```

The corresponding task configs are:

```text
cfg/task/G1/G1_3kp_stiff.yaml
cfg/task/G1/G1_3kp_ee_net_pull_force_b.yaml
cfg/task/G1/G1_3kp_ee_xyz_range_100_600.yaml
cfg/task/G1/G1_5kp.yaml
```

When the pipeline finishes, it prints the run path for high-level training:

```text
LOW_RUN_PATH=<entity>/<project>/compliance_3kp_stiff_finetune_limmt_full_force30_<timestamp>
```

To start from a local file instead, pass `checkpoint_path=/path/to/checkpoint.pt`
to `scripts/train.py`.

## High-level teacher-student training

`train_hl_teacher_student.sh` trains high-level EE policies on top of a frozen
low-level checkpoint, in two stages:

```text
teacher     root_student_force_ppo
adapt       root_student_force_ppo_adapt
```

| Variable | Default | Meaning |
|---|---|---|
| `WANDB_ENTITY` | (required) | W&B user or team |
| `HL_WANDB_PROJECT` | `your-high-level-project` | W&B project for high-level runs |
| `LOW_RUN_PATH` | (required, or `LOW_CHECKPOINT_PATH`) | Printed at the end of `train.sh` |
| `LOW_CHECKPOINT_PATH` | | Local low-level checkpoint instead of `LOW_RUN_PATH` |
| `RUN_ONLY` | `all` | `experts`, `baselines`, or `all` |
| `CUDA_VISIBLE_DEVICES` / `NPROC` / `MASTER_PORT` | `0,1,2,3` / `4` / `29502` | As above |
| `TEACHER_FRAMES` / `ADAPT_FRAMES` | `4000_000_000` / `1000_000_000` | Frames per stage |

Train the ten high-level experts used by the 100--600 N/m analytical MoE:

```bash
export LOW_RUN_PATH=<entity>/<project>/compliance_3kp_stiff_finetune_limmt_full_force30_<timestamp>
export HL_WANDB_PROJECT=my-high-level
RUN_ONLY=experts bash train_hl_teacher_student.sh
```

The ten expert targets are:

```text
100x, 100y, 100z,
200x, 200y, 200z,
400x, 400y, 400z,
600xyz.
```

Train the matched high-level baselines, which are the fixed 200xyz high-level
policy and the 100--600 ranged high-level policy:

```bash
RUN_ONLY=baselines bash train_hl_teacher_student.sh
```

The corresponding task configs are under `cfg/task/G1/hl/ee/`.

Root-compliance tasks are under `cfg/task/G1/hl/root/`; uncomment the
corresponding lines in `train_hl_teacher_student.sh` to train them.

## Analytical MoE

The analytical MoE is training-free. It freezes the ten expert checkpoints,
interpolates their residual actions in physical displacement space, and
composes the x, y, and z components independently. The manifest is:

```text
cfg/moe/G1_ee_analytical_100_200_400_600_xyz600_direct_posscale035.yaml
```

Fill the ten `experts.<name>` entries before launching. Each entry accepts:

```yaml
checkpoint_path: /path/to/checkpoint.pt          # local file
checkpoint_path: run:entity/project/run_id       # W&B, latest checkpoint
checkpoint_path: run:entity/project/run_id:2000  # W&B, checkpoint_2000.pt
run_path: entity/project/run_id                  # W&B, latest checkpoint
```

Then evaluate a commanded stiffness, passing the low-level policy the experts
were trained with:

```bash
python scripts/eval_manipulation.py \
  --full_collision \
  --ee-compliance-eval \
  --ee-compliance-num-envs 8 \
  --moe-experts-config \
    cfg/moe/G1_ee_analytical_100_200_400_600_xyz600_direct_posscale035.yaml \
  --low-run-path <entity>/<low-level-project>/<low_level_finetune_run> \
  --ee-compliance-stiffness 300 450 600
```

Use `--low-checkpoint /path/to/low_level.pt` for a local low-level file.

## Baselines and evaluation

Evaluate a W&B policy:

```bash
python scripts/eval_manipulation.py \
  --full_collision \
  --ee-compliance-eval \
  --ee-compliance-num-envs 8 \
  --run_path <entity>/<project>/<run_id> \
  --ee-compliance-stiffness 200 200 200
```

Evaluate a local checkpoint with its training config (`cfg.yaml`, saved next to
the checkpoint):

```bash
python scripts/eval_manipulation.py \
  --full_collision \
  --ee-compliance-eval \
  --ee-compliance-num-envs 8 \
  --checkpoint /path/to/checkpoint.pt \
  --config-file /path/to/cfg.yaml \
  --ee-compliance-stiffness 200 200 200
```

The principal baseline modes are:

| Baseline | Training or evaluation entry point |
|---|---|
| Analytical MoE | `--moe-experts-config cfg/moe/...yaml --low-run-path ...` |
| Stiff.-conditioned high-level | `RUN_ONLY=baselines bash train_hl_teacher_student.sh`, ranged HL task |
| Stiff.-conditioned end-to-end | `PIPELINE=range_100_600 bash train.sh` |
| Fixed end-to-end | `PIPELINE=fixed_ee bash train.sh` |
| Raw stiff low-level | `PIPELINE=stiff30 bash train.sh` |
| Estimated-force analytical compliance | Add `--ee-compliance-force-estimator-ablation` when evaluating a **high-level** EE policy |
| Oracle-force analytical compliance | Add `--ee-compliance-oracle-force` when evaluating the **stiff low-level** checkpoint |

Both force modes set the EE target to force / commanded stiffness; the oracle
mode uses the applied force, the estimator mode uses the policy's force
estimate.

Example oracle-force evaluation:

```bash
python scripts/eval_manipulation.py \
  --full_collision \
  --ee-compliance-eval \
  --ee-compliance-num-envs 8 \
  --run_path <entity>/<low-level-project>/<stiff_low_level_finetune_run> \
  --ee-compliance-oracle-force \
  --ee-compliance-stiffness 300 450 600
```

### Bimanual evaluation

Use `--ee-bimanual-compliance-eval` instead of `--ee-compliance-eval`, with six
stiffness values (`left_xyz right_xyz`) and a per-EE task:

```bash
python scripts/eval_manipulation.py \
  --full_collision \
  --ee-bimanual-compliance-eval \
  --ee-compliance-num-envs 8 \
  --moe-experts-config \
    cfg/moe/G1_ee_analytical_100_200_400_600_xyz600_direct_posscale035.yaml \
  --task G1/hl/ee/G1_hl_ee_bimanual_analytical_moe_200_600_force_b_student \
  --low-run-path <entity>/<low-level-project>/<low_level_finetune_run> \
  --ee-compliance-stiffness 200 400 600 600 400 200
```

## Dummy teleoperation and interactive environments

`teleop_dummy_pub.py` publishes root, head, and hand poses over UDP. Run the
simulator and the publisher in separate terminals.

### Live teleoperation with `eval_manipulation.py`

Start the policy first (`-p` = interactive mode):

```bash
python scripts/eval_manipulation.py \
  --run_path <entity>/<project>/<run_id> \
  --full_collision \
  --num_envs 1 \
  --obs_source udp \
  --objects cfg/objects/table_scene.yaml \
  -p
```

Then start the dummy publisher in a second terminal:

```bash
python teleop_dummy_pub.py \
  --mode 0 \
  --dst_ip 127.0.0.1 \
  --dst_port 15000 \
  --hz 30
```

Packet format (`G6D1`): 4-byte magic, `uint32` sequence number, then four
world-frame poses (root, head, left hand, right hand) as `x,y,z, qx,qy,qz,qw`
little-endian `float32`.

The keyboard controls in mode 0 are:

```text
Hands: I/K = x +/-, J/L = y apart/together, U/O = z +/-
Root:  W/S = x +/-, A/D = y +/-, F/H = z +/-, Q/E = yaw +/-
Exit:  ESC or Ctrl+C
```

### Available object environments

Pass one of the following files through `--objects` in interactive play mode:

| Scene | Contents |
|---|---|
| `cfg/objects/controller_test.yaml` | Minimal dynamic-box collision check |
| `cfg/objects/boxes_scene.yaml` | Platform and dynamic cylinder |
| `cfg/objects/table_scene.yaml` | Table, box, and sphere |
| `cfg/objects/room_scene.yaml` | Table, cylinder, and bed |
| `cfg/objects/wall_scene.yaml` | Front and side wall obstacles |

Objects are loaded only in interactive mode (`-p`).

### Dummy publisher modes

| Mode | Command | Purpose |
|---|---|---|
| `0` (default) | `python teleop_dummy_pub.py --mode 0` | Keyboard-controlled fixed pose and root motion |
| `1` | `python teleop_dummy_pub.py --mode 1` | VR pose input; requires a local `vr` (OpenVR) module that is **not** included in this repository |
| `2` | `python teleop_dummy_pub.py --mode 2` | Random hand pose every five seconds for smoke testing |
| `3` | `python teleop_dummy_pub.py --mode 3` | Receives root-state CSV on UDP port `15001` and republishes a pose packet to port `15000` |
| `4` | `python teleop_dummy_pub.py --mode 4 --motion-file /path/to/motion.npz` | Replays a motion with optional foot positions |
| `5` | `python teleop_dummy_pub.py --mode 5 --motion-file /path/to/motion.npz` | Replays a motion and appends joint targets |

Mode 3 expects CSV lines `timestamp,x,y,z,qw,qx,qy,qz` on UDP port `15001`.

### Using a local checkpoint

With a local checkpoint:

```bash
python scripts/eval_manipulation.py \
  --checkpoint /path/to/checkpoint.pt \
  --config-file /path/to/cfg.yaml \
  --full_collision \
  --obs_source udp \
  --objects cfg/objects/controller_test.yaml \
  -p
```

Do not combine teleoperation with `--ee-compliance-eval` or `--root-compliance-eval`.

## Evaluation protocol and outputs

The EE compliance evaluator applies forces of 5, 10, 15, 20, and 30 N along
`+/-x`, `+/-y`, `+/-z` and computes the apparent stiffness from the steady-state
displacement.

Other evaluation modes:

```bash
# EE tracking with scripted target poses.
python scripts/eval_manipulation.py \
  --run_path <entity>/<project>/<run_id> \
  --ee-tracking-eval \
  --num_envs 8

# Root compliance evaluation.
python scripts/eval_manipulation.py \
  --run_path <entity>/<project>/<run_id> \
  --root-compliance-eval \
  --root-compliance-num-envs 8
```

Useful options:

| Option | Meaning |
|---|---|
| `--seed N` | Override the evaluation seed |
| `-i N` | Evaluate `checkpoint_N.pt` of a W&B run instead of the latest |
| `--ee-output PATH` / `--root-output PATH` | Explicit report path |

Reports are written as JSON under `outputs/`. If Isaac Sim hangs on exit after
printing the results, press Ctrl+C; the report is already saved.

## Reproducibility notes

- Compare policies with the same task config, low-level checkpoint, number of
  environments, and stiffness command.
- Set the evaluation seed with `--seed` and the training seed with `seed=N`.
- Keep each high-level checkpoint paired with the low-level checkpoint it was
  trained on.

## License and external data

Code: [MIT License](LICENSE). Motion datasets remain subject to their original
licenses.
