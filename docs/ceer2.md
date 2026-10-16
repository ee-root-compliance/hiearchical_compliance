[← README](../README.md)

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

