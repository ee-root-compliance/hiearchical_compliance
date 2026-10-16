[← README](../README.md)

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

